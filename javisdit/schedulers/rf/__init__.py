import math
import os
import sys
import time
from typing import Optional, Callable
import torch
from torch import Tensor
from tqdm import tqdm
import numpy as np
import torch.nn as nn
import torch.optim as optim


from javisdit.registry import SCHEDULERS
from .rectified_flow import RFlowScheduler, timestep_transform
from .reward_util import to_tensor_or_zeros, compute_rank_based_rewards


@SCHEDULERS.register_module("rflow")
class RFLOW:
    def __init__(
        self,
        num_sampling_steps=10,
        num_timesteps=1000,
        cfg_scale=4.0,
        use_discrete_timesteps=False,
        use_timestep_transform=False,
        **kwargs,
    ):
        self.num_sampling_steps = num_sampling_steps
        self.num_timesteps = num_timesteps
        self.cfg_scale = cfg_scale
        self.use_discrete_timesteps = use_discrete_timesteps
        self.use_timestep_transform = use_timestep_transform

        self.scheduler = RFlowScheduler(
            num_timesteps=num_timesteps,
            num_sampling_steps=num_sampling_steps,
            use_discrete_timesteps=use_discrete_timesteps,
            use_timestep_transform=use_timestep_transform,
            **kwargs,
        )
        self.reward_zscore_history = {}

    def _history_zscore(self, scores, name):
        if scores.numel() == 0:
            return scores

        history = self.reward_zscore_history.get(name, [])
        history_tensors = [tensor.to(scores.device) for tensor in history]
        all_scores = torch.cat(history_tensors + [scores.detach()], dim=0) if history_tensors else scores.detach()

        score_std = all_scores.std(unbiased=False)
        if torch.isfinite(score_std) and score_std > 0:
            z_scores = (scores - all_scores.mean()) / score_std
        else:
            print(f"Warning: {name} history scores have no variance. Using raw scores.")
            z_scores = scores

        self.reward_zscore_history.setdefault(name, []).append(scores.detach().cpu())
        return z_scores

    def sample(
        self,
        model,
        text_encoder,
        z,
        prompts,
        device,
        additional_args=None,
        mask=None,
        guidance_scale=None,
        progress=True,
    ):
        # if no specific guidance scale is provided, use the default scale when initializing the scheduler
        if guidance_scale is None:
            guidance_scale = self.cfg_scale

        n = len(prompts)
        # text encoding
        ## y: shape(bs,1,300,4096), last hidden state
        model_args = text_encoder.encode(prompts)
        ## y_null: randn(bs,1,300,4096), seems repeated; null prompt for classifier-free guidance
        y_null = text_encoder.null(n)
        model_args["y"] = torch.cat([model_args["y"], y_null], 0)
        if additional_args is not None:
            model_args.update(additional_args)

        # prepare timesteps
        ## num_timesteps means diffusion/training steps; num_sampling_steps means denoising/inference steps
        timesteps = [(1.0 - i / self.num_sampling_steps) * self.num_timesteps for i in range(self.num_sampling_steps)]
        if self.use_discrete_timesteps:
            timesteps = [int(round(t)) for t in timesteps]
        timesteps = [torch.tensor([t] * z.shape[0], device=device) for t in timesteps]
        if self.use_timestep_transform:
            timesteps = [timestep_transform(t, additional_args, num_timesteps=self.num_timesteps) for t in timesteps]

        if mask is not None:
            noise_added = torch.zeros_like(mask, dtype=torch.bool)
            noise_added = noise_added | (mask == 1)

        progress_wrap = tqdm if progress else (lambda x: x)
        for i, t in enumerate(progress_wrap(timesteps)):
            # mask for adding noise
            if mask is not None:
                mask_t = mask * self.num_timesteps
                x0 = z.clone()
                x_noise = self.scheduler.add_noise(x0, torch.randn_like(x0), t)

                mask_t_upper = mask_t >= t.unsqueeze(1)
                model_args["x_mask"] = mask_t_upper.repeat(2, 1)
                mask_add_noise = mask_t_upper & ~noise_added

                z = torch.where(mask_add_noise[:, None, :, None, None], x_noise, x0)
                noise_added = mask_t_upper

            # classifier-free guidance
            z_in = torch.cat([z, z], 0)
            t = torch.cat([t, t], 0)
            pred = model(z_in, t, **model_args).chunk(2, dim=1)[0]
            pred_cond, pred_uncond = pred.chunk(2, dim=0)
            v_pred = pred_uncond + guidance_scale * (pred_cond - pred_uncond)

            # update z
            dt = timesteps[i] - timesteps[i + 1] if i < len(timesteps) - 1 else timesteps[i]
            dt = dt / self.num_timesteps
            z = z + v_pred * dt[:, None, None, None, None]

            if mask is not None:
                z = torch.where(mask_t_upper[:, None, :, None, None], z, x0)

        return z
    
    def multimodal_sample(
        self,
        model,
        text_encoder,
        latent_dict,
        prompts,
        device,
        additional_args=None,
        mask=None,
        prior_encoder=None,
        guidance_scale=None,
        progress=True,
        return_attn_map=False,
        verifier=None,
        evosearch=None,
        vae=None,
        audio_vae=None,
        num_frames=None,
        dtype=None,
    ):
        if os.environ.get("WALL_CLOCK", "false") == "true":
            torch.cuda.synchronize()  # 이전 GPU 작업 완료 대기
            start_time = time.time()

        modal_keys = list(latent_dict.keys())  # e.g., ['video', 'audio']

        # if no specific guidance scale is provided, use the default scale when initializing the scheduler
        if guidance_scale is None:
            guidance_scale = self.cfg_scale

        n = latent_dict[modal_keys[0]].shape[0]
        # text encoding
        ## y: shape(bs,1,300,4096), last hidden state;
        if isinstance(prompts, dict):
            model_args = prompts
            prompts = prompts.pop("prior_emb")  # compatible for prior_encoder.encode(prompts)
        else:
            model_args = text_encoder.encode(prompts)

        ## y_null: randn(bs,1,300,4096), seems repeated; null prompt for classifier-free guidance
        y_null = text_encoder.null(n)
        model_args["y"] = torch.cat([model_args["y"], y_null], 0)

        # spatio-temporal prior encoding
        if prior_encoder is not None:
            prior_dict = prior_encoder.encode(prompts)
            # model_args.update({k: torch.cat((v, v), dim=0) for k, v in prior_dict.items()})
            spatial_prior, temporal_prior = prior_dict['spatial_prior'], prior_dict['temporal_prior']
            if getattr(prior_encoder.st_prior_embedder, 'y_embedding', None) is not None:
                null_spatial_prior, null_temporal_prior = prior_encoder.null(n)
            else:
                null_spatial_prior = torch.randn_like(spatial_prior)
                null_temporal_prior = torch.randn_like(temporal_prior)
            prior_null_dict = {
                'spatial_prior': null_spatial_prior, 
                'temporal_prior': null_temporal_prior, 
            }
            model_args.update({  # compatible for onset prediction
                k: torch.cat((v, prior_null_dict.get(k, torch.randn_like(v))), dim=0) \
                    for k, v in prior_dict.items()
            })
        if additional_args is not None:
            model_args.update(additional_args)
        model_args['num_timesteps'] = self.num_timesteps
        model_args['return_attn_map'] = return_attn_map

        # prepare timesteps
        ## num_timesteps means diffusion/training steps; num_sampling_steps means denoising/inference steps
        timesteps = [(1.0 - i / self.num_sampling_steps) * self.num_timesteps for i in range(self.num_sampling_steps)]
        if self.use_discrete_timesteps:
            timesteps = [int(round(t)) for t in timesteps]
        
        raw_timesteps = timesteps
        timesteps = [torch.tensor([t] * n, device=device) for t in timesteps]
        if self.use_timestep_transform:
            timesteps = [timestep_transform(t, additional_args, num_timesteps=self.num_timesteps) for t in timesteps]

        if mask is not None:
            noise_added = {}
            for k, m in mask.items():
                noise_added[k] = torch.zeros_like(m, dtype=torch.bool)
                noise_added[k] = noise_added[k] | (m == 1)

        # EvoSearch Initialization
        if verifier is not None:
            self.best_reward=-100
            self.best_video=None
            self.best_audio=None
            self.video_list=[]
            self.audio_list=[]
            generation_steps = 0
            self.population_audio_list=None
            self.population_video_list=None
            self.rewards_list=None

            evolution_schedule = evosearch.get('evolution_schedule', [0])
            population_size_schedule=evosearch.get('population_size_schedule', [1])
            self.population_audio_list = [[] for _ in evolution_schedule]
            self.population_video_list = [[] for _ in evolution_schedule]
            self.rewards_list = [[] for _ in evolution_schedule]

            # 현재 프롬프트의 리워드만 추적 (재평가용)
            if evosearch.score_method == 'adaptive':
                self.current_prompt_score_history = {}

        attn_map_dict = {}
        progress_wrap = tqdm if progress else (lambda x: x)
        for i, t in enumerate(progress_wrap(timesteps)):
            z_in = {}
            for k, z in latent_dict.items():
                # mask for adding noise (필요x)
                if mask is not None:
                    mask_t = mask[k] * self.num_timesteps
                    x0 = z.clone()
                    x_noise = self.scheduler.add_noise(x0, torch.randn_like(x0), t)

                    mask_t_upper = mask_t >= t.unsqueeze(1)
                    model_args["x_mask" if k == 'video' else 'ax_mask'] = mask_t_upper.repeat(2, 1)
                    mask_add_noise = mask_t_upper & ~noise_added[k]

                    if k == 'video':
                        mask_add_noise = mask_add_noise[:, None, :, None, None]
                    elif k == 'audio':
                        mask_add_noise = mask_add_noise[:, None, :, None]
                    assert len(mask_add_noise.shape) == len(x_noise.shape) == len(x0.shape)

                    z = torch.where(mask_add_noise, x_noise, x0)
                    noise_added[k] = mask_t_upper

                # classifier-free guidance
                z_in[k] = torch.cat([z, z], 0)
            t_in = torch.cat([t, t], 0)

            # EvoSearch
            if verifier is not None and i in evolution_schedule:
                latents_audio_new, latents_video_new = self.evosearch(
                    model,
                    model_args,
                    text_encoder,
                    latent_dict,
                    prompts=prompts,
                    device=device,
                    dtype=dtype,
                    num_inference_steps=self.num_sampling_steps,
                    timesteps=raw_timesteps,
                    num_samples_per_prompt=population_size_schedule[generation_steps],
                    verifier=verifier,
                    seed=additional_args.get('seed', 42) if additional_args else 42,
                    evolution_schedule=evolution_schedule,
                    population_size_schedule=population_size_schedule,
                    elite_size=evosearch.get('elite_size', 1),
                    mutation_rate=evosearch.get('mutation_rate', 0.1),
                    guidance_reward=evosearch.get('guidance_reward', "VideoReward"),
                    generation_steps=generation_steps,
                    video_fps=additional_args.get('fps', 24.0) if additional_args else 24.0,
                    tournament_ratio=evosearch.get('tournament_ratio', 0.9),
                    stage_verifiers=evosearch.get('stage_verifiers', None),
                    stage_weights=evosearch.get('stage_weights', None),
                    score_method=evosearch.get('score_method', 'rank'),
                    javis_boost=evosearch.get('javis_boost', 0.5),
                    vae=vae,
                    audio_vae=audio_vae,
                    num_frames=num_frames,
                    guidance_scale=guidance_scale,
                    mask=mask,
                    noise_added=noise_added if mask is not None else None,
                    evosearch_config=evosearch,
                    VQ_mean=evosearch.get('VQ_mean', None), VQ_std=evosearch.get('VQ_std', None),
                    JS_mean=evosearch.get('JS_mean', None), JS_std=evosearch.get('JS_std', None),
                    )

                # Update latent_dict with successful evosearch result
                latent_dict['audio'] = latents_audio_new
                latent_dict['video'] = latents_video_new

                generation_steps += 1
                print('Updated best reward',self.best_reward)

                # Refresh z_in with optimized latents for the subsequent denoise step
                for k, z in latent_dict.items():
                    z_in[k] = torch.cat([z, z], 0)

            ###################################################################
            # Model Forward: seqeuce processing or batch processing
            ###################################################################
            pop_size = latent_dict['video'].shape[0]
            sequential_processing = evosearch.get('sequential_processing', False) if evosearch else False
            
            if sequential_processing and pop_size > 1:
                mm_pred_collected = {}
                for p in range(pop_size):
                    # Extract single population member with CFG (cond + uncond)
                    # z_in shape: {k: [cond0, cond1, ..., condN, uncond0, uncond1, ..., uncondN]}
                    z_in_p = {}
                    for k, v in z_in.items():
                        # v shape: [2*pop_size, ...] -> take [p] (cond) and [pop_size + p] (uncond)
                        z_in_p[k] = torch.cat([v[p:p+1], v[pop_size + p:pop_size + p + 1]], dim=0)
                    
                    # Prepare model_args for single population
                    model_args_p = {}
                    for key, value in model_args.items():
                        if isinstance(value, torch.Tensor):
                            if value.shape[0] == 2 * pop_size:
                                # CFG duplicated: [cond_all, uncond_all]
                                model_args_p[key] = torch.cat([value[p:p+1], value[pop_size + p:pop_size + p + 1]], dim=0)
                            elif value.shape[0] == pop_size:
                                # Not CFG duplicated
                                model_args_p[key] = value[p:p+1]
                            else:
                                # Other shapes (e.g., shared parameters)
                                model_args_p[key] = value
                        else:
                            model_args_p[key] = value
                    
                    t_in_p = torch.cat([t[p:p+1], t[p:p+1]], dim=0)
                    
                    # Forward pass for single population member
                    pred_p = model(z_in_p, t_in_p, **model_args_p)
                    
                    # Collect predictions - each pred_p[k] has shape [2, ...]  (cond, uncond)
                    for k, v in pred_p.items():
                        if k not in mm_pred_collected:
                            mm_pred_collected[k] = {'cond': [], 'uncond': []}
                        if isinstance(v, torch.Tensor):
                            mm_pred_collected[k]['cond'].append(v[0:1])   # cond part
                            mm_pred_collected[k]['uncond'].append(v[1:2]) # uncond part
                        else:
                            mm_pred_collected[k] = v  # For non-tensor (e.g., attn_maps)
                
                # Concatenate all predictions to match batch format: [cond_all, uncond_all]
                mm_pred = {}
                for k, v_dict in mm_pred_collected.items():
                    if isinstance(v_dict, dict):
                        # Concatenate: [cond0, cond1, ..., condN, uncond0, uncond1, ..., uncondN]
                        cond_cat = torch.cat(v_dict['cond'], dim=0)
                        uncond_cat = torch.cat(v_dict['uncond'], dim=0)
                        mm_pred[k] = torch.cat([cond_cat, uncond_cat], dim=0)
                    else:
                        mm_pred[k] = v_dict  # For non-tensor
            else:
                # Original batch processing
                mm_pred = model(z_in, t_in, **model_args)

            attn_maps = mm_pred.pop('attn_maps', None)
            if i in np.linspace(0, len(timesteps)-1, 4, dtype=int):
                attn_map_dict[f'step{i}'] = attn_maps
            

            ###################################################################
            # Update z
            ###################################################################
            for k, pred in mm_pred.items():
                pred = pred.chunk(2, dim=1)[0]
                pred_cond, pred_uncond = pred.chunk(2, dim=0)
                v_pred = pred_uncond + guidance_scale * (pred_cond - pred_uncond)

                # update z
                z = latent_dict[k]
                x0 = z.clone()
                dt = timesteps[i] - timesteps[i + 1] if i < len(timesteps) - 1 else timesteps[i]
                dt = dt / self.num_timesteps
                z = z + v_pred * dt.view(-1, *([1] * (len(z.shape) - 1)))

                if mask is not None: #(필요x)
                    if k == 'video':
                        mask_t_upper = noise_added[k][:, None, :, None, None]
                    elif k == 'audio':
                        mask_t_upper = noise_added[k][:, None, :, None]
                    assert len(mask_t_upper.shape) == len(z.shape) == len(x0.shape), f'{mask_t_upper.shape} {z.shape} {x0.shape}'
                    z = torch.where(mask_t_upper, z, x0)
                
                latent_dict[k] = z

        if return_attn_map:
            latent_dict['attn_map'] = attn_map_dict
        
        ###################################################################
        # Decode video and audio and Calculate rewards
        ###################################################################
        if vae is not None and audio_vae is not None:
            video_samples = latent_dict['video']
            audio_samples = latent_dict['audio']
            
            # Get original_waveform_length from audio latent shape
            audio_length_in_s = num_frames / 24 if num_frames is not None else 4.0  # default fps=24
            # Calculate original_waveform_length (assuming 16kHz)
            original_waveform_length = int(audio_length_in_s * 16000)
            
            # Decode latents to actual video and audio
            video = vae.decode(video_samples.to(dtype), num_frames=num_frames)
            audio = audio_vae.decode_audio(audio_samples, original_waveform_length=original_waveform_length)

            # Calculate rewards and update best results if verifier is provided
            if verifier is not None and len(video) > 1:
                # Get evosearch parameters
                stage_verifiers = evosearch.get('stage_verifiers', [['VideoReward']])
                stage_weights = evosearch.get('stage_weights', [[1.0]])
                score_method = evosearch.get('score_method', 'weighted')
                
                # Add to global lists and prepare for reward calculation
                audio_list_for_eval = []
                video_list_for_eval = []
                for m in range(len(video)):
                    self.audio_list.append(audio[m])
                    self.video_list.append(video[m])
                    audio_list_for_eval.append(audio[m])
                    video_list_for_eval.append(video[m])
                
                # Calculate rewards using unified method with stage-specific verifiers
                rewards, vqa_rewards, javis_scores = self._calculate_rewards(
                    model, audio_list_for_eval, video_list_for_eval, prompts[0], verifier, video_fps=24,
                    stage_verifiers=stage_verifiers, stage_weights=stage_weights, generation_steps=generation_steps,
                    score_method=score_method, device=device,
                    VQ_mean=evosearch.get('VQ_mean', None), VQ_std=evosearch.get('VQ_std', None), 
                    JS_mean=evosearch.get('JS_mean', None), JS_std=evosearch.get('JS_std', None),
                )

                # (TODO)Re-evaluate cached_rewards if adaptive
                if score_method == 'adaptive' and hasattr(self, 'current_prompt_score_history') and len(list(self.current_prompt_score_history.values())[0]) > 1:
                    weighter = model.weighter
                    
                    # 1. Collect all history for current prompt
                    current_prompt_all_scores = {}
                    for k, v_list in self.current_prompt_score_history.items():
                        current_prompt_all_scores[k] = torch.cat(v_list).to(device)
                    
                    # 2. Recalculate weighted scores for ALL generations with NEWEST weights
                    recalculated_all_rewards = weighter.get_weighted_score(current_prompt_all_scores, javis_boost=evosearch.get('javis_boost', 0.5))
                    
                    print(f"Re-evaluated {len(recalculated_all_rewards)} samples (Gen 0~{generation_steps}) with updated weights")
                    print("Recalculated rewards:", recalculated_all_rewards)
                    
                    # 3. Update rewards for selection
                    cached_rewards = recalculated_all_rewards

                    # 4. select
                    elite_rew, elite_indices = torch.topk(cached_rewards, 1)
                    self.best_reward = elite_rew[0]
                    ind = elite_indices[0]
                    self.best_video = self.video_list[ind]
                    self.best_audio = self.audio_list[ind]
                    print(f"Final best reward {self.best_reward}")

                    latent_dict['video_decoded'] = self.best_video.unsqueeze(0)
                    latent_dict['audio_decoded'] = self.best_audio.unsqueeze(0)

                else:
                    elite_rew, elite_indices = torch.topk(rewards, 1)
                    print(f"DEBUG: elite_indices={elite_indices}, elite_rew={elite_rew}")
                    print(f"DEBUG: best_reward before update={self.best_reward}")
                    
                    if elite_rew[0] > self.best_reward:
                        self.best_reward = elite_rew[0]
                        ind = elite_indices[0]
                        self.best_video = video[ind]
                        self.best_audio = audio[ind]
                        print(f"DEBUG: Updated best_video with index {ind}, score {self.best_reward}")
                    else:
                        print(f"DEBUG: No update. Best score {self.best_reward} >= current max {elite_rew[0]}")
                    print('Updated best reward', self.best_reward)
                    
                    # Return best samples
                    latent_dict['video_decoded'] = self.best_video.unsqueeze(0)
                    latent_dict['audio_decoded'] = self.best_audio.unsqueeze(0)

        if os.environ.get("WALL_CLOCK", "false") == "true":
            torch.cuda.synchronize()  # 모든 GPU 작업 완료 대기
            end_time = time.time()
            print(f"Wall-clock time: {end_time - start_time} seconds")
            sys.exit(0)

        return latent_dict
    
    def training_losses(self, model, x_start, model_kwargs=None, noise=None, mask=None, weights=None, t=None):
        return self.scheduler.training_losses(model, x_start, model_kwargs, noise, mask, weights, t)

    def multimodal_training_losses(self, model, x_start, model_kwargs=None, noise=None, mask=None, weights=None, t=None):
        return self.scheduler.multimodal_training_losses(model, x_start, model_kwargs, noise, mask, weights, t)

    @staticmethod
    def _convert_video_for_reward_model(video):
        """
        Convert video tensor to reward model compatible format.
        
        Args:
            video: torch.Tensor [C, T, H, W], float/bfloat16 (-1~1 or 0~1 range)
        
        Returns:
            torch.Tensor [C, T, H, W], uint8 (0-255 range)
        """
        # bfloat16이면 float32로 변환
        video = video.float()
        
        # 값 범위 확인 및 정규화
        if video.min() < 0 or video.max() > 1.1:  # [-1, 1] 범위
            video = (video + 1.0) / 2.0  # [-1, 1] → [0, 1]
        
        # [0, 1] → [0, 255] uint8
        video = (video * 255.0).clamp(0, 255).to(torch.uint8)
        
        return video
    
    @staticmethod
    def _convert_audio_for_reward_model(audio):
        """
        Convert audio tensor to reward model compatible format.
        
        Args:
            audio: torch.Tensor [T], float32 (-1~1 or large range)
        
        Returns:
            numpy.ndarray [T], int16 (-32768 ~ 32767 range)
        """
        if audio.dtype != torch.int16:
            # 값 범위 확인
            if audio.abs().max() <= 1.0:
                # [-1, 1] 범위 → [-32768, 32767]
                audio = (audio * 32767.0).clamp(-32768, 32767).to(torch.int16)
            else:
                # 이미 큰 범위면 그냥 int16으로 변환
                audio = audio.clamp(-32768, 32767).to(torch.int16)
        
        # torch.Tensor → numpy.ndarray 변환
        audio = audio.cpu().numpy()
        
        return audio


    def _calculate_rewards(self, model, audio_list, video_list, prompt, verifier, video_fps, 
                         stage_verifiers=None, stage_weights=None, generation_steps=None, score_method=None,
                         VQ_mean=None, VQ_std=None, JS_mean=None, JS_std=None, javis_boost=0.5, device=None):
        """Calculate rewards using stage-specific verifier combinations."""
        if not audio_list or not video_list:
            return tuple(torch.tensor([]).to(device) for _ in range(3))

        ###############################################
        # Settings
        ###############################################
        current_verifiers = []
        current_weights = []

        if stage_verifiers and stage_weights and generation_steps is not None and generation_steps < len(stage_verifiers):
            current_verifiers = stage_verifiers[generation_steps]
            current_weights = stage_weights[generation_steps]
            print(f"Score method: {score_method}")
            print(f"Stage {generation_steps}: Using verifiers {current_verifiers} with weights {current_weights}")
        
        verifier_flags = {
            'use_vr': "VideoReward" in current_verifiers,
            'use_vqa': "VQA" in current_verifiers,
            'use_align': "align" in current_verifiers,
            'use_javis': "JavisScore" in current_verifiers,
            'use_clap': "CLAP" in current_verifiers,
            'use_avh': "AVHScore" in current_verifiers,
            'use_avib': "AVIB" in current_verifiers,
        }
        
        weight_mapping = {v: current_weights[i] for i, v in enumerate(current_verifiers) if i < len(current_weights)}
        current_vr_weight = weight_mapping.get("VideoReward", 0.0)
        current_vqa_weight = weight_mapping.get("VQA", 0.0) 
        current_javis_weight = weight_mapping.get("JavisScore", 0.0)
        current_align_weight = weight_mapping.get("align", 0.0)
        current_clap_weight = weight_mapping.get("CLAP", 0.0)
        current_avh_weight = weight_mapping.get("AVHScore", 0.0)
        current_avib_weight = weight_mapping.get("AVIB", 0.0)
        
        ###############################################
        # Calculate rewards
        ###############################################
        vqa_reward_list = []
        align_list = []
        javis_list = []
        audio_reward_list = []
        avh_list = []
        avib_list = []

        with torch.no_grad():
            for audio, video in zip(audio_list, video_list):
                # Convert to reward model compatible format
                video = self._convert_video_for_reward_model(video)
                audio = self._convert_audio_for_reward_model(audio)
                
                if verifier_flags['use_vqa'] and verifier is not None:
                    vqa_reward = sum(verifier(video[:, i:i+1].permute(1, 0, 2, 3), prompt=prompt) for i in range(video.size(1))) / video.size(1)
                    # vqa_reward = sum(verifier(video[:, i:i+1], prompt=prompt) for i in range(video.size(1))) / video.size(1)
                    vqa_reward_list.append(vqa_reward)
                    
                elif verifier_flags['use_vr']:
                    score = verifier(video, audio, prompt=prompt)
                    ALL, VQ, MQ, TA, JS, AVH, CLAP, AVIB = score
                    vqa_reward_list.append(TA)

                    if verifier_flags['use_javis']:
                        javis_list.append(JS)
                    if verifier_flags['use_clap']:
                        audio_reward_list.append(CLAP)
                    if verifier_flags['use_align']:
                        align = av_align(audio, video, video_fps)
                        align_list.append(align)
                    if verifier_flags['use_avh']:
                        avh_list.append(AVH)
                    if verifier_flags['use_avib']:
                        avib_list.append(AVIB)
                else:
                    print("no vqa or vr verifier used")
                    vqa_reward_list.append(0.0)

        vqa_rewards = to_tensor_or_zeros(vqa_reward_list, len(audio_list), device)
        javis_scores = to_tensor_or_zeros(javis_list, len(audio_list), device)

        if verifier_flags['use_clap']:
            audio_scores = to_tensor_or_zeros(audio_reward_list, len(audio_list), device)
        if verifier_flags['use_align']:
            align_scores = to_tensor_or_zeros(align_list, len(audio_list), device)
        if verifier_flags['use_avh']:
            avh_scores = to_tensor_or_zeros(avh_list, len(audio_list), device)
        if verifier_flags['use_avib']:
            avib_scores = to_tensor_or_zeros(avib_list, len(audio_list), device)

        ###############################################
        # Calculate final rewards
        ###############################################
        if score_method == "rank":
            if verifier_flags['use_clap'] and verifier_flags['use_javis']:
                rewards = compute_rank_based_rewards(vqa_rewards, javis_scores, audio_scores)
            elif verifier_flags['use_align'] and verifier_flags['use_javis']:
                rewards = compute_rank_based_rewards(vqa_rewards, javis_scores, align_scores)
            elif verifier_flags['use_align']:
                rewards = compute_rank_based_rewards(vqa_rewards, align_scores)
                return rewards, vqa_rewards, align_scores
            elif verifier_flags['use_javis']:
                rewards = compute_rank_based_rewards(vqa_rewards, javis_scores)
            else:
                rewards = compute_rank_based_rewards(vqa_rewards)
                
        elif score_method == "weighted":
            if verifier_flags['use_vqa']:
                rewards = current_vqa_weight * vqa_rewards + current_javis_weight * javis_scores
            elif verifier_flags['use_vr']:
                if verifier_flags['use_javis']:
                    rewards = current_vr_weight * vqa_rewards + current_javis_weight * javis_scores
                elif verifier_flags['use_align']:
                    rewards = current_vr_weight * vqa_rewards + current_align_weight * align_scores
                    return rewards, vqa_rewards, align_scores
                elif verifier_flags['use_clap'] and verifier_flags['use_javis']:
                    rewards = current_vr_weight * vqa_rewards + current_javis_weight * javis_scores + current_clap_weight * audio_scores
                else:
                    rewards = current_vr_weight * vqa_rewards
            else:
                rewards = torch.zeros_like(vqa_rewards) if len(vqa_rewards) > 0 else torch.zeros(len(audio_list)).to(device)
            
            if verifier_flags['use_vr']:
                print("TA rewards", vqa_rewards)
            if verifier_flags['use_javis']:
                print("Javis rewards", javis_scores)
            print("Final rewards", rewards)
        
        elif score_method == "zscore":
            # Z-score normalization: (x - mean) / std
            if verifier_flags['use_vqa']:
                if VQ_mean is not None and VQ_std is not None:
                    vqa_z_scores = (vqa_rewards - VQ_mean) / VQ_std
                else:
                    print("Warning: VQ_mean or VQ_std not provided. Using raw VQA scores.")
                    vqa_z_scores = vqa_rewards
                if verifier_flags['use_javis'] and JS_mean is not None and JS_std is not None:
                    javis_z_scores = (javis_scores - JS_mean) / JS_std
                    rewards = current_vqa_weight * vqa_z_scores + current_javis_weight * javis_z_scores
                else:
                    rewards = vqa_z_scores                    
            elif verifier_flags['use_vr']:
                if VQ_mean is not None and VQ_std is not None:
                    vqa_z_scores = (vqa_rewards - VQ_mean) / VQ_std
                else:
                    print("Warning: VQ_mean or VQ_std not provided. Using raw scores for VQA.")
                    vqa_z_scores = vqa_rewards
                if verifier_flags['use_javis'] and JS_mean is not None and JS_std is not None:
                    javis_z_scores = (javis_scores - JS_mean) / JS_std
                else:
                    print("Warning: JS_mean or JS_std not provided. Using raw scores for Javis.")
                    javis_z_scores = javis_scores
                rewards = current_vr_weight * vqa_z_scores + current_javis_weight * javis_z_scores
            else:
                print("Warning: No VQA or VideoReward verifier used in zscore mode.")
                rewards = torch.zeros_like(vqa_rewards) if len(vqa_rewards) > 0 else torch.zeros(len(audio_list)).to(device)

            print("VQA rewards", vqa_rewards)
            print("VQA zscore rewards", vqa_z_scores)
            print("Javis rewards", javis_scores)
            print("Javis zscore rewards", javis_z_scores)
            print("Final rewards", rewards)

        elif score_method == "zscore_candidate":
            # Candidate-wise z-score normalization using only current population rewards.
            def _candidate_zscore(scores, name):
                if scores.numel() == 0:
                    return scores

                score_std = scores.std(unbiased=False)
                if torch.isfinite(score_std) and score_std > 0:
                    return (scores - scores.mean()) / score_std

                print(f"Warning: {name} candidate scores have no variance. Using raw scores.")
                return scores

            vqa_z_scores = vqa_rewards
            javis_z_scores = javis_scores
            if verifier_flags['use_vqa']:
                vqa_z_scores = _candidate_zscore(vqa_rewards, "VQA")
                if verifier_flags['use_javis']:
                    javis_z_scores = _candidate_zscore(javis_scores, "Javis")
                    rewards = current_vqa_weight * vqa_z_scores + current_javis_weight * javis_z_scores
                else:
                    rewards = vqa_z_scores

            elif verifier_flags['use_vr']:
                vqa_z_scores = _candidate_zscore(vqa_rewards, "VideoReward")
                if verifier_flags['use_javis']:
                    javis_z_scores = _candidate_zscore(javis_scores, "Javis")
                    rewards = current_vr_weight * vqa_z_scores + current_javis_weight * javis_z_scores
                else:
                    rewards = vqa_z_scores
            else:
                print("Warning: No VQA or VideoReward verifier used in zscore_candidate mode.")
                rewards = torch.zeros_like(vqa_rewards) if len(vqa_rewards) > 0 else torch.zeros(len(audio_list)).to(device)

            print("VQA rewards", vqa_rewards)
            print("VQA candidate zscore rewards", vqa_z_scores)
            print("Javis rewards", javis_scores)
            print("Javis candidate zscore rewards", javis_z_scores)
            print("Final rewards", rewards)

        elif score_method == "zscore_history":
            # History-aware z-score normalization using accumulated reward buffers.
            vqa_z_scores = vqa_rewards
            javis_z_scores = javis_scores
            if verifier_flags['use_vqa']:
                vqa_z_scores = self._history_zscore(vqa_rewards, "VQA")
                if verifier_flags['use_javis']:
                    javis_z_scores = self._history_zscore(javis_scores, "Javis")
                    rewards = current_vqa_weight * vqa_z_scores + current_javis_weight * javis_z_scores
                else:
                    rewards = vqa_z_scores

            elif verifier_flags['use_vr']:
                vqa_z_scores = self._history_zscore(vqa_rewards, "VideoReward")
                if verifier_flags['use_javis']:
                    javis_z_scores = self._history_zscore(javis_scores, "Javis")
                    rewards = current_vr_weight * vqa_z_scores + current_javis_weight * javis_z_scores
                else:
                    rewards = vqa_z_scores
            else:
                print("Warning: No VQA or VideoReward verifier used in zscore_history mode.")
                rewards = torch.zeros_like(vqa_rewards) if len(vqa_rewards) > 0 else torch.zeros(len(audio_list)).to(device)

            print("VQA rewards", vqa_rewards)
            print("VQA history zscore rewards", vqa_z_scores)
            print("Javis rewards", javis_scores)
            print("Javis history zscore rewards", javis_z_scores)
            print("Final rewards", rewards)

        elif score_method == "minmax":
            # Min-Max normalization: (x - min) / (max - min)
            if verifier_flags['use_vqa']:
                vqa_min = vqa_rewards.min()
                vqa_max = vqa_rewards.max()
                if vqa_max > vqa_min:
                    vqa_normalized = (vqa_rewards - vqa_min) / (vqa_max - vqa_min)
                else:
                    print("Warning: VQA scores have no variance. Using raw scores.")
                    vqa_normalized = vqa_rewards
                
                if verifier_flags['use_javis']:
                    javis_min = javis_scores.min()
                    javis_max = javis_scores.max()
                    if javis_max > javis_min:
                        javis_normalized = (javis_scores - javis_min) / (javis_max - javis_min)
                    else:
                        print("Warning: Javis scores have no variance. Using raw scores.")
                        javis_normalized = javis_scores
                    rewards = current_vqa_weight * vqa_normalized + current_javis_weight * javis_normalized
                else:
                    rewards = vqa_normalized
                    
            elif verifier_flags['use_vr']:
                # VideoReward verifier case
                vqa_min = vqa_rewards.min()
                vqa_max = vqa_rewards.max()
                if vqa_max > vqa_min:
                    vqa_normalized = (vqa_rewards - vqa_min) / (vqa_max - vqa_min)
                else:
                    print("Warning: VQA scores have no variance. Using raw scores.")
                    vqa_normalized = vqa_rewards
                
                if verifier_flags['use_javis']:
                    javis_min = javis_scores.min()
                    javis_max = javis_scores.max()
                    if javis_max > javis_min:
                        javis_normalized = (javis_scores - javis_min) / (javis_max - javis_min)
                    else:
                        print("Warning: Javis scores have no variance. Using raw scores.")
                        javis_normalized = javis_scores
                    
                    # Combine normalized scores with weights
                    rewards = current_vr_weight * vqa_normalized + current_javis_weight * javis_normalized
                else:
                    rewards = vqa_normalized
            else:
                print("Warning: No VQA or VideoReward verifier used in minmax mode.")
                rewards = torch.zeros_like(vqa_rewards) if len(vqa_rewards) > 0 else torch.zeros(len(audio_list)).to(device)
            
            print("VQA minmax normalized rewards:", vqa_normalized)
            if verifier_flags['use_javis']:
                print("Javis minmax normalized rewards:", javis_normalized)
            print("Final rewards:", rewards)

        elif score_method == 'adaptive':
            if not hasattr(model, 'weighter') or model.weighter is None:
                raise RuntimeError(
                    "AdaptiveRewardWeighter not initialized. "
                    "Please initialize model.weighter in evosearch.py before calling sample()."
                )
            weighter = model.weighter
            
            # 1. Collect all available scores
            score_dict = {}
            if verifier_flags['use_vr']:
                score_dict['VR'] = vqa_rewards
            
            if verifier_flags['use_javis']:
                score_dict['JS'] = javis_scores
            
            if verifier_flags['use_clap']:
                score_dict['CLAP'] = audio_scores
                
            if verifier_flags['use_align']:
                score_dict['Align'] = align_scores
                
            if verifier_flags['use_avh']:
                score_dict['AVH'] = avh_scores
            
            if verifier_flags['use_avib']:
                score_dict['AVIB'] = avib_scores

            if not score_dict:
                print("Warning: Adaptive method called but no scores collected. Returning zeros.")
                rewards = torch.zeros_like(vqa_rewards) if len(vqa_rewards) > 0 else torch.zeros(len(audio_list)).to(device)
            else:
                # 2. Store current prompt's rewards for re-evaluation (History Tracking)
                if hasattr(self, 'current_prompt_score_history'):
                    for k, v in score_dict.items():
                        if k not in self.current_prompt_score_history:
                            self.current_prompt_score_history[k] = []
                        self.current_prompt_score_history[k].append(v)
                
                # 3. Fit sigma
                print(f"Fitting adaptive weights on generation_step {generation_steps}...")
                weighter.fit_sigma(score_dict)
                
                # 4. Get weighted score
                rewards = weighter.get_weighted_score(score_dict, javis_boost=javis_boost)

                # Debug print
                sigmas = weighter.get_sigmas()
                print(f"Adaptive Weighting (Gen {generation_steps}): Sigmas={sigmas}")
                print(f"Weighted Score: {rewards}")

        return rewards, vqa_rewards, javis_scores


    ###############################################
    # EvoSearch
    ###############################################
    def evosearch(
        self,
        model,
        model_args,
        text_encoder,
        latent_dict,
        prompts,
        device,
        dtype,
        num_inference_steps: int,
        timesteps: list = None,
        num_samples_per_prompt: int = 1,
        verifier: Optional[Callable] = None,
        seed: Optional[int] = None,
        evolution_schedule: list[int] = [0],
        population_size_schedule: list[int] = [1],
        elite_size: int = 3,
        mutation_rate: float = 0.2, 
        guidance_reward: str="VideoReward",
        generation_steps: int = 0,
        iterations: int = 1,
        video_fps: float = 25.0,
        tournament_ratio: float = 0.9,
        stage_verifiers: Optional[list] = None,
        stage_weights: Optional[list] = None,
        score_method: str = "rank",
        VQ_mean: Optional[float] = None,
        VQ_std: Optional[float] = None,
        JS_mean: Optional[float] = None,
        JS_std: Optional[float] = None,
        javis_boost: float = 0.5,
        vae=None,
        audio_vae=None,
        num_frames=None,
        guidance_scale=1.0,
        mask=None,
        noise_added=None,
        evosearch_config=None,
    ) -> tuple[Tensor, Tensor]:
        
        # EvoSearch Initialization
        generation_steps_id = generation_steps
        std_audio_list=[-1 for _ in evolution_schedule]
        std_video_list=[-1 for _ in evolution_schedule]
        video_latent_to_decode_fn = lambda x: x
        audio_latent_to_decode_fn = lambda x: x
        if vae is not None and audio_vae is not None:
            # Helper to decode for reward calculation
            def decode_video(latents):
                 return vae.decode(latents.to(dtype), num_frames=num_frames)
            def decode_audio(latents):
                audio_length_in_s = num_frames / 24 if num_frames is not None else 4.0
                original_waveform_length = int(audio_length_in_s * 16000)
                return audio_vae.decode_audio(latents, original_waveform_length=original_waveform_length)
            video_latent_to_decode_fn = decode_video
            audio_latent_to_decode_fn = decode_audio

        current_step = evolution_schedule[generation_steps]
        latents_audio_total = latent_dict['audio']
        latents_video_total = latent_dict['video']
        
        # Sanitize timesteps if they are tensors (from multimodal_sample CFG prep)
        if timesteps is not None and isinstance(timesteps[0], torch.Tensor):
             timesteps = [t[0].item() for t in timesteps]

        # Prepare timesteps (legacy fallback if None)
        if timesteps is None:
             timesteps = [(1.0 - i / self.num_sampling_steps) * self.num_timesteps for i in range(self.num_sampling_steps)]
             if self.use_discrete_timesteps:
                 timesteps = [int(round(t)) for t in timesteps]

        if self.use_timestep_transform:
             # This might be tricky if additional_args depends on batch size, but usually it's static config
             pass 

        # We step starting from current_step
        progress_wrap = tqdm
        start_idx = current_step
        
        # Determine population size from input latents
        pop_size = latents_video_total.shape[0]
        
        # 1. Denoising loop
        for i in range(start_idx, num_inference_steps):
            t_scalar = timesteps[i]
            if isinstance(t_scalar, torch.Tensor):
                t_scalar = t_scalar[0].item()
            
            # Create batch of t
            t = torch.tensor([t_scalar] * pop_size, device=device)
            # Evolution Step
            if i in evolution_schedule:
                # 1. Variance Estimation
                # In RF, flow is straight, noise correlation is simpler. 
                # We use t/T to scale mutation.
                current_t_val = timesteps[i] if isinstance(timesteps[i], (float, int)) else timesteps[i].item()
                rf_noise_scale = current_t_val / self.num_timesteps
                
                std_audio = rf_noise_scale
                std_video = rf_noise_scale
                # TODO: Change ODE to SDE
                std_audio_list[generation_steps_id] = std_audio
                std_video_list[generation_steps_id] = std_video

                self.population_audio_list[generation_steps_id].append(latents_audio_total)
                self.population_video_list[generation_steps_id].append(latents_video_total)
                
                generation_steps_id += 1

            # timestep transform for model input
            if self.use_timestep_transform:
                 t = timestep_transform(t, model_kwargs={**model_args, 'height': model_args['height'], 'width':model_args['width'], 'num_frames':model_args['num_frames']}, num_timesteps=self.num_timesteps)

            # Construct inputs
            z_in = {
                'video': torch.cat([latents_video_total, latents_video_total], 0), # CFG
                'audio': torch.cat([latents_audio_total, latents_audio_total], 0)
            }
            t_in = torch.cat([t, t], 0)
            
            # Update masking if needed (simplified from multimodal_sample)
            current_model_args = model_args.copy()
            if mask is not None:
                mask_t = {}
                for k in ['video', 'audio']:
                    mask_t[k] = mask[k] * self.num_timesteps
                # Here we skip complex mask logic for EvoSearch unless strictly required.
                pass

            # Update model_args batch size if needed (e.g. embeddings)
            if 'y' in current_model_args:
                y = current_model_args['y'] 
                # y is [2*n, ...] from multimodal_sample (cfg)
                # We need to resize it to match [2*pop_size, ...]
                
                # Split conditional and unconditional
                n_orig = y.shape[0] // 2
                y_cond = y[:n_orig]
                y_uncond = y[n_orig:]
                
                # Repeat for population
                repeat_factor = pop_size // n_orig
                y_cond_new = y_cond.repeat(repeat_factor, 1, 1, 1)
                y_uncond_new = y_uncond.repeat(repeat_factor, 1, 1, 1)
                
                current_model_args['y'] = torch.cat([y_cond_new, y_uncond_new], 0)
            
            # Adjust other args similarly if they are batch-dependent
            # Adjust other args similarly if they are batch-dependent
            for k, v in current_model_args.items():
                if isinstance(v, torch.Tensor) and k != 'y':
                    # Case 1: Already doubled (CFG) [2N]
                    if v.shape[0] == y.shape[0]: 
                         n_orig = v.shape[0] // 2
                         v_cond = v[:n_orig]
                         v_uncond = v[n_orig:]
                         repeat_factor = pop_size // n_orig
                         v_cond_new = v_cond.repeat(repeat_factor, *([1]*(v.dim()-1)))
                         v_uncond_new = v_uncond.repeat(repeat_factor, *([1]*(v.dim()-1)))
                         current_model_args[k] = torch.cat([v_cond_new, v_uncond_new], 0)
                    # Case 2: Single batch [N] (e.g. fps) - Need to double for CFG then expand
                    elif v.shape[0] == y.shape[0] // 2:
                         # Assume same value for cond and uncond
                         v_doubled = torch.cat([v, v], dim=0)
                         # Now treat as above
                         n_orig = v_doubled.shape[0] // 2
                         v_cond = v_doubled[:n_orig]
                         v_uncond = v_doubled[n_orig:]
                         repeat_factor = pop_size // n_orig
                         v_cond_new = v_cond.repeat(repeat_factor, *([1]*(v.dim()-1)))
                         v_uncond_new = v_uncond.repeat(repeat_factor, *([1]*(v.dim()-1)))
                         current_model_args[k] = torch.cat([v_cond_new, v_uncond_new], 0)

            # Sequential processing for population if specified
            sequential_processing = evosearch_config.get('sequential_processing', False) if evosearch_config else False

            if sequential_processing and pop_size > 1:
                mm_pred_collected = {}
                for p in range(pop_size):
                     z_in_p = {k: torch.cat([v[p:p+1], v[pop_size + p:pop_size + p + 1]], dim=0) for k, v in z_in.items()}
                     t_in_p = torch.cat([t[p:p+1], t[p:p+1]], dim=0)
                     
                     args_p = {}
                     for k, v in current_model_args.items():
                         if isinstance(v, torch.Tensor) and v.shape[0] == 2*pop_size:
                             args_p[k] = torch.cat([v[p:p+1], v[pop_size+p:pop_size+p+1]], dim=0)
                         else:
                             args_p[k] = v
                     
                     pred_p = model(z_in_p, t_in_p, **args_p)
                     
                     for k, v in pred_p.items():
                         if k not in mm_pred_collected: mm_pred_collected[k] = {'cond': [], 'uncond': []}
                         if isinstance(v, torch.Tensor):
                             mm_pred_collected[k]['cond'].append(v[0:1])
                             mm_pred_collected[k]['uncond'].append(v[1:2])
                
                mm_pred = {}
                for k, v_dict in mm_pred_collected.items():
                    mm_pred[k] = torch.cat([torch.cat(v_dict['cond']), torch.cat(v_dict['uncond'])], dim=0)
            else:
                mm_pred = model(z_in, t_in, **current_model_args)

            # Update Latents
            for k, pred in mm_pred.items():
                if k not in ['video', 'audio']: continue
                pred = pred.chunk(2, dim=1)[0]
                pred_cond, pred_uncond = pred.chunk(2, dim=0)
                v_pred = pred_uncond + guidance_scale * (pred_cond - pred_uncond)

                z = latents_video_total if k == 'video' else latents_audio_total
                
                # RF Step
                # Use transformed t for dt calculation to match main loop dynamics
                # t is already transformed above if self.use_timestep_transform is True
                current_t = t[0] # scalar tensor
                next_t_scalar = timesteps[i + 1] if i < len(timesteps) - 1 else 0
                next_t = torch.tensor([next_t_scalar], device=device)
                if self.use_timestep_transform:
                    next_t = timestep_transform(next_t, model_kwargs={**model_args, 'height': model_args['height'], 'width':model_args['width'], 'num_frames':model_args['num_frames']}, num_timesteps=self.num_timesteps)
                
                dt = current_t - next_t[0]
                dt = dt / self.num_timesteps
                z = z + v_pred * dt # .view(-1, *([1] * (len(z.shape) - 1)))
                
                if k == 'video':
                    latents_video_total = z
                else:
                    latents_audio_total = z

        # 2. Decode and Evaluate
        audio_list_for_eval = []
        video_list_for_eval = []
        population_audio = audio_latent_to_decode_fn(latents_audio_total)
        population_video = video_latent_to_decode_fn(latents_video_total)
        
        # Add to global lists and evaluation lists
        for m in range(len(population_video)):
            self.audio_list.append(population_audio[m])
            self.video_list.append(population_video[m])
            audio_list_for_eval.append(population_audio[m])
            video_list_for_eval.append(population_video[m])
        
        rewards, vqa_rewards, align_scores = self._calculate_rewards(
            model=model,
            audio_list=audio_list_for_eval,
            video_list=video_list_for_eval,
            prompt=prompts[0],
            verifier=verifier,
            video_fps=video_fps,
            stage_verifiers=stage_verifiers,
            stage_weights=stage_weights,
            generation_steps=generation_steps,
            score_method=score_method,
            VQ_mean=VQ_mean, VQ_std=VQ_std,
            JS_mean=JS_mean, JS_std=JS_std,
            javis_boost=javis_boost,
            device=device
        )
        
        for id in range(generation_steps, len(self.rewards_list)):
            self.rewards_list[id].append(rewards)

        # 2. Select parent populations and calculate statistics
        population_audio = torch.cat(self.population_audio_list[generation_steps])
        population_video = torch.cat(self.population_video_list[generation_steps])
        
        mean_std_audio = std_audio_list[generation_steps]
        mean_std_video = std_video_list[generation_steps]

        # (TODO)Re-evaluate cached_rewards if adaptive
        if score_method == 'adaptive' and hasattr(self, 'current_prompt_score_history') and len(list(self.current_prompt_score_history.values())[0]) > 1:
            weighter = model.weighter
            
            # 1. Collect all history for current prompt
            current_prompt_all_scores = {}
            for k, v_list in self.current_prompt_score_history.items():
                current_prompt_all_scores[k] = torch.cat(v_list).to(device)
            
            # 2. Recalculate weighted scores for ALL generations with NEWEST weights
            recalculated_all_rewards = weighter.get_weighted_score(current_prompt_all_scores, javis_boost=javis_boost)
            
            print(f"Re-evaluated {len(recalculated_all_rewards)} samples (Gen 0~{generation_steps}) with updated weights")
            print("Original rewards:", torch.cat(self.rewards_list[generation_steps]))
            print("Recalculated rewards:", recalculated_all_rewards)
            
            # 3. Update cached_rewards for selection
            cached_rewards = recalculated_all_rewards
        else:
            cached_rewards = torch.cat(self.rewards_list[generation_steps])
        
        elite_rew, elite_indices = torch.topk(cached_rewards, elite_size)

        # 3. (TODO) Elitism
        if score_method == 'adaptive':
            self.best_reward = elite_rew[0]
            ind = elite_indices[0]
            self.best_video = self.video_list[ind]
            self.best_audio = self.audio_list[ind]
        else:
            if elite_rew[0] > self.best_reward:
                self.best_reward = elite_rew[0]
                ind = elite_indices[0]
                self.best_video = self.video_list[ind]
                self.best_audio = self.audio_list[ind]

        elites_video = population_video[elite_indices]
        elites_audio = population_audio[elite_indices]
        
        # 4. Selection & Mutation
        next_pop_size = population_size_schedule[generation_steps+1] if generation_steps + 1 < len(population_size_schedule) else pop_size
        
        # Tournament selection for parents
        parents_video, parents_audio = self._tournament_selection(
            population_video, population_audio, cached_rewards, 
            next_pop_size, elite_size, tournament_ratio
        )
        
        # Mutation
        std_weight = 1.0
        children_video, children_audio = self._mutate_population(
            parents_video, parents_audio, elites_video, elites_audio,
            generation_steps, mutation_rate, mean_std_audio, mean_std_video, std_weight=std_weight
        )
        
        del population_audio, population_video, elites_video, elites_audio
        del parents_video, parents_audio, cached_rewards
        del audio_list_for_eval, video_list_for_eval
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        return children_audio, children_video


    def _tournament_selection(self, population_video, population_audio, rewards, 
                             population_size, elite_size, tournament_ratio):
        """Select parents using tournament selection."""
        if population_size <= elite_size:
            return torch.empty(0, *population_video.shape[1:]), torch.empty(0, *population_audio.shape[1:])

        num_parents = population_size - elite_size
        candidates_size = int(population_video.shape[0] * tournament_ratio)
        
        parent_indices = []
        for _ in range(num_parents):
            candidates = torch.randperm(population_video.shape[0])[:candidates_size]
            candidate_rewards = rewards[candidates]
            winner = candidates[torch.argmax(candidate_rewards)]
            parent_indices.append(winner)
        
        parent_indices = torch.stack(parent_indices)
        return population_video[parent_indices], population_audio[parent_indices]

    def _mutate_population(self, parents_video, parents_audio, elites_video, elites_audio,
                          generation_steps, mutation_rate, mean_std_audio, mean_std_video, std_weight=None):
        """Apply mutation to create children from parents."""
        if parents_video.numel() == 0:  # No parents to mutate
            return elites_video, elites_audio

        if generation_steps == 0:
            # Initial generation uses global mutation
            mutation_factor = math.sqrt(1 - mutation_rate**2)
            children_video = parents_video * mutation_factor + mutation_rate * torch.randn_like(parents_video)
            children_audio = parents_audio * mutation_factor + mutation_rate * torch.randn_like(parents_audio)
        else:
            # Later generations use adaptive mutation based on variance
            children_video = parents_video + std_weight * mean_std_video * torch.randn_like(parents_video)
            children_audio = parents_audio + std_weight * mean_std_audio * torch.randn_like(parents_audio)

        # Combine elites with children
        return torch.cat([elites_video, children_video]), torch.cat([elites_audio, children_audio])

class AdaptiveRewardWeighter(nn.Module):
    def __init__(self, reward_names=None, lr=0.01, max_iter=50, optimizer_type="Adam", history_size=None):
        """
        reward_names: List of reward keys to initialize (e.g. ['VR', 'JS'])
        lr: Learning rate for internal optimization
        max_iter: Number of iterations for Test-Time Training
        history_size: Max number of recent values to keep. None = keep all (default)
        """
        super().__init__()
        self.max_iter = max_iter
        self.lr = lr
        self.optimizer_type = optimizer_type
        self.history_size = history_size  # None = keep all, int = keep last N values

        # Dictionary to store log_vars for each reward type
        self.log_vars = nn.ParameterDict()

        # History buffer: dictionary of lists
        self.history = {}
        self.online_trace = []
        self.online_step = 0
        self.online_step_trace = []
        self.total_optimizer_step = 0

        # Initialize parameters and optimizer immediately if keys are provided
        if reward_names:
            for key in reward_names:
                self.log_vars[key] = nn.Parameter(torch.zeros(1))
                self.history[key] = []
            
            # Initialize optimizer immediately
            self.optimizer = self._build_optimizer()
        else:
            # Fallback for empty init (though deprecated use)
            self.optimizer = None

    def _build_optimizer(self):
        optimizer_type = self.optimizer_type.lower()
        if optimizer_type == "adam":
            return optim.Adam(self.log_vars.parameters(), lr=self.lr)
        if optimizer_type == "adamw":
            return optim.AdamW(self.log_vars.parameters(), lr=self.lr)
        if optimizer_type == "rmsprop":
            return optim.RMSprop(self.log_vars.parameters(), lr=self.lr)
        if optimizer_type == "sgd":
            return optim.SGD(self.log_vars.parameters(), lr=self.lr, momentum=0.9)
        if optimizer_type == "adagrad":
            return optim.Adagrad(self.log_vars.parameters(), lr=self.lr)
        if optimizer_type == "lbfgs":
            return optim.LBFGS(self.log_vars.parameters(), lr=self.lr, max_iter=5, history_size=10, line_search_fn="strong_wolfe")
        raise ValueError(f"Unsupported optimizer_type: {self.optimizer_type}")

    def _get_device(self):
        if len(self.log_vars) > 0:
            return next(self.log_vars.parameters()).device
        return torch.device("cpu")

    def _build_loss_inputs(self, keys=None, device=None):
        if device is None:
            device = self._get_device()

        if keys is None:
            keys = [key for key, values in self.history.items() if values]

        all_data = {}
        for key in keys:
            if key in self.history and self.history[key]:
                all_data[key] = torch.cat(self.history[key]).to(device)

        loss_inputs = {}
        for key, data in all_data.items():
            if data.numel() > 1:
                loss_inputs[key] = (data - data.mean()).pow(2) + 1e-6

        return loss_inputs

    @staticmethod
    def _compute_total_loss(log_vars, loss_inputs, keys):
        total_loss = None
        for key in keys:
            if key not in loss_inputs or key not in log_vars:
                continue
            log_var = log_vars[key]
            prec = torch.exp(-log_var)
            loss_component = torch.mean(0.5 * prec * loss_inputs[key] + 0.5 * log_var)
            total_loss = loss_component if total_loss is None else total_loss + loss_component
        return total_loss

    def get_online_trace(self):
        return list(self.online_trace)

    def get_online_step_trace(self):
        return list(self.online_step_trace)

    def fit_sigma(self, score_dict):
        """
        [Test-Time Training]
        Adaptively learn sigma for each reward type based on variance.
        Input:
            score_dict: Dictionary of {name: tensor [Batch_size]}
        """
        # 1. Update history
        for key, val in score_dict.items():
            if key not in self.history:
                self.history[key] = []
            self.history[key].append(val.detach().cpu())

            # Keep only last history_size values if specified
            if self.history_size is not None and len(self.history[key]) > self.history_size:
                # Remove oldest value to maintain size limit
                self.history[key] = self.history[key][-self.history_size:]
            
        # 2. Prepare data for training
        current_keys = list(score_dict.keys())
        if not current_keys:
            return

        # Ensure all keys have history initialized (if new keys appeared)
        for key in current_keys:
             if key not in self.log_vars:
                 self.log_vars[key] = nn.Parameter(torch.zeros(1).to(val.device)) # Initialize on correct device
                 # If optimizer exists, we might need to re-init or add param group, 
                 # but usually reward_names are fixed. 
                 # For safety, let's assume fixed keys or recreate optimizer if needed (simplified here)
                 if self.optimizer:
                     # Re-init optimizer to include new params? 
                     # For simply usage, we assume reward_names covered most cases or we re-create.
                     self.optimizer = self._build_optimizer()

        first_key = current_keys[0]
        # Check if we have history for the first key (assuming synchronized history)
        if not self.history[first_key]:
            return
            
        # Check total length
        total_len = sum(len(x) for x in self.history[first_key])
        if total_len <= 1:
            return

        # Concatenate history for all keys and calculate variance targets
        loss_inputs = self._build_loss_inputs(keys=current_keys)

        self.train()
        
        # 3. Instant Training Loop
        if self.optimizer is None:
             self.optimizer = self._build_optimizer()

        with torch.enable_grad():
            loss_before = self._compute_total_loss(self.log_vars, loss_inputs, current_keys)
            for inner_iter in range(self.max_iter):
                if self.optimizer_type.lower() == "lbfgs":
                    def closure():
                        self.optimizer.zero_grad()
                        total_loss = self._compute_total_loss(self.log_vars, loss_inputs, current_keys)
                        if total_loss is None:
                            raise RuntimeError("LBFGS closure received no loss.")
                        total_loss.backward()
                        return total_loss

                    self.optimizer.step(closure)
                else:
                    self.optimizer.zero_grad()
                    total_loss = self._compute_total_loss(self.log_vars, loss_inputs, current_keys)
                    if total_loss is None:
                        return
                    total_loss.backward()
                    self.optimizer.step()

                loss_current = self._compute_total_loss(self.log_vars, loss_inputs, current_keys)
                self.total_optimizer_step += 1
                step_row = {
                    "prompt_index": int(self.online_step),
                    "inner_iter": int(inner_iter + 1),
                    "optimizer_step": int(self.total_optimizer_step),
                    "num_history": int(total_len),
                    "loss": float(loss_current.detach().item()) if loss_current is not None else None,
                    "max_iter": int(self.max_iter),
                }
                for key in current_keys:
                    if key in self.log_vars:
                        log_var = self.log_vars[key].detach()
                        step_row[f"log_var_{key}"] = float(log_var.item())
                        step_row[f"sigma_{key}"] = float(torch.exp(0.5 * log_var).item())
                self.online_step_trace.append(step_row)

        loss_after = self._compute_total_loss(self.log_vars, loss_inputs, current_keys)
        trace_row = {
            "prompt_index": int(self.online_step),
            "num_history": int(total_len),
            "loss_before": float(loss_before.detach().item()) if loss_before is not None else None,
            "loss_after": float(loss_after.detach().item()) if loss_after is not None else None,
            "loss_delta": float((loss_after - loss_before).detach().item())
            if loss_before is not None and loss_after is not None else None,
            "max_iter": int(self.max_iter),
        }
        for key in current_keys:
            if key in self.log_vars:
                log_var = self.log_vars[key].detach()
                trace_row[f"log_var_{key}"] = float(log_var.item())
                trace_row[f"sigma_{key}"] = float(torch.exp(0.5 * log_var).item())
                latest_score = score_dict[key]
                trace_row[f"score_mean_{key}"] = float(latest_score.detach().mean().item())
        self.online_trace.append(trace_row)
        self.online_step += 1
            
    def get_weighted_score(self, score_dict, weights=None, javis_boost=None):
        """
        Compute weighted sum of scores using learned sigmas.
        weights: Dictionary of {name: weight_value}
        javis_boost: Legacy parameter for boosting JS over VR. 
        """
        self.eval()
        final_score = 0
        
        # Handle legacy javis_boost if weights not provided
        if weights is None and javis_boost is not None:
            weights = {}
            if 'VR' in score_dict:
                weights['VR'] = 1.0 - javis_boost
            if 'JS' in score_dict:
                weights['JS'] = javis_boost
            if 'CLAP' in score_dict:
                weights['CLAP'] = javis_boost # Assumes boosting new scores too if generic
            if 'AVH' in score_dict:
                weights['AVH'] = javis_boost
            if 'Align' in score_dict:
                weights['Align'] = javis_boost

        with torch.no_grad():
            for key, score in score_dict.items():
                if key not in self.log_vars:
                   # If key not known, treat sigma as 1.0 (log_var=0)
                   sigma = 1.0
                else:
                   sigma = torch.exp(0.5 * self.log_vars[key])
                
                # Normalize: Score / Sigma
                norm_score = score / (sigma + 1e-6)
                
                # Apply weight
                w = 1.0
                if weights is not None and key in weights:
                    w = weights[key]
                
                final_score += w * norm_score
                
        return final_score

    def get_sigmas(self):
        """Debug: return dict of sigmas"""
        with torch.no_grad():
            return {k: torch.exp(0.5 * v).item() for k, v in self.log_vars.items()}
