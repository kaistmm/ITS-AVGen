import os
import sys
import time
from pathlib import Path
from pprint import pformat
import warnings
warnings.filterwarnings('ignore')

import colossalai
import torch
import torch.distributed as dist
from colossalai.cluster import DistCoordinator
from mmengine.runner import set_random_seed
from tqdm import tqdm
import numpy as np
import pandas as pd

from javisdit.acceleration.parallel_states import set_sequence_parallel_group
from javisdit.datasets import save_sample
from javisdit.datasets.aspect import get_image_size, get_num_frames
from javisdit.models.text_encoder.t5 import text_preprocessing
from javisdit.registry import MODELS, SCHEDULERS, build_module
from javisdit.utils.config_utils import parse_configs
from javisdit.utils.inference_utils import (
    add_watermark,
    append_generated,
    append_score_to_prompts,
    apply_va_mask_strategy,
    collect_va_references_batch,
    dframe_to_frame,
    extract_json_from_prompts,
    extract_prompts_loop,
    get_save_path_name,
    load_prompts,
    merge_prompt,
    prepare_multi_resolution_info,
    refine_prompts_by_openai,
    split_prompt
)
from javisdit.utils.misc import all_exists, create_logger, is_distributed, is_main_process, to_torch_dtype


def save_arw_online_artifacts(weighter, save_dir, logger, probe_cfg=None):
    probe_cfg = probe_cfg or {}
    artifact_prefix = probe_cfg.get("online_prefix", "arw_online")
    step_artifact_prefix = probe_cfg.get("online_step_prefix", "arw_online_step")
    trace = weighter.get_online_trace()
    step_trace = weighter.get_online_step_trace()
    if not trace:
        logger.warning("Skipping ARW online export: no online updates were recorded.")
        return

    trace_df = pd.DataFrame(trace)
    output_dir = Path(save_dir)
    csv_path = output_dir / f"{artifact_prefix}.csv"
    trace_df.to_csv(csv_path, index=False)

    png_path = output_dir / f"{artifact_prefix}.png"
    try:
        import matplotlib.pyplot as plt

        sigma_cols = [col for col in trace_df.columns if col.startswith("sigma_")]
        num_rows = 2 if sigma_cols else 1
        fig, axes = plt.subplots(num_rows, 1, figsize=(9, 4 * num_rows), sharex=True)
        if num_rows == 1:
            axes = [axes]

        x = trace_df["prompt_index"]
        axes[0].plot(x, trace_df["loss_before"], label="loss_before", linewidth=1.4)
        axes[0].plot(x, trace_df["loss_after"], label="loss_after", linewidth=1.4)
        axes[0].set_ylabel("Loss")
        axes[0].set_title("ARW online loss by prompt index")
        axes[0].grid(True, alpha=0.3)
        axes[0].legend()

        if sigma_cols:
            for col in sigma_cols:
                axes[1].plot(x, trace_df[col], label=col, linewidth=1.4)
            axes[1].set_xlabel("Prompt index")
            axes[1].set_ylabel("Sigma")
            axes[1].set_title("ARW online sigma by prompt index")
            axes[1].grid(True, alpha=0.3)
            axes[1].legend()
        else:
            axes[0].set_xlabel("Prompt index")

        fig.tight_layout()
        fig.savefig(png_path, dpi=200, bbox_inches="tight")
        plt.close(fig)
        logger.info("Saved ARW online curve to %s", png_path)
    except Exception as exc:
        logger.warning("Failed to render ARW online plot: %s", exc)

    if step_trace:
        step_df = pd.DataFrame(step_trace)
        step_csv_path = output_dir / f"{step_artifact_prefix}.csv"
        step_df.to_csv(step_csv_path, index=False)

        step_png_path = output_dir / f"{step_artifact_prefix}.png"
        try:
            import matplotlib.pyplot as plt

            sigma_cols = [col for col in step_df.columns if col.startswith("sigma_")]
            num_rows = 2 if sigma_cols else 1
            fig, axes = plt.subplots(num_rows, 1, figsize=(9, 4 * num_rows), sharex=True)
            if num_rows == 1:
                axes = [axes]

            x = step_df["optimizer_step"]
            axes[0].plot(x, step_df["loss"], label="loss", linewidth=1.4)
            axes[0].set_ylabel("Loss")
            axes[0].set_title("ARW online loss by total optimizer step")
            axes[0].grid(True, alpha=0.3)
            axes[0].legend()

            if sigma_cols:
                for col in sigma_cols:
                    axes[1].plot(x, step_df[col], label=col, linewidth=1.4)
                axes[1].set_xlabel("Total optimizer step")
                axes[1].set_ylabel("Sigma")
                axes[1].set_title("ARW online sigma by total optimizer step")
                axes[1].grid(True, alpha=0.3)
                axes[1].legend()
            else:
                axes[0].set_xlabel("Total optimizer step")

            fig.tight_layout()
            fig.savefig(step_png_path, dpi=200, bbox_inches="tight")
            plt.close(fig)
            logger.info("Saved ARW online step curve to %s", step_png_path)
        except Exception as exc:
            logger.warning("Failed to render ARW online step plot: %s", exc)
        logger.info("Saved ARW online step table to %s", step_csv_path)
    else:
        logger.warning("Skipping ARW online step export: no optimizer-step trace was recorded.")

    logger.info("Saved ARW online table to %s", csv_path)


def save_arw_checkpoint(weighter, save_dir, logger, probe_cfg=None):
    probe_cfg = probe_cfg or {}
    ckpt_name = probe_cfg.get("checkpoint_name", "adaptive_reward_weighter.pt")
    ckpt_path = Path(save_dir) / ckpt_name
    history = {
        key: [tensor.clone() for tensor in values]
        for key, values in weighter.history.items()
    }
    payload = {
        "state_dict": weighter.state_dict(),
        "optimizer_state_dict": weighter.optimizer.state_dict() if weighter.optimizer is not None else None,
        "history": history,
        "online_trace": weighter.get_online_trace(),
        "online_step_trace": weighter.get_online_step_trace(),
        "lr": weighter.lr,
        "max_iter": weighter.max_iter,
        "optimizer_type": weighter.optimizer_type,
    }
    torch.save(payload, ckpt_path)
    logger.info("Saved AdaptiveRewardWeighter checkpoint to %s", ckpt_path)


def main():
    torch.set_grad_enabled(False)
    # ======================================================
    # configs & runtime variables
    # ======================================================
    # == parse configs ==
    cfg = parse_configs(training=False)
    audio_only = cfg.get('audio_only', False)

    # == device and dtype ==
    device = "cuda" if torch.cuda.is_available() else "cpu"
    cfg_dtype = cfg.get("dtype", "fp32")
    assert cfg_dtype in ["fp16", "bf16", "fp32"], f"Unknown mixed precision {cfg_dtype}"
    dtype = to_torch_dtype(cfg.get("dtype", "bf16"))
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    # == init distributed env ==
    if is_distributed():
        colossalai.launch_from_torch({})
        coordinator = DistCoordinator()
        enable_sequence_parallelism = coordinator.world_size > 1
        if enable_sequence_parallelism:
            set_sequence_parallel_group(dist.group.WORLD)
    else:
        coordinator = None
        enable_sequence_parallelism = False
    set_random_seed(seed=cfg.get("seed", 1024))

    # == init logger ==
    logger = create_logger()
    logger.info("Inference configuration:\n %s", pformat(cfg.to_dict()))
    verbose = cfg.get("verbose", 1)
    progress_wrap = tqdm if verbose == 1 else (lambda x: x)

    # ======================================================
    # Init Reward model
    # ======================================================
    correct = cfg.get('correct', None)
    evosearch = cfg.get('evosearch', None)
    if correct is None or evosearch is None:
        verifier = None
        print("No verifier")
    else:
        if evosearch.guidance_reward == "VideoReward":
            sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
            from reward_model.videoreward import VideoRewardModel
            print("Initializing VideoReward model...")
            verifier = VideoRewardModel(correct)
        elif evosearch.guidance_reward == "VQA":
            sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
            from reward_model.vqa import VQARewardModel
            print("Initializing VQA model...")
            verifier = VQARewardModel(correct)

    # ======================================================
    # build model & load weights
    # ======================================================
    logger.info("Building models...")
    # == build text-encoder and vae ==
    text_encoder = build_module(cfg.text_encoder, MODELS, device=device, dtype=dtype)
    prior_encoder = build_module(cfg.get('prior_encoder', None), MODELS)
    if prior_encoder is not None:
        prior_encoder = prior_encoder.to(device, dtype).eval()        
    vae = build_module(cfg.vae, MODELS).to(device, dtype).eval()
    audio_vae = build_module(cfg.audio_vae, MODELS, device=device, dtype=dtype)

    # == prepare video size ==
    image_size = cfg.get("image_size", None)
    if image_size is None:
        resolution = cfg.get("resolution", None)
        aspect_ratio = cfg.get("aspect_ratio", None)
        assert (
            resolution is not None and aspect_ratio is not None
        ), "resolution and aspect_ratio must be provided if image_size is not provided"
        image_size = get_image_size(resolution, aspect_ratio)
    num_frames = get_num_frames(cfg.num_frames)

    # == build diffusion model ==
    input_size = (num_frames, *image_size)
    v_latent_size = vae.get_latent_size(input_size)  # [t//4 for every 17 frame, h//8, w//8]
    model = (
        build_module(
            cfg.model,
            MODELS,
            input_size=v_latent_size,
            in_channels=vae.out_channels,
            caption_channels=text_encoder.output_dim,
            model_max_length=text_encoder.model_max_length,
            enable_sequence_parallelism=enable_sequence_parallelism,
            from_pretrained=cfg.get('model_path', cfg.model.get('from_pretrained')),
            weight_init_from=cfg.model.get('weight_init_from', {}),
        )
        .to(device, dtype)
        .eval()
    )
    text_encoder.y_embedder = model.y_embedder  # HACK: for classifier-free guidance
    if prior_encoder is not None:
        prior_encoder.st_prior_embedder = model.st_prior_embedder

    # == build scheduler ==
    scheduler = build_module(cfg.scheduler, SCHEDULERS)

    # ======================================================
    # inference
    # ======================================================
    # == load prompts ==
    prompts = cfg.get("prompt", None)
    start_idx = cfg.get("start_index", 0)
    if prompts is None:
        if cfg.get("prompt_path", None) is not None:
            prompts = load_prompts(cfg.prompt_path, start_idx, cfg.get("end_index", None), prompt_key='text')
        else:
            prompts = [cfg.get("prompt_generator", "")] * 1_000_000  # endless loop
    elif isinstance(prompts, str):
        prompts = [prompts]
    
    # == dataset sharding for multi-GPU ==
    nshard = cfg.get("nshard", 1)
    shard_id = cfg.get("shard_id", 0)
    
    if nshard > 1:
        total_prompts = len(prompts)
        shard_size = (total_prompts + nshard - 1) // nshard  # Ceiling division
        shard_start = shard_id * shard_size
        shard_end = min(shard_start + shard_size, total_prompts)
        
        logger.info(f"Sharding dataset: shard {shard_id}/{nshard}, processing samples {shard_start}-{shard_end-1} of {total_prompts}")
        prompts = prompts[shard_start:shard_end]
        start_idx = start_idx + shard_start  # Adjust global index

    # == prepare reference ==
    # Adjust reference paths and mask strategies for the shard
    reference_path_full = cfg.get("reference_path", None)
    mask_strategy_full = cfg.get("mask_strategy", None)
    
    if nshard > 1 and (reference_path_full is not None or mask_strategy_full is not None):
        # Shard reference paths and mask strategies
        if reference_path_full is None:
            reference_path = [("", "")] * len(prompts)
        elif isinstance(reference_path_full, str) and os.path.isfile(reference_path_full):
            reference_df = pd.read_csv(reference_path_full)
            reference_list = list(zip(reference_df['path'].tolist(), reference_df['audio_path'].tolist()))
            shard_start = shard_id * shard_size
            shard_end = min(shard_start + shard_size, len(reference_list))
            reference_path = reference_list[shard_start:shard_end]
        else:
            shard_start = shard_id * shard_size
            shard_end = min(shard_start + shard_size, len(reference_path_full))
            reference_path = reference_path_full[shard_start:shard_end]
        
        if mask_strategy_full is None:
            mask_strategy = [""] * len(prompts)
        elif isinstance(mask_strategy_full, str):
            mask_strategy = [mask_strategy_full] * len(prompts)
        else:
            shard_start = shard_id * shard_size
            shard_end = min(shard_start + shard_size, len(mask_strategy_full))
            mask_strategy = mask_strategy_full[shard_start:shard_end]
    else:
        # No sharding, use original logic
        reference_path = cfg.get("reference_path", [("", "")] * len(prompts))
        mask_strategy = cfg.get("mask_strategy", [""] * len(prompts))
    if isinstance(reference_path, str) and os.path.isfile(reference_path):
        reference_df = pd.read_csv(reference_path)
        reference_path = list(zip(reference_df['path'].tolist(), reference_df['audio_path'].tolist()))
    if isinstance(mask_strategy, str):
        mask_strategy = [mask_strategy] * len(prompts)
    assert len(reference_path) == len(prompts), "Length of reference must be the same as prompts"
    assert len(mask_strategy) == len(prompts), "Length of mask_strategy must be the same as prompts"

    # == prepare arguments ==
    fps = cfg.fps
    save_fps = cfg.get("save_fps", fps // cfg.get("frame_interval", 1))
    multi_resolution = cfg.get("multi_resolution", None)
    batch_size = cfg.get("batch_size", 1)
    num_sample = cfg.get("num_sample", 1)
    loop = cfg.get("loop", 1)
    condition_frame_length = cfg.get("condition_frame_length", 5)
    condition_frame_edit = cfg.get("condition_frame_edit", 0.0)
    align = cfg.get("align", None)
    assert loop == 1, "not implemented"
    audio_fps = cfg.get("audio_fps", 16000)

    save_dir = cfg.save_dir
    os.makedirs(save_dir, exist_ok=True)
    sample_name = cfg.get("sample_name", None)
    prompt_as_path = cfg.get("prompt_as_path", False)

    # == Timing initialization (only for single-GPU mode, first sample only) ==
    enable_timing = True
    first_sample_time = None
    if enable_timing:
        inference_start_time = time.time()

    # Initialize AdaptiveRewardWeighter once for all prompts (cross-prompt accumulation)
    if evosearch is not None:
        score_method = getattr(evosearch, 'score_method', 'zscore')
        if score_method == 'adaptive':
            from javisdit.schedulers.rf import AdaptiveRewardWeighter
            adaptive_optimizer = cfg.get("adaptive_optimizer", "RMSprop")
            history_size = cfg.get("history_size", None)  # None = keep all, int = keep last N
            model.weighter = AdaptiveRewardWeighter(
                lr=0.05,
                max_iter=1,
                optimizer_type=adaptive_optimizer,
                history_size=history_size,
            ).to(device)
            logger.info("Initialized AdaptiveRewardWeighter for cross-prompt accumulation")
            if history_size is not None:
                logger.info(f"History size limited to {history_size} recent values")
    adaptive_convergence = cfg.get("adaptive_convergence", None)

    # == Iter over all samples ==
    for i in progress_wrap(range(0, len(prompts), batch_size)):
        # == prepare batch prompts ==
        batch_prompts = prompts[i : i + batch_size]
        ms = mask_strategy[i : i + batch_size]
        refs = reference_path[i : i + batch_size]

        # == get json from prompts ==
        batch_prompts, refs, ms = extract_json_from_prompts(batch_prompts, refs, ms, noimpl=True)
        original_batch_prompts = batch_prompts

        # == get reference for condition ==
        refs = collect_va_references_batch(refs, vae, image_size, 
                                           audio_vae=audio_vae, audio_cfg=cfg.get("audio_cfg", {}))

        # == multi-resolution info ==
        model_args = prepare_multi_resolution_info(
            multi_resolution, len(batch_prompts), image_size, num_frames, fps, device, dtype
        )

        # == Iter over number of sampling for one prompt ==
        for k in range(num_sample):
            # == prepare save paths ==
            save_paths = [
                get_save_path_name(
                    save_dir,
                    sample_name=sample_name,
                    sample_idx=start_idx + idx,
                    prompt=original_batch_prompts[idx],
                    prompt_as_path=prompt_as_path,
                    num_sample=num_sample,
                    k=k,
                )
                for idx in range(len(batch_prompts))
            ]
            if coordinator and coordinator.world_size > 1:
                dist.barrier()
            if all(os.path.exists(f'{_path}.mp4') for _path in save_paths):
                continue

            # NOTE: Skip if the sample already exists
            # This is useful for resuming sampling VBench
            if prompt_as_path and all_exists(save_paths):
                continue

            # == process prompts step by step ==
            # 0. split prompt
            # each element in the list is [prompt_segment_list, loop_idx_list]
            batched_prompt_segment_list = []
            batched_loop_idx_list = []
            for prompt in batch_prompts:
                prompt_segment_list, loop_idx_list = split_prompt(prompt)
                batched_prompt_segment_list.append(prompt_segment_list)
                batched_loop_idx_list.append(loop_idx_list)

            # 1. refine prompt by openai
            if cfg.get("llm_refine", False):
                # only call openai API when
                # 1. seq parallel is not enabled
                # 2. seq parallel is enabled and the process is rank 0
                if not enable_sequence_parallelism or (enable_sequence_parallelism and is_main_process()):
                    for idx, prompt_segment_list in enumerate(batched_prompt_segment_list):
                        batched_prompt_segment_list[idx] = refine_prompts_by_openai(prompt_segment_list)

                # sync the prompt if using seq parallel
                if enable_sequence_parallelism:
                    coordinator.block_all()
                    prompt_segment_length = [
                        len(prompt_segment_list) for prompt_segment_list in batched_prompt_segment_list
                    ]

                    # flatten the prompt segment list
                    batched_prompt_segment_list = [
                        prompt_segment
                        for prompt_segment_list in batched_prompt_segment_list
                        for prompt_segment in prompt_segment_list
                    ]

                    # create a list of size equal to world size
                    broadcast_obj_list = [batched_prompt_segment_list] * coordinator.world_size
                    dist.broadcast_object_list(broadcast_obj_list, 0)

                    # recover the prompt list
                    batched_prompt_segment_list = []
                    segment_start_idx = 0
                    all_prompts = broadcast_obj_list[0]
                    for num_segment in prompt_segment_length:
                        batched_prompt_segment_list.append(
                            all_prompts[segment_start_idx : segment_start_idx + num_segment]
                        )
                        segment_start_idx += num_segment

            # 2. append score
            for idx, prompt_segment_list in enumerate(batched_prompt_segment_list):
                batched_prompt_segment_list[idx] = append_score_to_prompts(
                    prompt_segment_list,
                    aes=cfg.get("aes", None),
                    flow=cfg.get("flow", None),
                    camera_motion=cfg.get("camera_motion", None),
                )

            # 3. clean prompt with T5
            for idx, prompt_segment_list in enumerate(batched_prompt_segment_list):
                batched_prompt_segment_list[idx] = [text_preprocessing(prompt) for prompt in prompt_segment_list]

            # 4. merge to obtain the final prompt
            batch_prompts = []
            for prompt_segment_list, loop_idx_list in zip(batched_prompt_segment_list, batched_loop_idx_list):
                batch_prompts.append(merge_prompt(prompt_segment_list, loop_idx_list))

            # == Iter over loop generation ==
            video_clips, audio_clips = [], []
            for loop_i in range(loop):
                # == get prompt for loop i ==
                batch_prompts_loop = extract_prompts_loop(batch_prompts, loop_i)

                # == add condition frames for loop ==
                if loop_i > 0:
                    raise NotImplementedError
                    refs, ms = append_generated(
                        vae, video_clips[-1], refs, ms, loop_i, condition_frame_length, condition_frame_edit
                    )

                # == sampling ==
                if enable_timing:
                    sample_start_time = time.time()
                
                population_size_schedule = evosearch.population_size_schedule
                initial_population_size = population_size_schedule[0]
                
                # Replicate references and mask strategies for each population member
                refs_expanded = refs * initial_population_size
                ms_expanded = ms * initial_population_size
                
                vz = torch.randn(initial_population_size, vae.out_channels, *v_latent_size, device=device, dtype=dtype)
                audio_length_in_s = num_frames / fps
                az, original_waveform_length = audio_vae.prepare_latents(audio_length_in_s, initial_population_size, device=device, dtype=dtype)
                masks = apply_va_mask_strategy(vz, az, refs_expanded, ms_expanded, loop_i, align=align, v2a_t_scale=1/5*17/fps/(10.24/1024)/4)
                
                # Expand model_args for population_size
                model_args_expanded = {}
                for key, value in model_args.items():
                    if isinstance(value, torch.Tensor):
                        # Replicate tensor for each population member
                        model_args_expanded[key] = value.repeat(initial_population_size, *([1] * (value.dim() - 1)))
                    else:
                        model_args_expanded[key] = value

                # Replicate prompts for each population member
                batch_prompts_loop_expanded = batch_prompts_loop * initial_population_size
                
                samples = scheduler.multimodal_sample(
                    model,
                    text_encoder,
                    {'video': vz, 'audio': az},
                    batch_prompts_loop_expanded,
                    device=device,
                    additional_args=model_args_expanded,
                    progress=verbose >= 2,
                    mask=masks,
                    prior_encoder=prior_encoder,
                    verifier=verifier,
                    evosearch=evosearch,
                    vae=vae,
                    audio_vae=audio_vae,
                    num_frames=num_frames,
                    dtype=dtype,
                )
                
                # Get decoded samples if available, otherwise get latents
                if 'video_decoded' in samples and 'audio_decoded' in samples:
                    # Best sample already selected by scheduler
                    video_samples = samples['video_decoded']
                    audio_samples = samples['audio_decoded']
                else:
                    # No verifier, decode latents here
                    video_samples = samples['video']
                    audio_samples = samples['audio']
                    video_samples = vae.decode(video_samples.to(dtype), num_frames=num_frames)
                    audio_samples = audio_vae.decode_audio(audio_samples, original_waveform_length=original_waveform_length)
                
                video_clips.append(video_samples)
                audio_clips.append(audio_samples)

            # Record timing for first sample only (only in single-GPU mode)
            if enable_timing:
                torch.cuda.synchronize()
                sample_end_time = time.time()
                first_sample_time = sample_end_time - sample_start_time
                logger.info(f"First sample completed in {first_sample_time:.2f}s")

            # == save samples ==
            if is_main_process():
                for idx, batch_prompt in enumerate(batch_prompts):
                    if verbose >= 2:
                        logger.info("Prompt: %s", batch_prompt)
                    save_path = save_paths[idx]
                    video = video_clips[0][idx]
                    audio = audio_clips[0][idx]

                    save_path = save_sample(
                        video,
                        fps=save_fps,
                        audio=audio,
                        audio_fps=audio_fps,
                        save_path=save_path,
                        verbose=verbose >= 2,
                        audio_only=audio_only,
                    )
                    if save_path.endswith(".mp4") and cfg.get("watermark", False):
                        time.sleep(1)  # prevent loading previous generated video
                        add_watermark(save_path)
        
        start_idx += len(batch_prompts)
    
    # == Timing summary ==
    logger.info("Inference finished.")
    logger.info("Saved %s samples to %s", start_idx, save_dir)

    if (
        adaptive_convergence
        and adaptive_convergence.get("enabled", True)
        and is_main_process()
        and evosearch is not None
        and getattr(evosearch, "score_method", "zscore") == "adaptive"
        and hasattr(model, "weighter")
        and model.weighter is not None
    ):
        save_arw_online_artifacts(model.weighter, save_dir, logger, adaptive_convergence)
        save_arw_checkpoint(model.weighter, save_dir, logger, adaptive_convergence)
    
    if enable_timing:
        total_time = time.time() - inference_start_time
        logger.info("="*60)
        logger.info("Final Summary:")
        logger.info(f"  Total inference time: {total_time:.2f}s ({total_time/60:.2f}min)")
        if first_sample_time is not None:
            logger.info(f"  First sample time: {first_sample_time:.2f}s")
        logger.info("="*60)


if __name__ == "__main__":
    main()
