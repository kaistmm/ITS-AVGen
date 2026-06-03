import os
os.environ["HF_USE_FLASH_ATTENTION_2"] = "0"
os.environ["TRANSFORMERS_ATTENTION_IMPLEMENTATION"] = "sdpa"
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
import argparse
import time
import sys
import math
from multiprocessing.managers import BaseManager
from torch.profiler import profile, ProfilerActivity
from transformers import AutoProcessor, ClapModel

sys.path.append(os.path.dirname(__file__))
from utils.video_preprocessing import (
    normalize_video_tensor_shape, 
    prepare_single_video_for_model,
    preprocess_video_for_imagebind,
    preprocess_video_global_for_imagebind,
    preprocess_audio_for_imagebind,
    segment_clip_transform
)


class RemoteVQAManager(BaseManager):
    pass


PRINTED_FLOP_METRICS = set()


def profile_metric_flops(metric_name, fn):
    if not print_flops or metric_name in PRINTED_FLOP_METRICS:
        return fn()

    activities = [ProfilerActivity.CPU]
    if torch.cuda.is_available() and str(device).startswith("cuda"):
        activities.append(ProfilerActivity.CUDA)

    try:
        with profile(activities=activities, with_flops=True) as prof:
            result = fn()
            if ProfilerActivity.CUDA in activities:
                torch.cuda.synchronize()
        total_flops = sum(event.flops for event in prof.key_averages() if event.flops)
        print(f"[FLOPs] {metric_name}: {total_flops / 1e9:.3f} GFLOPs")
    except Exception as e:
        print(f"[FLOPs] {metric_name}: unavailable ({e})")
        result = fn()

    PRINTED_FLOP_METRICS.add(metric_name)
    return result

@torch.no_grad()
def process_VQA(input):
    start_time = time.time()
    images = input.get("images")
    text_prompt = input.get("text")
    align_reward_weight = input.get("arw")

    scores = []
    for idx in range(0, len(images), vqa_batch_size):
        cur_batch_size = min(vqa_batch_size, len(images) - idx)
        cur_images = images[idx:idx+cur_batch_size]
        cur_scores = profile_metric_flops(
            "VQA",
            lambda: vqa_reward_model(images=cur_images, texts=[text_prompt]),
        )

        scores += cur_scores.reshape(-1).cpu().numpy().tolist()

    output = {
        "scores": scores
    }

    print("Reward calculation took {:.3f}s".format(time.time() - start_time))
    return output


@torch.no_grad()
def process_VideoReward(input):
    global vqa_reward_model
    start_time = time.time()
    videos = input.get("videos")
    audios = input.get("audios")
    text_prompt = input.get("text")
    align_reward_weight = input.get("arw")
    use_norm = True

    if not isinstance(videos, torch.Tensor):
        videos = torch.from_numpy(videos)
    if not isinstance(audios, torch.Tensor):
        audios = torch.from_numpy(audios)

    videos = normalize_video_tensor_shape(videos, target_format=None, return_numpy=False).unsqueeze(0)
    videos = videos.to(vqa_reward_model.device)
    audios = audios.to(vqa_reward_model.device)

    scores = []
    VQ_list = []
    MQ_list = []
    TA_list = []
    for idx in range(0, len(videos), vqa_batch_size):
        cur_batch_size = min(vqa_batch_size, len(videos) - idx)
        cur_videos = videos[idx:idx+cur_batch_size]

        for i in range(cur_batch_size):
            single_video = cur_videos[i]
            single_video = prepare_single_video_for_model(single_video)

            # Try prepare_infer_batch, fallback to manual batch creation
            if hasattr(vqa_reward_model, 'prepare_infer_batch'):
                batch = vqa_reward_model.prepare_infer_batch(
                    videos=[single_video],  # [T, C, H, W]
                    prompts=[text_prompt]
                )
            else:
                # Fallback: use _prepare_inputs directly
                batch = vqa_reward_model._prepare_inputs({
                    'video': single_video.unsqueeze(0) if single_video.dim() == 3 else single_video
                })
            # batch = vqa_reward_model._prepare_inputs(batch)
            rewards = profile_metric_flops(
                "VideoReward",
                lambda: vqa_reward_model.model(return_dict=True, **batch)["logits"],
            )
            rewards = [{'VQ': reward[0].item(), 'MQ': reward[1].item(), 'TA': reward[2].item()} for reward in rewards]

            # cur_score = rewards.mean(dim=1).item()
            for i in range(len(rewards)):
                if use_norm == True:
                    rewards[i] = vqa_reward_model._norm(rewards[i])
                rewards[i]['Overall'] = rewards[i]['VQ'] + rewards[i]['MQ'] + rewards[i]['TA']

            scores.append(rewards[0]['Overall'])
            VQ_list.append(rewards[0]['VQ'])
            MQ_list.append(rewards[0]['MQ'])
            TA_list.append(rewards[0]['TA'])

    # JavisScore
    javis_scores = []
    avh_scores = []
    av_ib_scores = []

    if align_model in ['JavisScore', 'AVHScore', 'AVIB', 'All'] and IMAGEBIND_AVAILABLE:
        cos = nn.CosineSimilarity(dim=-1, eps=1e-6)
        topk_min = input.get("topk_min", 0.4)
        try:
            video_tensor = preprocess_video_for_imagebind(videos, target_frames=videos.shape[1])
            audio_tensor = preprocess_audio_for_imagebind(audios)

            # Normalize audio to float [-1, 1]
            if audios.dtype == torch.int16:
                norm_audios = audios.float() / 32768.0
            else:
                norm_audios = audios

            if norm_audios.ndim == 1:
                norm_audios = norm_audios.unsqueeze(0)

            video_windows, audio_clips = segment_clip_transform(frames=videos[0], wavform=norm_audios, fps=24)
            video_windows_indices = torch.stack([torch.arange(*video_window) for video_window in video_windows], dim=0)

            # for AVHScore
            inputs = {
                imagebind_model_module.ModalityType.VISION: video_tensor.to(vqa_reward_model.device),
                imagebind_model_module.ModalityType.AUDIO: audio_tensor.to(vqa_reward_model.device),
            }
            embeddings = profile_metric_flops(
                "AVHScore",
                lambda: javis_imagebind_model(inputs),
            )
            embed_frames = embeddings[imagebind_model_module.ModalityType.VISION]  # shape(1,1024)
            embed_audio = embeddings[imagebind_model_module.ModalityType.AUDIO]    # shape(1,1024)
            
            if align_model in ['AVHScore', 'All']:
                avh_score = cos(embed_frames, embed_audio).mean().item() #* 1000
                avh_scores.append(avh_score)

            if align_model in ['JavisScore', 'All']:
                # JavisScore
                javis_inputs = {
                    imagebind_model_module.ModalityType.AUDIO: audio_clips,
                }
                
                M, N = video_windows_indices.shape[:2]
                embeddings = profile_metric_flops(
                    "JavisScore",
                    lambda: javis_imagebind_model(javis_inputs),
                )
                embed_video = embed_frames[video_windows_indices.flatten()].view(M, N, -1)  # shape(M,N,1024)
                embed_audio = embeddings[imagebind_model_module.ModalityType.AUDIO].unsqueeze(1)    # shape(M,1,1024)
                
                # JavisScore 계산 로직
                javis_score_clip = cos(embed_video, embed_audio)  # shape(M,N)
                k = topk_min if isinstance(topk_min, int) else int(N * topk_min)
                topk_values, _ = torch.topk(javis_score_clip, k, dim=1, largest=False, sorted=False)
                javis_score_window = topk_values.mean(dim=1) # [M]
                javis_score = javis_score_window.mean(dim=0).item()
                javis_scores.append(javis_score)

            if align_model in ['AVIB', 'All']:
                # AV-IB (Global ImageBind)
                # Re-use audio_tensor [1, 3, 1, 128, 204]
                # Preprocess video for Global ImageBind [15, 3, 2, 224, 224]
                video_tensor_global = preprocess_video_global_for_imagebind(videos, target_frames=videos.shape[1])
                # Add batch dim: [1, 15, 3, 2, 224, 224]
                video_tensor_global = video_tensor_global.unsqueeze(0)

                inputs_global = {
                    imagebind_model_module.ModalityType.VISION: video_tensor_global.to(vqa_reward_model.device),
                    imagebind_model_module.ModalityType.AUDIO: audio_tensor.to(vqa_reward_model.device)
                }
                embeddings_global = profile_metric_flops(
                    "AVIB",
                    lambda: javis_imagebind_model(inputs_global),
                )
                
                embed_frames_global = embeddings_global[imagebind_model_module.ModalityType.VISION] 
                embed_audio_global = embeddings_global[imagebind_model_module.ModalityType.AUDIO]
                
                av_ib_score = cos(embed_frames_global, embed_audio_global).mean().item()
                av_ib_scores.append(av_ib_score)

        except Exception as e:
            print(f"Error processing JavisScore with tensors: {e}")

    # audio model
    clap_scores = []
    if audio_model == "clap":
        try:
            audio_numpy = audios.cpu().numpy()
            
            if audio_numpy.ndim > 1:
                audio_numpy = audio_numpy.squeeze()
            
            inputs = processor(
                text=[text_prompt],
                audios=[audio_numpy],
                return_tensors="pt", 
                padding=True, 
                sampling_rate=48000 
            ).to(device=device)
            
            outputs = profile_metric_flops(
                "CLAP",
                lambda: audio_align_model(**inputs),
            )
            
            clap_score = audio_cos(outputs.text_embeds, outputs.audio_embeds).mean().item()
            clap_scores.append(clap_score)
            print('CLAP score: ', clap_score)
            
        except Exception as e:
            print(f"Error processing CLAP: {e}")
            clap_scores.append(0.0)

    elif audio_model == "TAIB":
        try:
            inputs = {
                imagebind_model_module.ModalityType.TEXT: imagebind_data_module.load_and_transform_text([text_prompt], vqa_reward_model.device),
            }

            if 'embed_audio_global' in locals():
                audio_embeds = embed_audio_global
            else:
                if 'audio_tensor' not in locals():
                    audio_tensor = preprocess_audio_for_imagebind(audios)
                inputs[imagebind_model_module.ModalityType.AUDIO] = audio_tensor.to(vqa_reward_model.device)

            embeddings = profile_metric_flops(
                "TAIB",
                lambda: javis_imagebind_model(inputs),
            )
            text_embeds = embeddings[imagebind_model_module.ModalityType.TEXT]
            
            if 'embed_audio_global' not in locals():
                audio_embeds = embeddings[imagebind_model_module.ModalityType.AUDIO]

            cos = torch.nn.CosineSimilarity(dim=1, eps=1e-6)
            sim_ta = cos(text_embeds, audio_embeds).mean().item()
            
            clap_scores.append(sim_ta)
            print('TAIB score: ', sim_ta)
        except Exception as e:
            print(f"Error processing TAIB: {e}")
            clap_scores.append(0.0)

        
    output = {
        "scores": scores,
        "VQ": VQ_list,
        "MQ": MQ_list,
        "TA": TA_list,
    }
    
    if javis_scores:
        output["JS"] = javis_scores
    if avh_scores:
        output["AVH"] = avh_scores
    if av_ib_scores:
        output["AVIB"] = av_ib_scores

    if audio_model == "clap":
        output["CLAP"] = clap_scores
    elif audio_model == "TAIB":
        output["CLAP"] = clap_scores

    print("Reward calculation took {:.3f}s".format(time.time() - start_time))
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--gpu", type = str, default = "0")
    parser.add_argument("--addr", type = int, default = 5000)
    parser.add_argument("--vqa_model", type = str, default = "clip-flant5-xxl")
    parser.add_argument("--vqa_batch_size", type = int, default = 32)
    parser.add_argument("--reward_model", type = str, default = "vqascore", choices = ["vqascore", "VideoReward"])
    parser.add_argument("--align_model", type = str, default = "JavisScore", choices = ["av_align", "JavisScore", "AVHScore", "AVIB", "All", "None"])
    parser.add_argument("--align_weight", type = float, default = 0.5)
    parser.add_argument("--javis_topk_min", type = float, default = 0.4, help = "topk_min parameter for JavisScore")
    parser.add_argument("--audio_model", type = str, default = "clap", choices = ["clap", "TAIB", "None"])
    parser.add_argument("--print_flops", action="store_true", help="Profile and print FLOPs once for each metric on first execution")
    args = parser.parse_args()

    device = "cuda:{}".format(args.gpu)
    vqa_batch_size = args.vqa_batch_size
    print_flops = args.print_flops
    
    # VQA model
    if args.reward_model == "vqascore":
        RemoteVQAManager.register("process_VQA", callable = process_VQA)
        import t2v_metrics
        vqa_model = args.vqa_model
        print("Loading VQA model {}...".format(vqa_model))
        vqa_reward_model = t2v_metrics.get_score_model(model = vqa_model, device = device)
    elif args.reward_model == "VideoReward":
        RemoteVQAManager.register("process_VideoReward", callable = process_VideoReward)
        # Ensure our JavisDiT_ITS paths are prioritized
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        if project_root not in sys.path:
            sys.path.insert(0, project_root)
        videoalign_path = os.path.abspath(os.path.join(project_root, "VideoAlign"))
        if videoalign_path not in sys.path:
            sys.path.insert(0, videoalign_path)
        from VideoAlign.inference import VideoVLMRewardInference
        print("Initializing VideoReward model...")
        load_from_pretrained = './checkpoints/VideoReward'
        vqa_reward_model = VideoVLMRewardInference(load_from_pretrained, device=device, dtype=torch.bfloat16)

    # align_model 초기화
    align_model = args.align_model
    if args.align_model == 'av_align':
        print("Initializing AV-Align model...")
        from reward_model.av_align import av_align
        align_reward_model = av_align.AVAlignModel(device=device)
    elif args.align_model in ['JavisScore', 'AVHScore', 'AVIB', 'All'] or args.audio_model == 'TAIB':
        try:
            print("Initializing ImageBind model for JavisScore...")

            # Add local ImageBind repo to path first
            imagebind_repo = os.path.abspath("./ImageBind")
            if imagebind_repo not in sys.path:
                sys.path.insert(0, imagebind_repo)

            # Try to import imagebind from local repo
            try:
                from imagebind import models
                from imagebind.models import imagebind_model
                from imagebind.models.imagebind_model import ModalityType
                print("Using local ImageBind repository")
            except ImportError:
                # Fallback: try installed package
                if imagebind_repo in sys.path:
                    sys.path.remove(imagebind_repo)
                from imagebind import models
                from imagebind.models import imagebind_model
                from imagebind.models.imagebind_model import ModalityType
                print("Using installed ImageBind package")

            # Create global reference for imagebind_model module
            imagebind_model_module = imagebind_model

            IMAGEBIND_AVAILABLE = True
            print("Successfully imported ImageBind modules")

            # Load from local checkpoint if available
            imagebind_ckpt_path = "./checkpoints/imagebind_huge.pth"
            if os.path.exists(imagebind_ckpt_path):
                print(f"Loading ImageBind from {imagebind_ckpt_path}")
                javis_imagebind_model = imagebind_model.imagebind_huge(pretrained=False)
                ckpt = torch.load(imagebind_ckpt_path, map_location=device)
                javis_imagebind_model.load_state_dict(ckpt)
                print(f"Loaded ImageBind from {imagebind_ckpt_path}")
            else:
                print("Loading ImageBind from pretrained (downloading)...")
                javis_imagebind_model = imagebind_model.imagebind_huge(pretrained=True)

            javis_imagebind_model.eval()
            javis_imagebind_model.to(device)

        except ImportError as e:
            print(f"Warning: Could not import ImageBind: {e}")
            print(f"JavisScore and related models will not be available")
            IMAGEBIND_AVAILABLE = False
            sys.exit(1)

    else:
        print('No align model')

    # audio model 초기화
    audio_model = args.audio_model
    if args.audio_model == "clap":
        try:
            print("Initializing CLAP model...")
            audio_align_model = ClapModel.from_pretrained("laion/clap-htsat-unfused").eval()
            processor = AutoProcessor.from_pretrained("laion/clap-htsat-unfused")
            audio_align_model.to(device=device)
            audio_cos = torch.nn.CosineSimilarity(dim=1, eps=1e-6)
        except ImportError as e:
            print(f"Error importing CLAP: {e}")
            sys.exit(1)
    elif args.audio_model == "TAIB":
        if not IMAGEBIND_AVAILABLE:
             print("Error: ImageBind not loaded for TAIB")
             sys.exit(1)
        print("Using ImageBind for TAIB...")

    manager = RemoteVQAManager(address=("localhost", int(args.addr)), authkey=b"secret")
    server = manager.get_server()
    print("Server started... Listening for requests.")
    server.serve_forever()
