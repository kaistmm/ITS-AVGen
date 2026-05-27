from tqdm import tqdm
import pandas as pd
import numpy as np
import sys
import os
import os.path as osp
import shutil
import random
import math
from typing import List, Dict, Optional

from PIL import Image
import cv2

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoProcessor, ClapModel

from .AudioCLIP.get_embedding import load_audioclip_pretrained, get_audioclip_embeddings_scores
from .utils import polynomial_mmd, Extract_CAVP_Features

sys.path.append(os.path.join(os.path.dirname(__file__), "ImageBind"))
from .ImageBind.imagebind import data
from .ImageBind.imagebind.models import imagebind_model
from .ImageBind.imagebind.models.imagebind_model import ModalityType

from .dataset import (
    create_dataloader, 
    create_dataloader_for_fvd_vanilla, create_dataloader_for_fvd_mmdiff
)


def calc_imagebind_score(video_list, audio_list, prompt_list, audio_prompt_list=None,
                         device='cuda:0', cat2indices: List[List[List[int]]] = None, bs=1):
    # Original code from "https://github.com/sonyresearch/svg_baseline"

    model = imagebind_model.imagebind_huge(pretrained=True)
    model.eval()
    model.to(device)

    if audio_prompt_list is None:
        audio_prompt_list = prompt_list
    text_embeds, audio_text_embeds, video_embeds, audio_embeds = [], [], [], []
    for i in tqdm(range(0, len(video_list), bs), desc='ib_score'):
        prompts = prompt_list[i:i+bs] + audio_prompt_list[i:i+bs]
        videos, audios = video_list[i:i+bs], audio_list[i:i+bs]
        inputs = {
            ModalityType.TEXT: data.load_and_transform_text(prompts, device),
            ModalityType.VISION: data.load_and_transform_video_data(videos, device),
            ModalityType.AUDIO: data.load_and_transform_audio_data(audios, device),
        }

        with torch.no_grad():
            embeddings = model(inputs)

        text_embed, audio_text_embed = embeddings[ModalityType.TEXT].chunk(2, dim=0)
        text_embeds.append(text_embed)
        audio_text_embeds.append(audio_text_embed)
        video_embeds.append(embeddings[ModalityType.VISION])
        audio_embeds.append(embeddings[ModalityType.AUDIO])

    text_embeds, audio_text_embeds = torch.cat(text_embeds), torch.cat(audio_text_embeds)
    video_embeds, audio_embeds = torch.cat(video_embeds), torch.cat(audio_embeds)

    cos = torch.nn.CosineSimilarity(dim=1, eps=1e-6)
    sim_tv_list = cos(text_embeds, video_embeds)
    sim_tv = sim_tv_list.mean().item()
    sim_ta_list = cos(audio_text_embeds, audio_embeds)
    sim_ta = sim_ta_list.mean().item()
    sim_av_list = cos(video_embeds, audio_embeds)
    sim_av = sim_av_list.mean().item()
    
    if cat2indices is not None:
        sim_tv = {'overall': sim_tv, 'all': sim_tv_list.tolist()}
        sim_ta = {'overall': sim_ta, 'all': sim_ta_list.tolist()}
        sim_av = {'overall': sim_av, 'all': sim_av_list.tolist()}
        for ai, index_list in enumerate(cat2indices):
            sim_tv[ai], sim_ta[ai], sim_av[ai] = [], [], []
            for ci, indices in enumerate(index_list):
                text_embeds_sub, audio_text_embeds_sub, video_embeds_sub, audio_embeds_sub = \
                    text_embeds[indices], audio_text_embeds[indices], video_embeds[indices], audio_embeds[indices]
                sim_tv[ai].append(cos(text_embeds_sub, video_embeds_sub).mean().item())
                sim_ta[ai].append(cos(audio_text_embeds_sub, audio_embeds_sub).mean().item())
                sim_av[ai].append(cos(video_embeds_sub, audio_embeds_sub).mean().item())

    return sim_tv, sim_ta, sim_av

def calc_clap_score(audio_list, prompt_list, device='cuda:0', cat2indices=None):
    # Original code from "https://github.com/sonyresearch/svg_baseline"
    model = ClapModel.from_pretrained("laion/clap-htsat-unfused").eval()
    processor = AutoProcessor.from_pretrained("laion/clap-htsat-unfused")
    model.to(device=device)
    cos = torch.nn.CosineSimilarity(dim=1, eps=1e-6)

    dataloader = create_dataloader(
        metric='clap-score', 
        audio_path_list=audio_list,
        prompt_list=prompt_list,
        sr=48000,   # CLAP requires sample_rate=48000
        max_audio_len_s=None,
        batch_size=1
    )

    score_list, index_list = [], []
    for audios, prompts, indices in tqdm(dataloader, desc='clapscore'):
        assert len(audios) == len(prompts) == len(indices) == 1
        inputs = processor(text=prompts[0], audios=audios[0].squeeze(), 
                           return_tensors="pt", padding=True, 
                           sampling_rate=48000)   # CLAP requires sample_rate=48000
        inputs.to(device=device)
        outputs = model(**inputs)
        scores = cos(outputs.text_embeds, outputs.audio_embeds).mean()
        score_list.append(scores)
        index_list.append(indices)
    
    indices = torch.cat(index_list).flatten()
    clap_scores = torch.tensor(score_list)[indices.argsort()]

    clap_score = clap_scores.mean().item()

    if cat2indices is not None:
        clap_score = {'overall': clap_score, 'all': clap_scores.tolist()}
        for ai, index_list in enumerate(cat2indices):
            clap_score[ai] = []
            for ci, indices in enumerate(index_list):
                clap_score[ai].append(torch.mean(clap_scores[indices]).item())

    return clap_score


def calc_av_align(video_list, audio_list, cat2indices=None, size=None, return_score_list=False):
    # Original code from "https://yzxing87.github.io/Seeing-and-Hearing/"

    dataloader = create_dataloader(
        metric='av-align', 
        video_path_list=video_list,
        audio_path_list=audio_list,
        size=size,
        batch_size=1
    )

    align_score_list, index_list = [], []
    for align_score, index in tqdm(dataloader, desc='av-align'):
        align_score_list.append(align_score)
        index_list.append(index)

    indices = torch.cat(index_list).argsort()
    align_scores = torch.cat(align_score_list)[indices]

    align_score = align_scores.mean().item()

    if cat2indices is not None:
        align_score = {'overall': align_score, 'all': align_scores.tolist()}
        for ai, index_list in enumerate(cat2indices):
            align_score[ai] = []
            for ci, indices in enumerate(index_list):
                align_score[ai].append(torch.mean(align_scores[indices]).item())

    if return_score_list:
        return align_score, align_scores
    else:
        return align_score


def calc_av_score(video_list, audio_list, prompt_list, device='cuda:0', cat2indices=None,
                  sample_rate=16000, window_size_s=0.5, window_overlap_s=0, topk_min=0.4,
                  return_score_list=False):
    
    model = imagebind_model.imagebind_huge(pretrained=True)
    model.eval()
    model.to(device)

    dataloader = create_dataloader(
        metric='av-score', 
        video_path_list=video_list,
        audio_path_list=audio_list,
        prompt_list=prompt_list,
        sample_rate=sample_rate,
        window_size_s=window_size_s,
        window_overlap_s=window_overlap_s,
        batch_size=1
    )

    avh_score_list, javis_score_list, index_list = [], [], []
    cos = nn.CosineSimilarity(dim=-1, eps=1e-6)
    for avh_inputs, javis_inputs, video_windows_indices, index in tqdm(dataloader, desc='av-score'):
        assert len(index) == video_windows_indices.shape[0] == 1

        # image shape: (B,C,H,W), video shape: (B,15,C,2,H,W), audio shape(B,3,C,T,S), 
        avh_inputs = {k: v[0].to(device) for k, v in avh_inputs.items()}
        javis_inputs = {k: v[0].to(device) for k, v in javis_inputs.items()}
        video_windows_indices = video_windows_indices[0]

        # for AVHScore
        with torch.no_grad():
            embeddings = model(avh_inputs)
        embed_frames = embeddings[ModalityType.VISION]  # shape(T,1024)
        embed_audio = embeddings[ModalityType.AUDIO]    # shape(1,1024)
        avh_score = cos(embed_frames, embed_audio).mean().item() #* 1000
        avh_score_list.append(avh_score)

        # for JavisScore
        M, N = video_windows_indices.shape[:2]
        with torch.no_grad():
            embeddings = model(javis_inputs)
        embed_video = embed_frames[video_windows_indices.flatten()].view(M, N, -1)  # shape(M,N,1024)
        embed_audio = embeddings[ModalityType.AUDIO].unsqueeze(1)    # shape(M,1,1024)

        javis_score_clip = cos(embed_video, embed_audio)  # shape(M,N)
        k = topk_min if isinstance(topk_min, int) else int(N * topk_min)
        topk_values, _ = torch.topk(javis_score_clip, k, dim=1, largest=False, sorted=False)
        javis_score_window = topk_values.mean(dim=1)
        javis_score = javis_score_window.mean(dim=0).item()

        # javis_score_clip = cos(embed_video, embed_audio).mean(dim=1)  # shape(M)
        # k = topk_min if isinstance(topk_min, int) else math.ceil(M * topk_min)
        # topk_values, _ = torch.topk(javis_score_clip, k, dim=0, largest=False, sorted=False)
        # javis_score = topk_values.mean().item()
        
        javis_score_list.append(javis_score)

        index_list.append(index[0])
    
    indices = torch.tensor(index_list).argsort()
    avh_scores = torch.tensor(avh_score_list)[indices]
    javis_scores = torch.tensor(javis_score_list)[indices]

    avh_score, javis_score = avh_scores.mean().item(), javis_scores.mean().item()

    if cat2indices is not None:
        avh_score = {'overall': avh_score, 'all': avh_scores.tolist()}
        javis_score = {'overall': javis_score, 'all': javis_scores.tolist()}
        for ai, index_list in enumerate(cat2indices):
            avh_score[ai], javis_score[ai] = [], []
            for ci, indices in enumerate(index_list):
                avh_score[ai].append(torch.mean(avh_scores[indices]).item())
                javis_score[ai].append(torch.mean(javis_scores[indices]).item())

    if return_score_list:
        return avh_score, javis_score, avh_scores, javis_scores
    else:
        return avh_score, javis_score


def calc_audio_score(gt_audio_list, pred_audio_list, prompt_list, device='cuda:0', 
                     eval_num=None, bs=8, **kwargs):
    ############### Part I - FAD ###############
    from .AudioCLIP.get_embedding import preprocess_audio

    audioclip = load_audioclip_pretrained(device)

    real_loader, fake_loader = create_dataloader_for_fvd_vanilla(
        gt_audio_list, gt_audio_list, pred_audio_list, pred_audio_list, 
        audio_sr=kwargs.pop('audio_sr'),
        num_workers=kwargs.pop('num_workers'),
        audio_only=True,
        **kwargs
    )

    loader_dict = {'real': real_loader, 'fake': fake_loader}
    embed_dict = {}
    for t, loader in loader_dict.items():
        audio_embeds = []
        cnt = 0
        for _, sample in enumerate(tqdm(loader, desc=f'fad: {t}')):
            audio_sample = sample['audio'].to(device)

            audios = preprocess_audio(audio_sample).to(device)

            with torch.no_grad():
                audioclip_audio_embed = audioclip(audio=audios, video=None)[0][0][0]
            assert audio_sample.shape[0] == audioclip_audio_embed.shape[0]
            
            audio_embeds.append(audioclip_audio_embed)

            cnt += audio_sample.shape[0]
            if eval_num and cnt >= eval_num: 
                break

        embed_dict[t] = torch.cat(audio_embeds)
    
    # fad = frechet_distance(embed_dict['fake'], embed_dict['real']).item() * 10000
    ############### Part I - FAD ###############


    ############### Part II - IB_TA ###############
    model = imagebind_model.imagebind_huge(pretrained=True)
    model.eval()
    model.to(device)

    text_embeds, audio_embeds = [], []
    # fast enough in a for-loop
    for i in tqdm(range(0, len(pred_audio_list), bs), desc='ib_score'):
        prompts, audios = prompt_list[i:i+bs], pred_audio_list[i:i+bs]
        inputs = {
            ModalityType.TEXT: data.load_and_transform_text(prompts, device),
            ModalityType.AUDIO: data.load_and_transform_audio_data(audios, device),
        }

        with torch.no_grad():
            embeddings = model(inputs)

        text_embeds.append(embeddings[ModalityType.TEXT])
        audio_embeds.append(embeddings[ModalityType.AUDIO])

    text_embeds, audio_embeds = torch.cat(text_embeds), torch.cat(audio_embeds)

    cos = torch.nn.CosineSimilarity(dim=1, eps=1e-6)
    sim_ta = cos(text_embeds, audio_embeds).mean().item()
    ############### Part II - IB_TA ###############
    

    ############### Part III - CLAP ###############
    model = ClapModel.from_pretrained("laion/clap-htsat-unfused").eval()
    processor = AutoProcessor.from_pretrained("laion/clap-htsat-unfused")
    model.to(device=device)
    cos = torch.nn.CosineSimilarity(dim=1, eps=1e-6)

    dataloader = create_dataloader(
        metric='clap-score', 
        audio_path_list=pred_audio_list,
        prompt_list=prompt_list,
        sr=48000,   # CLAP requires sample_rate=48000
        max_audio_len_s=None,
        batch_size=1
    )

    score_list, index_list = [], []
    for audios, prompts, indices in tqdm(dataloader, desc='clapscore'):
        assert len(audios) == len(prompts) == len(indices) == 1
        inputs = processor(text=prompts[0], audios=audios[0].squeeze(), 
                           return_tensors="pt", padding=True, 
                           sampling_rate=48000)  # CLAP requires sample_rate=48000
        inputs.to(device=device)
        outputs = model(**inputs)
        scores = cos(outputs.text_embeds, outputs.audio_embeds).mean()
        score_list.append(scores)
        index_list.append(indices)
    
    indices = torch.cat(index_list).flatten()
    clap_scores = torch.tensor(score_list)[indices.argsort()]

    clap_score = clap_scores.mean().item()
    ############### Part III - CLAP ###############

    return fad, sim_ta, clap_score


def calc_vqa_score(video_list, prompt_list, device='cuda:0', cat2indices=None, 
                   model_name="clip-flant5-xxl", num_frames=8):
    """
    Calculate VQAScore for video-text alignment using t2v_metrics library.
    This function is based on the VQAScore implementation from:
    https://github.com/linzhiqiu/t2v_metrics
    
    Args:
        video_list: List of video file paths
        prompt_list: List of text prompts
        device: Device for computation
        cat2indices: Category indices for detailed evaluation
        model_name: Model name for VQA scoring (default: "clip-flant5-xxl")
        num_frames: Number of frames to sample from each video
    
    Returns:
        VQA score(s)
    """
    try:
        import t2v_metrics
    except ImportError:
        raise ImportError("Please install t2v_metrics library for VQAScore: pip install t2v-metrics")
    
    # Initialize VQAScore model
    vqa_scorer = t2v_metrics.VQAScore(model=model_name)
    
    vqa_score_list, indices_list = [], []
    
    for i, (video_path, prompt) in enumerate(tqdm(zip(video_list, prompt_list), desc='vqascore')):
        try:
            # Use the VQAScore model to compute score
            score = vqa_scorer(images=[video_path], texts=[prompt], num_frames=num_frames)
            vqa_score_list.append(score[0].item())  # Extract scalar from tensor
        except Exception as e:
            print(f"Warning: Failed to compute VQAScore for {video_path}: {e}")
            vqa_score_list.append(0.0)  # Default score for failed cases
        
        indices_list.append(i)
    
    indices = torch.tensor(indices_list)
    vqa_scores = torch.tensor(vqa_score_list)[indices.argsort()]
    
    vqa_score = vqa_scores.mean().item()
    
    if cat2indices is not None:
        vqa_score = {'overall': vqa_score, 'all': vqa_scores.tolist()}
        for ai, index_list in enumerate(cat2indices):
            vqa_score[ai] = []
            for ci, indices in enumerate(index_list):
                vqa_score[ai].append(torch.mean(vqa_scores[indices]).item())
    
    return vqa_score