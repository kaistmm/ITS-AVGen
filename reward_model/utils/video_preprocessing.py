"""
Video preprocessing utilities for VideoReward and VQA Server
비디오 텐서 형태 변환을 위한 공통 유틸리티 함수들
"""
import torch
import numpy as np

import sys
import os
from torchvision import transforms

# Add ImageBind path
current_dir = os.path.dirname(os.path.abspath(__file__))
imagebind_path = os.path.abspath(os.path.join(current_dir, "../../eval/javisbench/src/ImageBind"))
if imagebind_path not in sys.path:
    sys.path.append(imagebind_path)

from imagebind import data
from imagebind import data
from imagebind.data import ConstantClipsPerVideoSampler, SpatialCrop
from pytorchvideo import transforms as pv_transforms
from torchvision.transforms._transforms_video import NormalizeVideo

def normalize_video_tensor_shape(videos, target_format="BTCHW", return_numpy=False):
    """
    비디오 텐서를 표준 형태로 변환하는 공통 함수
    
    Args:
        videos: 입력 비디오 텐서 (torch.Tensor 또는 numpy.ndarray)
        target_format: 목표 형태 ("BTCHW" 또는 "TCHW")
        return_numpy: True시 numpy 배열 반환, False시 torch 텐서 반환
    
    Returns:
        변환된 비디오 텐서
    """
    if torch.is_tensor(videos):
        videos = videos.detach().cpu()
    elif isinstance(videos, np.ndarray):
        videos = torch.from_numpy(videos)
    
    if videos.dim() == 5:
        return _normalize_5d_tensor(videos)
    elif videos.dim() == 4:
        return _normalize_4d_tensor(videos)
    elif videos.dim() == 3:
        return _normalize_3d_tensor(videos)
    else:
        raise ValueError(f"Unsupported tensor dimension: {videos.dim()}. Expected 3D, 4D, or 5D tensor.")


def normalize_audio_tensor_shape(audios):
    """
    오디오 텐서를 표준 형태로 변환하는 공통 함수 (C=1 for audio)
    
    Args:
        audios: 입력 오디오 텐서
    
    Returns:
        변환된 오디오 텐서 [B, T, C, H, W] (C=1)
    """
    if torch.is_tensor(audios):
        audios = audios.detach().cpu()
    elif isinstance(audios, np.ndarray):
        audios = torch.from_numpy(audios)
    
    if audios.dim() == 5:
        return _normalize_5d_audio_tensor(audios)
    elif audios.dim() == 4:
        return _normalize_4d_audio_tensor(audios)
    elif audios.dim() == 3:
        return _normalize_3d_audio_tensor(audios)
    else:
        raise ValueError(f"Unsupported tensor dimension: {audios.dim()}. Expected 3D, 4D, or 5D tensor.")


def _normalize_5d_audio_tensor(tensor):
    """5D 오디오 텐서 [B, T, C, H, W] 또는 [B, T, H, W, C] 처리 (C=1)"""
    # 마지막 차원이 1이면 channels이 마지막에 있음
    if tensor.shape[-1] == 1:
        tensor = tensor.permute(0, 1, 4, 2, 3)  # [B, T, H, W, C] -> [B, T, C, H, W]
    return tensor


def _normalize_4d_audio_tensor(tensor):
    """4D 오디오 텐서 [T, C, H, W] 또는 [T, H, W, C] 처리 (C=1)"""
    # 마지막 차원이 1이면 channels이 마지막에 있음
    if tensor.shape[-1] == 1:
        tensor = tensor.permute(0, 3, 1, 2)  # [T, H, W, C] -> [T, C, H, W]
    # 배치 차원 추가
    tensor = tensor.unsqueeze(0)  # [T, C, H, W] -> [1, T, C, H, W]
    return tensor


def _normalize_3d_audio_tensor(tensor):
    """3D 오디오 텐서 [C, H, W] 또는 [H, W, C] 처리 (C=1)"""
    if tensor.shape[-1] == 1:  # [H, W, C]
        tensor = tensor.permute(2, 0, 1)  # [H, W, C] -> [C, H, W]
    # 배치와 시간 차원 추가
    tensor = tensor.unsqueeze(0).unsqueeze(0)  # [C, H, W] -> [1, 1, C, H, W]
    return tensor


def _normalize_5d_tensor(videos):
    """5차원 텐서 [B, T, ?, ?, ?] 정규화"""
    shape = videos.shape
    
    if shape[2] == 3:  # [B, T, C, H, W] - 이미 올바른 형태
        return videos
    elif shape[4] == 3:  # [B, T, H, W, C]
        return videos.permute(0, 1, 4, 2, 3)  # -> [B, T, C, H, W]
    elif shape[3] == 3:  # [B, T, H, C, W] - 비정상적이지만 처리
        return videos.permute(0, 1, 3, 2, 4)  # -> [B, T, C, H, W]
    else:
        return videos


def _normalize_4d_tensor(videos):
    """4차원 텐서 [T, ?, ?, ?] 정규화"""
    shape = videos.shape
    
    if shape[1] == 3:  # [T, C, H, W] - 이미 올바른 형태
        return videos
    elif shape[3] == 3:  # [T, H, W, C]
        return videos.permute(0, 3, 1, 2)  # -> [T, C, H, W]
    elif shape[2] == 3:  # [T, H, C, W] - 비정상적이지만 처리
        return videos.permute(0, 2, 1, 3)  # -> [T, C, H, W]
    else:
        # [C, T, H, W] 가능성 체크
        if shape[0] < shape[1] and shape[0] <= 16:  # channels이 첫 번째 차원
            return videos.permute(1, 0, 2, 3)  # [C, T, H, W] -> [T, C, H, W]
        else:
            return videos


def _normalize_3d_tensor(videos):
    """3차원 텐서 [?, ?, ?] 정규화"""
    shape = videos.shape
    
    if shape[0] == 3 and shape[0] < min(shape[1], shape[2]):  # [C, H, W]
        return videos
    elif shape[2] == 3:  # [H, W, C]
        return videos.permute(2, 0, 1)  # -> [C, H, W]
    elif shape[1] == 3:  # [H, C, W] - 비정상적이지만 처리
        return videos.permute(1, 0, 2)  # -> [C, H, W]
    else:
        return videos


def prepare_single_video_for_model(video_tensor, target_frames=16):
    """
    단일 비디오 텐서를 모델 입력용으로 최종 준비
    vqa_server의 process_VideoReward에서 사용
    
    Args:
        video_tensor: 입력 비디오 텐서 [B, T, C, H, W] 또는 [T, C, H, W]
        target_frames: 목표 프레임 수 (기본값: 16)
    
    Returns:
        torch.Tensor: [T, C, H, W] 형태의 텐서 (target_frames 프레임)
    """
    # 배치 차원 제거 (있는 경우)
    if video_tensor.ndim == 5:
        video_tensor = video_tensor.squeeze(0)  # [B, T, C, H, W] -> [T, C, H, W]
    
    # [C, T, H, W] 형태라면 [T, C, H, W]로 변경
    if video_tensor.ndim == 4:
        if video_tensor.shape[0] == 3 and video_tensor.shape[1] > 3:
            video_tensor = video_tensor.permute(1, 0, 2, 3)
    
    # 프레임 수 조정
    if video_tensor.shape[0] < target_frames:
        # 프레임이 부족하면 반복하여 target_frames으로 맞춤
        repeat_factor = (target_frames + video_tensor.shape[0] - 1) // video_tensor.shape[0]
        video_tensor = video_tensor.repeat(repeat_factor, 1, 1, 1)[:target_frames]
    elif video_tensor.shape[0] > target_frames:
        # 프레임이 많으면 균등하게 서브샘플링
        indices = torch.linspace(0, video_tensor.shape[0] - 1, target_frames).long()
        video_tensor = video_tensor[indices]
    
    return video_tensor


# ImageBind 공통 상수들
IMAGEBIND_CONSTANTS = {
    'video': {
        'target_size': 224,
        'mean': [0.48145466, 0.4578275, 0.40821073],
        'std': [0.26862954, 0.26130258, 0.27577711],
    },
    'audio': {
        'sample_rate': 16000,
        'target_samples': 32000,
        'mel_bins': 128,
        'target_length': 204,
        'mean': -4.268,
        'std': 9.138,
        'frame_length': 25,
        'frame_shift': 10,
    }
}


def _temporal_subsample(tensor, target_frames):
    """시간 축을 따라 균등하게 서브샘플링"""
    if tensor.shape[0] != target_frames:
        indices = torch.linspace(0, tensor.shape[0] - 1, target_frames).long()
        tensor = tensor[indices]
    return tensor


def _normalize_imagebind_video(video_tensor, device):
    """ImageBind 표준값으로 비디오 정규화"""
    mean = torch.tensor(IMAGEBIND_CONSTANTS['video']['mean']).view(3, 1, 1).to(device)
    std = torch.tensor(IMAGEBIND_CONSTANTS['video']['std']).view(3, 1, 1).to(device)
    return (video_tensor - mean.unsqueeze(0)) / std.unsqueeze(0)


def _pad_or_crop_to_length(tensor, target_length, dim=-1):
    """텐서를 지정된 길이로 패딩하거나 자름"""
    import torch.nn.functional as F
    
    current_length = tensor.shape[dim]
    if current_length < target_length:
        pad_size = target_length - current_length
        if dim == -1:
            # 마지막 차원에 패딩
            return F.pad(tensor, (0, pad_size), mode="constant", value=0)
        elif dim == 1 and tensor.ndim == 2:
            # 2D 텐서의 두 번째 차원에 패딩 [mel_bins, time_frames]
            return F.pad(tensor, (0, pad_size), mode="constant", value=0)
        else:
            # 일반적인 경우는 직접 구현
            pad_shape = list(tensor.shape)
            pad_shape[dim] = pad_size
            pad_tensor = torch.zeros(pad_shape, dtype=tensor.dtype, device=tensor.device)
            return torch.cat([tensor, pad_tensor], dim=dim)
    elif current_length > target_length:
        # 크롭
        indices = [slice(None)] * tensor.ndim
        indices[dim] = slice(target_length)
        return tensor[tuple(indices)]
    return tensor


def preprocess_video_for_imagebind(videos, target_frames=16):
    """
    ImageBind 모델을 위한 비디오 전처리
    
    Args:
        videos: 입력 비디오 텐서 [1, T, C, H, W] or [T, C, H, W]
        target_frames: 목표 프레임 수 (기본값: 16)
    
    Returns:
        torch.Tensor: ImageBind 형태의 비디오 텐서 [T, C, H, W]
    """
    import torch.nn.functional as F
    
    # 1. 기본 전처리 ([1,T,C,H,W] -> [T,C,H,W])
    video_tensor = videos.squeeze(0) if videos.dim() == 5 else videos
    video_tensor = video_tensor.float() / 255.0
    
    # 2. 시간 축 서브샘플링
    video_tensor = _temporal_subsample(video_tensor, target_frames)
    
    # 3. 공간 해상도 조정 (ImageBind 표준)
    target_size = IMAGEBIND_CONSTANTS['video']['target_size']
    video_tensor = F.interpolate(video_tensor, size=(target_size, target_size), 
                                mode='bilinear', align_corners=False)
    
    # 4. 정규화 (ImageBind 표준값)
    video_tensor = _normalize_imagebind_video(video_tensor, video_tensor.device)
    
    return video_tensor


def preprocess_audio_for_imagebind(audios):
    """
    ImageBind 모델을 위한 오디오 전처리
    
    Args:
        audios: 입력 오디오 텐서
    
    Returns:
        torch.Tensor: ImageBind 형태의 오디오 텐서 [1, 3, 1, 128, 204]
    """
    import torch.nn.functional as F
    import torchaudio
    
    constants = IMAGEBIND_CONSTANTS['audio']
    
    # 오디오 파형 전처리
    audio_waveform = audios.unsqueeze(0) if audios.dim() == 1 else audios
    audio_waveform = audio_waveform.float() / 32768.0  # int16 -> float32 정규화
    
    # 길이 조정 (target_samples로 맞춤)
    audio_waveform = _pad_or_crop_to_length(audio_waveform, constants['target_samples'], dim=-1)
    
    # 평균 제거
    audio_waveform = audio_waveform - audio_waveform.mean()
    
    # 멜 스펙트로그램 생성
    fbank = torchaudio.compliance.kaldi.fbank(
        audio_waveform,
        htk_compat=True,
        sample_frequency=constants['sample_rate'],
        use_energy=False,
        window_type="hanning",
        num_mel_bins=constants['mel_bins'],
        dither=0.0,
        frame_length=constants['frame_length'],
        frame_shift=constants['frame_shift'],
    )
    
    # 차원 조정 및 길이 맞춤
    fbank = fbank.transpose(0, 1)
    fbank = _pad_or_crop_to_length(fbank, constants['target_length'], dim=1)
    audio_tensor = fbank.unsqueeze(0)
    
    # 정규화 (ImageBind 기본값)
    audio_tensor = (audio_tensor - constants['mean']) / constants['std']
    
    # ImageBind forward를 위한 차원 추가
    # 오디오의 경우: [batch_size, num_clips, channels, mel_bins, time_frames]
    audio_tensor = audio_tensor.unsqueeze(0).unsqueeze(0)  # [1, 1, 1, 128, 204]
    audio_tensor = audio_tensor.repeat(1, 3, 1, 1, 1)  # [1, 3, 1, 128, 204]
    
    return audio_tensor


def preprocess_video_global_for_imagebind(videos, target_frames=16):
    """
    ImageBind-AV (Global)를 위한 비디오 전처리
    Standard ImageBind sampling: 5 clips x 2 frames x 3 spatial crops = 15 chunks
    
    Args:
        videos: 입력 비디오 텐서 [1, T, C, H, W] or [T, C, H, W]
    
    Returns:
        torch.Tensor: ImageBind 형태의 비디오 텐서 [15, 3, 2, 224, 224]
    """
    import torch.nn.functional as F
    
    # 1. 기본 전처리 ([1,T,C,H,W] -> [T,C,H,W])
    video_tensor = videos.squeeze(0) if videos.dim() == 5 else videos
    
    # Ensure float 0-1
    if video_tensor.dtype == torch.uint8:
        video_tensor = video_tensor.float() / 255.0
    elif video_tensor.max() > 1.0:
        video_tensor = video_tensor.float() / 255.0
    
    # [T, C, H, W] -> [C, T, H, W] for pytorchvideo transforms
    video_tensor = video_tensor.permute(1, 0, 2, 3)
    
    num_clips = 5
    num_frames_per_clip = 2
    
    T = video_tensor.shape[1]
    
    # Sample 5 clips
    indices = torch.linspace(0, T - num_frames_per_clip, num_clips).long()
    
    clips = []
    for start_idx in indices:
        # Extract clip: [C, 2, H, W]
        clip = video_tensor[:, start_idx:start_idx+num_frames_per_clip, :, :]
        clips.append(clip)
    
    # Transform: ShortSideScale(224) + Normalize
    video_transform = transforms.Compose([
        pv_transforms.ShortSideScale(224),
        NormalizeVideo(
            mean=IMAGEBIND_CONSTANTS['video']['mean'],
            std=IMAGEBIND_CONSTANTS['video']['std'],
        ),
    ])
    
    clips = [video_transform(clip) for clip in clips]
    
    # Spatial Crop: 3 crops per clip -> Returns list of 15 clips
    # SpatialCrop expects list of [C, T, H, W]
    all_clips = SpatialCrop(224, num_crops=3)(clips)
    
    # Stack: [15, C, T, H, W]
    # ImageBind model expects [B, 15, 3, 2, 224, 224] but here we process single video
    # So we return [15, 3, 2, 224, 224] (C=3, T=2)
    video_tensor = torch.stack(all_clips, dim=0)
    
    return video_tensor


def segment_clip_transform(frames, wavform, fps):
     # for Javis-Score
    window_size_s = 2.0
    window_overlap_s = 1.5
    sample_rate=16000

    video_window_size = int(window_size_s * fps)
    video_window_overlap = int(window_overlap_s * fps)
    video_windows = []
    n_vframes = len(frames)

    audio_window_size = int(window_size_s * sample_rate)
    audio_window_overlap = int(window_overlap_s * sample_rate)
    audio_clips = []
    n_aframes = wavform.shape[1]

    if n_vframes <= video_window_size or n_aframes <= audio_window_size:
        video_windows.append([0, n_vframes])
        audio_clips = transform_audio_clip(wavform)[None]
        return video_windows, audio_clips

    for start in range(0, n_vframes, video_window_size-video_window_overlap):
        start = min(start, n_vframes-video_window_size)
        end = start + video_window_size
        video_windows.append([start, end])
        if end >= n_vframes:
            break

    for start in range(0, n_aframes, audio_window_size-audio_window_overlap):
        start = min(start, n_aframes-audio_window_size)
        end = start + audio_window_size
        audio_clips.append(wavform[:, start:end])
        if end >= n_aframes:
            break

    clip_num = min(len(video_windows), len(audio_clips))
    video_windows, audio_clips = video_windows[:clip_num], audio_clips[:clip_num]
    audio_clips = [transform_audio_clip(clip) for clip in audio_clips]
    audio_clips = torch.stack(audio_clips)

    return video_windows, audio_clips


def transform_audio_clip(
        waveform, num_mel_bins=128, target_length=204,
        clip_duration=2, clips_per_video=3, mean=-4.268, std=9.138,
    ):
    sample_rate = 16000
    clip_sampler = ConstantClipsPerVideoSampler(
        clip_duration=clip_duration, clips_per_video=clips_per_video
    )
    all_clips_timepoints = data.get_clip_timepoints(
        clip_sampler, waveform.size(1) / sample_rate
    )
    all_clips = []
    for clip_timepoints in all_clips_timepoints:
        waveform_clip = waveform[
            :,
            int(clip_timepoints[0] * sample_rate) : 
            int(clip_timepoints[1] * sample_rate)
        ]
        waveform_melspec = data.waveform2melspec(
            waveform_clip, sample_rate, num_mel_bins, target_length
        )
        all_clips.append(waveform_melspec)

    normalize = transforms.Normalize(mean=mean, std=std)
    all_clips = [normalize(ac) for ac in all_clips]
    all_clips = torch.stack(all_clips, dim=0)

    return all_clips
