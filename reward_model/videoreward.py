import os 
import torch
from dataclasses import dataclass
import torchvision.transforms as transforms
import sys

from . import shared_modules as sm
from .utils.extra_utils import ignore_kwargs
# 절대 import로 변경
sys.path.append(os.path.dirname(__file__))
from utils.video_preprocessing import normalize_video_tensor_shape, normalize_audio_tensor_shape
from .vqa_server import RemoteVQAManager

class VideoRewardModel:
    @ignore_kwargs
    @dataclass
    class Config:
        batch_size: int = 1
        n_particles: int = 1
        text_prompt: str = ""
        align_weight: float = 0.0
        vqa_server_addr: int = 5000
        benchmark: bool = False
        img_idx: int = 0

    def __init__(self, cfg=None):
        self.cfg = self.Config(**cfg) if cfg else self.Config()

        print(f"[VideoRewardModel] Connecting to vqa_server at localhost:{self.cfg.vqa_server_addr}")
        # VQA 서버 연결 설정
        RemoteVQAManager.register("process_VideoReward")
        sam_manager = RemoteVQAManager(address=("localhost", self.cfg.vqa_server_addr), authkey=b"secret")
        print(f"[VideoRewardModel] RemoteVQAManager created")
        sam_manager.connect()
        print(f"[VideoRewardModel] Connected to vqa_server")
        self.vqa_function = sam_manager.process_VideoReward
        print(f"[VideoRewardModel] process_VideoReward function bound")

    def _preprocess_tensor(self, data, is_audio=False):
        """텐서 데이터를 numpy로 변환하고 정규화"""
        if isinstance(data, torch.Tensor):
            if is_audio:
                return normalize_audio_tensor_shape(data).detach().cpu().numpy()
            else:
                return normalize_video_tensor_shape(data).detach().cpu().numpy()
        return data

    def __call__(self, videos, audios=None, prompt=None, step=None):
        videos = self._preprocess_tensor(videos, is_audio=False)
        audios = self._preprocess_tensor(audios, is_audio=True) if audios is not None else None

        # VQA 서버 호출
        output = self.vqa_function({
            "videos": videos,
            "audios": audios,
            "text": prompt if prompt is not None else self.cfg.text_prompt,
            "arw": self.cfg.align_weight,
        })

        result_keys = ["scores", "VQ", "MQ", "TA", "JS", "AVH", "CLAP", "AVIB"]
        results = [torch.tensor(output.get(key, 0.0)) for key in result_keys]
        
        return tuple(results)
