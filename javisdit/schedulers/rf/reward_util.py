"""Utility functions for reward calculation and processing."""

import torch


def to_tensor_or_zeros(data_list, default_len, device):
    """
    Convert a list to tensor or return zeros tensor if list is empty.
    
    Args:
        data_list: List of values to convert to tensor
        default_len: Length of zero tensor to return if data_list is empty
        device: Device to place the tensor on
        
    Returns:
        torch.Tensor: Converted tensor or zeros tensor
    """
    return torch.tensor(data_list).to(device) if data_list else torch.zeros(default_len).to(device)


def compute_rank_based_rewards(vqa_tensor, align_tensor, additional_tensor=None):
    """
    Compute rank-based rewards from input tensors.
    
    Args:
        vqa_tensor: Tensor of VQA scores
        align_tensor: Tensor of alignment scores
        additional_tensor: Optional additional tensor (e.g., audio scores)
        
    Returns:
        torch.Tensor: Normalized rank-based rewards
    """
    vqa_ranks = torch.argsort(torch.argsort(vqa_tensor, descending=False)).float()
    align_ranks = torch.argsort(torch.argsort(align_tensor, descending=False)).float()
    
    if additional_tensor is not None:
        audio_ranks = torch.argsort(torch.argsort(additional_tensor, descending=False)).float()

    if len(vqa_tensor) > 1:
        vqa_ranks = vqa_ranks / (len(vqa_tensor) - 1)
        align_ranks = align_ranks / (len(align_tensor) - 1)
        if additional_tensor is not None:
            audio_ranks = audio_ranks / (len(additional_tensor) - 1)
    
    if additional_tensor is not None:
        rank_rewards = (vqa_ranks + align_ranks + audio_ranks) / 3.0
    else:
        rank_rewards = (vqa_ranks + align_ranks) / 2.0
    
    return rank_rewards
