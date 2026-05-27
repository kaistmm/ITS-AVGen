## <div align="center">[TMLR] Inference Time Scaling for Joint Audio-Video Generation</div>

<div align="center">

**[Jaemin Jung](https://jung-jaemin.github.io/)**<sup>1</sup>, [Kyeongha Rho](https://kyeongharho.github.io/)<sup>1</sup>, [Inkyu Shin](https://dlsrbgg33.github.io/)<sup>2</sup>, [Joon Son Chung](https://mm.kaist.ac.kr/joon/)<sup>1</sup>

<sup>1</sup> KAIST, <sup>2</sup> Luma AI

[[`Paper`](https://openreview.net/forum?id=MHNFjjm5nO)] 
[[`Project Page`](https://openreview.net/forum?id=MHNFjjm5nO)]
[[`Open Review`](https://openreview.net/forum?id=MHNFjjm5nO)]

</div>

---

<p align="center">
  <img src="assets/src/7_avits.gif" width="70%">
</p>

---

## Brief Introduction

<p align="center">
  <a href="assets/src/main.pdf" target="_blank">
    <img src="assets/src/main.png" width="80%">
  </a>
</p>

**Inference Time Scaling (ITS)** extends generation quality without retraining by leveraging pre-trained reward models at inference time. 

Rather than generating a single sample through the diffusion process, ITS generates a population of diverse candidates and uses reward signals (video quality, audio-video synchronization, etc.) to iteratively refine them. This approach enables:

- **BON (Best-of-N)**: Generate N samples and select the best one using reward ranking
- **EvoSearch**: Evolutionary refinement through multiple denoising stages with selection and mutation

Key advantages:
- ✅ No additional training required
- ✅ Works with any pre-trained generative model  
- ✅ Flexible reward combinations (VideoReward, JavisScore, CLAP, etc.)
- ✅ Significant quality improvements (up to +34% on sync metrics)

### 🎬 Supported Joint Audio-Video Generation Models

1. **[JavisDiT](https://github.com/JavisVerse/JavisDiT)** — Joint Audio-Video Diffusion Transformer
   - Synchronized audio-video generation with spatio-temporal priors

2. **LTX-2** ⭐ **— Most powerful audio-video generation model**
   - State-of-the-art quality and synchronization
   - Recommended for best results

3. **[MMDisCo](https://github.com/SonyResearch/MMDisCo)** — Cooperative Diffusion for Joint Audio-Video Generation (TBD)
   - Discriminator-guided multimodal generation
   - *Code will be released soon*

---

> For JavisDiT base model details, installation, and pre-training, refer to the [JavisDiT repository](https://github.com/JavisVerse/JavisDiT).

---

## Quick Start

### 1. Install JavisDiT

Follow the [JavisDiT installation guide](https://github.com/JavisVerse/JavisDiT) first.

Then clone this ITS repository:

```bash
git clone https://github.com/JavisDiT/JavisDiT-ITS.git
cd JavisDiT-ITS
```

<details>
<summary><b>Inference-only environment setup</b></summary>

If you only need inference (not training), install minimal dependencies:

```bash
# PyTorch with CUDA 12.1
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Core inference dependencies
pip install -r requirements/requirements-cu121.txt

# Optional: for faster generation
pip install flash-attn --no-build-isolation
```

</details>

### 2. Reward Server Environment Setup

The reward server requires additional dependencies for inference time scaling (BON or EvoSearch). Install on top of Step 1:

```bash
# Install evaluation dependencies
pip install -r requirements/requirements-eval.txt

# Reward model dependencies
pip install imagebind-huge  # or manually download imagebind_huge.pth
pip install einops ftfy

# Install VideoReward (for --reward_model VideoReward)
git clone https://github.com/KlingAIResearch/VideoAlign.git ./VideoAlign
pip install -e ./VideoAlign

# Install CLAP for audio alignment (optional, for --audio_model clap)
pip install transformers[audio]>=4.33.0
```

> **Note:** PyTorch (torch, torchvision, torchaudio) is already installed from Step 1. Do not reinstall.

**Option: Separate environment for reward server**

If you prefer to run the reward server in a separate GPU with isolated environment:

```bash
# Create separate conda env
conda create -n reward-server python=3.10
conda activate reward-server

# Install PyTorch (required for this env)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Install evaluation dependencies
pip install -r requirements/requirements-eval.txt
pip install imagebind-huge einops ftfy

# Install VideoReward
git clone https://github.com/KlingAIResearch/VideoAlign.git ./VideoAlign
pip install -e ./VideoAlign

# Optional: Install CLAP
pip install transformers[audio]>=4.33.0
```

Then run `bash vqa_server.sh [GPU_ID]` in this separate environment.

### 3. Download Pre-trained Weights

```bash
# ImageBind (required for JavisScore)
wget https://dl.fbaipublicfiles.com/imagebind/imagebind_huge.pth \
    -O ./checkpoints/imagebind_huge.pth
```

Model weights (JavisDiT-v0.1-jav-240p4s, prior) auto-download on first inference.

### 4. Test Installation

Verify everything works with a quick test:

```bash
python scripts/inference.py \
    configs/javisdit-v0-1/inference/sample_240p4s_standard.py \
    --prompt "a cat playing with a ball in a sunny garden" \
    --num-frames 2s --resolution 240p --aspect-ratio 9:16 \
    --save-dir samples/test_output \
    --verbose 2
```

This generates a 2-second video. If successful, output will be saved in `samples/test_output/`.

---

## Inference Time Scaling

### Step 1: Start the Reward Server

The reward server must run on a separate GPU:

```bash
bash vqa_server.sh [GPU_ID]
# Example: bash vqa_server.sh 0
```

This launches VideoReward + JavisScore verifiers on port 5001. Keep this running during generation.

**Server Configuration Options:**

Edit `vqa_server.sh` or pass arguments directly:

```bash
CUDA_VISIBLE_DEVICES=0 python reward_model/vqa_server.py \
    --gpu 0 \
    --addr 5001 \
    --reward_model [REWARD_MODEL] \
    --align_model [ALIGN_MODEL] \
    --audio_model [AUDIO_MODEL]
```

| Option | Values | Default |
|--------|--------|---------|
| `--reward_model` | `VideoReward`, `vqascore` | `VideoReward` |
| `--align_model` | `JavisScore`, `AVHScore`, `AVIB`, `All`, `None` | `JavisScore` |
| `--audio_model` | `clap`, `None` | `None` |

**Examples:**
```bash
# Default (VideoReward + JavisScore)
bash vqa_server.sh 0

# With audio alignment (CLAP)
CUDA_VISIBLE_DEVICES=0 python reward_model/vqa_server.py \
    --gpu 0 --addr 5001 \
    --reward_model VideoReward \
    --align_model JavisScore \
    --audio_model clap

# All alignment models
CUDA_VISIBLE_DEVICES=0 python reward_model/vqa_server.py \
    --gpu 0 --addr 5001 \
    --reward_model VideoReward \
    --align_model All
```

### Step 2: Run Inference

#### Option A: Standard (No ITS)

Generate without inference time scaling:

```bash
bash scripts/inference_standard.sh [GPU_ID] [NSHARD] [SHARD_ID]
# Single GPU: bash scripts/inference_standard.sh 0
# Multi-GPU: for i in {0..3}; do bash scripts/inference_standard.sh $i 4 $i & done; wait
```

**Config** (`sample_240p4s_standard.py`):
- No ITS - pure joint audio-video generation
- Single sample per prompt

#### Option B: BON (Best-of-N)

Generate 5 candidates, select best:

```bash
bash scripts/inference.sh [GPU_ID] [NSHARD] [SHARD_ID]
# Single GPU: bash scripts/inference.sh 0
# Multi-GPU: for i in {0..3}; do bash scripts/inference.sh $i 4 $i & done; wait
```

**Config** (`sample_240p4s.py`):
- `evolution_schedule=[51]` → evaluate only at end
- `population_size_schedule=[5, 5]` → 5 candidates
- `score_method="zscore_history"` → reward normalization

#### Option C: EvoSearch

Evolutionary refinement at steps 0 and 10 of denoising:

```bash
bash scripts/inference_evo.sh [GPU_ID] [NSHARD] [SHARD_ID]
# Single GPU: bash scripts/inference_evo.sh 0
# Multi-GPU: for i in {0..3}; do bash scripts/inference_evo.sh $i 4 $i & done; wait
```

**Config** (`sample_240p4s_evo.py`):
- `evolution_schedule=[0, 10]` → evolve at steps 0 and 10
- `population_size_schedule=[5, 5, 5]` → 3 generations
- `mutation_rate=0.2` → noise for diversity
- `score_method="adaptive"` → learnable reward weights

---

## Configuration

Both methods are configured in `configs/javisdit-v0-1/inference/sample_*.py`:

**Reward Models (Verifiers):**
```python
stage_verifiers=[["VideoReward", "JavisScore"]]  # Change verifiers
stage_weights=[[0.5, 0.5]]                        # Adjust combination
```

Available verifiers:
| Verifier | Purpose | Requires |
|----------|---------|----------|
| `VideoReward` | Video quality metric | VideoAlign |
| `JavisScore` | Audio-video synchronization | ImageBind |
| `VQA` | Generic video QA scoring | CLIP + T5 |
| `CLAP` | Audio-prompt alignment | CLAP (optional) |
| `AVHScore` | Audio-visual harmony | ImageBind |
| `AVIB` | Audio-visual interaction | ImageBind |

Example with multiple verifiers:
```python
stage_verifiers=[["VideoReward", "JavisScore", "CLAP"]]
stage_weights=[[0.4, 0.4, 0.2]]  # VideoReward 40%, JavisScore 40%, CLAP 20%
```

**Evolution:**
```python
evolution_schedule=[0, 10]          # When to evolve
population_size_schedule=[5, 5, 5]  # Population per stage
mutation_rate=0.2                   # Mutation strength
elite_size=2                        # Keep top-K
score_method="adaptive"             # Normalization method
```

---

## Evaluation

For detailed evaluation instructions and metrics:

- **JavisBench Evaluation**: https://github.com/JavisVerse/JavisDiT/blob/main/eval/javisbench/README.md
- **VideoReward Model**: https://github.com/KlingAIResearch/VideoAlign
- **VBench Metrics**: https://github.com/Vchitect/VBench

---

## Troubleshooting

| Issue | Solution |
|-------|----------|
| Connection refused (5001) | Ensure `vqa_server.sh` is running |
| Out of memory | Reduce `population_size_schedule` or resolution |
| Flash attention error | Pass `--flash-attn False --layernorm-kernel False` or edit config |

---

## Citation

```bibtex
@article{syncinference,
    title={Inference-Time Scaling for Joint Audio--Video Generation},
    author={Sync, Audio-Visual}
}
```

---

## Reference

- **JavisDiT**: https://github.com/JavisVerse/JavisDiT
- **JavisBench**: https://huggingface.co/datasets/JavisDiT/JavisBench
- **Paper**: https://arxiv.org/abs/2503.23377

For questions, open an issue or visit the [JavisDiT project page](https://javisdit.github.io/).
