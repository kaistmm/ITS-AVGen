## <div align="center">[TMLR] Inference Time Scaling for Joint Audio-Video Generation</div>

<div align="center">

**[Jaemin Jung](https://jung-jaemin.github.io/)**<sup>1</sup>, [Kyeongha Rho](https://kyeongharho.github.io/)<sup>1</sup>, [Inkyu Shin](https://dlsrbgg33.github.io/)<sup>2</sup>, [Joon Son Chung](https://mm.kaist.ac.kr/joon/)<sup>1</sup>

<sup>1</sup> KAIST, <sup>2</sup> Luma AI

[![Paper](https://img.shields.io/badge/📰-Paper-1f72be?style=flat)](https://openreview.net/forum?id=MHNFjjm5nO)
[![Project](https://img.shields.io/badge/🚀-Project%20Page-50c878?style=flat)](https://openreview.net/forum?id=MHNFjjm5nO)
[![Code](https://img.shields.io/badge/💻-Code-ff6b6b?style=flat)](https://openreview.net/forum?id=MHNFjjm5nO)

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

1. **[JavisDiT-ITS](https://github.com/kaistmm/ITS-AVGen)** ⭐ **— Joint Audio-Video Diffusion Transformer**
   - Synchronized audio-video generation with spatio-temporal priors

2. **[LTX2-ITS](https://github.com/kaistmm/ITS-AVGen-LTX2)** — Most powerful audio-video generation model
   - State-of-the-art quality and synchronization
   - Recommended for best results
   - Production-ready outputs with multiple resolution modes
   - **Currently implemented:** BON (Best-of-N)
   - **Coming soon:** EvoSearch (Evolutionary Search)

3. **[MMDisCo-ITS](https://github.com/SonyResearch/MMDisCo)** — Cooperative Diffusion for Joint Audio-Video Generation (TBD)
   - Discriminator-guided multimodal generation
   - *Code will be released soon*

---

> For JavisDiT base model details, installation, and pre-training, refer to the [JavisDiT repository](https://github.com/JavisVerse/JavisDiT).

---

## Installation

### Step 1: Setup Environment

```bash
git clone https://github.com/JavisDiT/JavisDiT-ITS.git
cd JavisDiT-ITS

# Create conda environment
conda create -n javisdit_its python=3.10
conda activate javisdit_its

# Install core dependencies
conda install -c conda-forge ffmpeg cuda-toolkit=12.1 -y
pip install -r requirements/requirements-cu121.txt
pip install -r requirements/requirements.txt

# Important: setuptools version for pkg_resources compatibility
pip install "setuptools<81" --force-reinstall

# Install JavisDiT-ITS
pip install -v -e .
```

### Step 2: Setup Reward Server

We recommend creating a separate environment for reward models to use across multiple generation models (JavisDiT, LTX2, etc.). However, you can also work within a single environment if preferred:

```bash
# Clone VideoAlign and setup environment
git clone https://github.com/KwaiVGI/VideoAlign
cd VideoAlign
conda env create -f environment.yaml
conda activate VideoReward
pip install flash-attn==2.5.8 --no-build-isolation
cd ..

# Install additional dependencies
pip install "setuptools<81" --force-reinstall
pip install git+https://github.com/facebookresearch/ImageBind.git

# Download model checkpoints
mkdir -p checkpoints
cd checkpoints
git clone https://huggingface.co/KwaiVGI/VideoReward
wget https://dl.fbaipublicfiles.com/imagebind/imagebind_huge.pth -O imagebind_huge.pth
cd ..
```

### Step 3: Test Installation

**Test 1: Standard Inference (No Reward Server)**

```bash
conda activate javisdit_its

python scripts/inference.py configs/javisdit-v0-1/inference/sample_240p4s_standard.py \
    --prompt "a cat playing with a ball in a sunny garden" \
    --num-frames 2s --resolution 240p --aspect-ratio 9:16 \
    --save-dir samples/test_output --verbose 2
```

Output: Video saved in `samples/test_output/`

**Test 2: With Reward Server (Optional)**

If you installed the reward server (Step 2):

```bash
# Terminal 1: Start reward server
conda activate VideoReward
bash vqa_server.sh 0
# Should see: "Server started... Listening for requests."

# Terminal 2: Run inference with rewards
conda activate javisdit_its
CUDA_VISIBLE_DEVICES=1 python scripts/inference.py \
    configs/javisdit-v0-1/inference/sample_240p4s.py \
    --prompt "a cat playing with a ball in a sunny garden" \
    --num-frames 2s --resolution 240p --aspect-ratio 9:16 \
    --save-dir samples/test_output --verbose 2
```

Expected: vqa_server terminal shows reward computation logs


---

## Usage

### Standard Inference (No ITS)

Generate without reward-based selection:

```bash
conda activate javisdit_its

python scripts/inference.py \
    configs/javisdit-v0-1/inference/sample_240p4s_standard.py \
    --prompt "your prompt here" \
    --num-frames 2s --resolution 240p --aspect-ratio 9:16 \
    --save-dir samples/output
```

### With Reward Models (BON or EvoSearch)

Requires reward server running:

```bash
# Terminal 1: Start reward server
conda activate VideoReward
CUDA_VISIBLE_DEVICES=0 bash vqa_server.sh 0

# Terminal 2: Run inference with rewards
conda activate javisdit_its
CUDA_VISIBLE_DEVICES=1 python scripts/inference.py \
    configs/javisdit-v0-1/inference/sample_240p4s.py \
    --prompt "your prompt here" \
    --num-frames 2s --resolution 240p --aspect-ratio 9:16 \
    --save-dir samples/output
```

**Available configs:**
- `sample_240p4s_standard.py` - Standard generation (no ITS)
- `sample_240p4s.py` - BON (Best-of-N) with 2 candidates
- `sample_240p4s_evo.py` - EvoSearch with evolutionary refinement

Edit the config files to adjust:
- Population size and evolution schedule
- Reward models (VideoReward, JavisScore, CLAP)
- Adaptive reward weighting settings

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
evolution_schedule=[0, 10]          # When to evolve (denoising steps)
population_size_schedule=[5, 5, 5]  # Population per generation
mutation_rate=0.2                   # Mutation strength (Gaussian noise σ)
elite_size=2                        # Keep top-K performers
score_method="adaptive"             # Score aggregation method
sequential_processing=False         # False: batch processing (faster)
                                    # True: sequential (memory efficient)
```

**Processing Mode:**
- `sequential_processing=False` (default): Batch process all samples in one forward pass. **Faster** but uses more VRAM.
- `sequential_processing=True`: Process samples one-by-one sequentially. **Slower** but memory-efficient for large populations.

**Score Aggregation Methods:**

| Method | Description |
|--------|-------------|
| `"zscore"` | Z-score normalization across all samples |
| `"rank"` | Rank-based scoring (ordinal ranking) |
| `"weighted"` | Weighted combination of verifier scores |
| `"minmax"` | Min-max normalization (0-1 range) |
| `"adaptive"` | Learnable weights via Adaptive Reward Weighting (ARW) |

---

## Evaluation

For detailed evaluation instructions and metrics:

- **JavisBench Evaluation**: https://github.com/JavisVerse/JavisDiT/blob/main/eval/javisbench/README.md
- **VideoReward Model**: https://github.com/KlingAIResearch/VideoAlign
- **VBench Metrics**: https://github.com/Vchitect/VBench


## Citation

```bibtex
@article{its2026,
    title={Inference-Time Scaling for Joint Audio--Video Generation},
    author={Jung, Jaemin and Rho, Kyeongha and Shin, Inkyu and Chung, Joon Son},
    journal={Transactions on Machine Learning Research},
    year={2026}
}
```

---

## Reference

- **JavisDiT**: https://github.com/JavisVerse/JavisDiT
- **JavisBench**: https://huggingface.co/datasets/JavisDiT/JavisBench
