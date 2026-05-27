resolution = "240p"
aspect_ratio = "9:16"
num_frames = "4s"
fps = 24
audio_fps = 16000
frame_interval = 1
save_fps = 24

save_dir = "./samples/samples_240/VGGSound/test"
seed = 42
batch_size = 1 #for prompt
multi_resolution = "OpenSora"
dtype = "bf16"
loop = 1  # loop for video extension
condition_frame_length = 5  # used for video extension conditioning
align = 5  # TODO: unknown mechanism, maybe for conditional frame alignment?
verbose = 2

spatial_token_num = 32
temporal_token_num = 32
st_prior_channel = 128

# === Inference Time Scaling (ITS) Configuration for EvoSearch ===
# EvoSearch: Evolutionary search that refines samples through multiple generations
# evolution_schedule=[0, 10] means evolution at step 0 and step 10 of 30 total denoising steps

evosearch = dict(
  # When to evolve: [0, 10] = evolve at denoising steps 0 and 10 (out of 30 total)
  # Step 0: Initial generation with diversity
  # Step 10: Refinement after early denoising
  evolution_schedule=[0, 10],

  # Population per generation: 5 candidates in each of 3 generations
  population_size_schedule=[5, 5, 5],

  # Process samples sequentially to save memory
  sequential_processing=True,

  # Don't use online guidance during generation
  guidance_reward="VideoReward",

  # Reward models for evaluating candidates
  stage_verifiers=[
    ["VideoReward", "JavisScore"],  # Gen 1: evaluate quality and sync
    ["VideoReward", "JavisScore"],  # Gen 2: evaluate quality and sync
    ["VideoReward", "JavisScore"],  # Gen 3: final evaluation
  ],

  # Combine verifiers with equal weight
  stage_weights=[
    [0.5, 0.5],
    [0.5, 0.5],
    [0.5, 0.5],
  ],

  # Keep top 2 performers for next generation
  elite_size=2,

  iterations=1,

  # Mutation: add Gaussian noise to latents of selected candidates (σ = 0.2)
  mutation_rate=0.2,

  vqa_weight=1.0,
  align_weight=0.0,

  # Tournament selection: higher ratio = stronger selection pressure
  tournament_ratio=0.5,

  # Adaptive: learn optimal weight combining multiple verifiers (ARW loss)
  # Alternative: zscore_history, zscore, rank, weighted, minmax
  score_method="adaptive",
  javis_boost=0.5,

  # Statistics for z-score normalization
  VQ_mean=-0.4779,
  VQ_std=0.8785,
  JS_mean=0.1613,
  JS_std=0.1033,
)

model = dict(
    type="VASTDiT3-XL/2",
    weight_init_from=[],
    from_pretrained="./checkpoints/JavisDiT-v0.1-jav-240p4s",
    qk_norm=True,
    enable_flash_attn=True,
    enable_layernorm_kernel=False,
    # video-audio joint generation
    freeze_y_embedder=True,
    freeze_video_branch=True,
    freeze_audio_branch=True,
    train_st_prior_attn=True,
    train_va_cross_attn=True,
    spatial_prior_len=spatial_token_num,
    temporal_prior_len=temporal_token_num,
    st_prior_channel=st_prior_channel,
    audio_patch_size=(4, 1)
)
vae = dict(
    type="OpenSoraVAE_V1_2",
    from_pretrained="./checkpoints/OpenSora-VAE-v1.2",
    micro_frame_size=17,
    micro_batch_size=4,
)
audio_vae = dict(
    type="AudioLDM2",
    from_pretrained="./checkpoints/audioldm2",
)
text_encoder = dict(
    type="t5",
    from_pretrained="./checkpoints/t5-v1_1-xxl",
    model_max_length=300,
)
prior_encoder = dict(
    type="STIBPrior",
    imagebind_ckpt_path="./checkpoints",
    from_pretrained="./checkpoints/JavisDiT-v0.1-prior",
    spatial_token_num=spatial_token_num,
    temporal_token_num=temporal_token_num,
    out_dim=st_prior_channel,
    hidden_size=512,
    apply_sampling=True,
    encode_va=False,
    qk_norm=True,
    enable_flash_attn=True,
    enable_layernorm_kernel=False,
)
scheduler = dict(
    type="rflow",
    use_timestep_transform=True,
    num_sampling_steps=30,
    cfg_scale=7.0,
)

aes = 6.5    # aesthetic score
flow = None  # motion score


correct = dict(
  vqa_server_addr=5001, # check here!!! (default: 5000)
  corrector="particle",
  reward_weight=0.5,
  reward_score="VideoReward",
  vqa_model="clip-flant5-xxl",
  vqa_batch_size=1,
  vqa_device=1,
  align_weight=0.5,
)
