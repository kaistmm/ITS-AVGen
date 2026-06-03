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

# === Inference Time Scaling (ITS) Configuration for BON ===
# BON (Best-of-N): Generate N samples and select the best one
# evolution_schedule=[51] means evaluation happens at the last step (after full generation)

evosearch = dict(
  evolution_schedule=[51],

  # How many candidates to generate: [2, 2] = 2 generations with 2 samples each
  population_size_schedule=[2, 2],

  # Process samples in batch (False) for faster inference
  # False = batch processing (faster, more memory) - 1 forward pass per denoising step
  # True = sequential processing (slower, less memory) - multiple forward passes per step
  sequential_processing=False,

  # Don't use online guidance during generation
  guidance_reward="VideoReward",

  # Which reward models to use for evaluation
  stage_verifiers=[
    ["VideoReward", "JavisScore"],  # Gen 0: Video quality + Audio-video sync
  ],

  # Combine multiple verifiers with equal weight
  stage_weights=[
    [0.5, 0.5],  # Gen 0
  ],

  iterations=1,
  mutation_rate=0.2,
  elite_size=2,
  vqa_weight=1.0,
  align_weight=0.0,
  tournament_ratio=0.5,
  javis_boost=0.5,

  # Score aggregation method: zscore, rank, weighted, minmax, or adaptive
  score_method="weighted",

  # Statistics for z-score normalization (empirically measured on validation set)
  VQ_mean=-0.4779,
  VQ_std=0.8785,
  JS_mean=0.1613,
  JS_std=0.1033,
)

adaptive_convergence = dict(
  enabled=True,
  online_prefix="arw_online",
  online_step_prefix="arw_online_step",
  checkpoint_name="adaptive_reward_weighter.pt",
)

adaptive_optimizer = "Adam"  # Adam, AdamW, RMSprop, SGD, Adagrad, LBFGS

model = dict(
    type="VASTDiT3-XL/2",
    weight_init_from=[],
    from_pretrained="./checkpoints/JavisDiT-v0.1-jav-240p4s",
    qk_norm=True,
    enable_flash_attn=False,  # Disable for compatibility
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
