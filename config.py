from dataclasses import dataclass

import torch


@dataclass
class Config:
    """
    The configuration for the FastLCM.
    """
    ####################################################################
    # Model configuration
    ####################################################################
    # LCM model
    model_id_or_path: str = "discus0434/lcm_anything_v5"
    # TinyVAE model
    vae_id: str = "madebyollin/taesd"
    # Device to use
    device: torch.device = torch.device("cuda")
    # Data type
    dtype: torch.dtype = torch.float16
    # LCMScheduler parameters
    config_path: str = "/app/assets/config.json"
    # Whether to compile the model
    compile: bool = True
    ####################################################################
    # Inference configuration
    ####################################################################
    # Image to transfer
    image_path: str = "/app/assets/sample.png"
    # Generation resolution
    resolution: int = 512
    # Prompt
    prompt: str = "1girl, (masterpiece, best quality:1.2)"
    # Number of inference steps
    num_inference_steps: int = 1
    # Strength
    strength: float = 0.2
    # Guidance scale
    guidance_scale: float = 1
    # Original inference steps if not using LCM
    original_inference_steps: int = 50
    # Whether to save the image. If True, the image will be saved to
    # assets/output/output_{i}.png, where i is the iteration number.
    # False to disable, which makes the script run faster.
    save_image: bool = False
