from dataclasses import dataclass

import torch


@dataclass
class Config:
    """
    The configuration for the FastLCM.
    """
    # LCM model
    model_id_or_path: str = "discus0434/lcm_anything_v5"
    # TinyVAE model
    vae_id: str = "madebyollin/taesd"
    # Device to use
    device: torch.device = torch.device("cuda")
    # Data type
    dtype: torch.dtype = torch.float16
    # Image to transfer
    image_path: str = "/app/assets/sample.png"
    # LCMScheduler parameters
    config_path: str = "/app/assets/config.json"
    # Generation resolution
    resolution: int = 512
    # Prompt
    prompt: str = "1girl, (masterpiece, best quality:1.2)"
    # Whether to compile the model
    compile: bool = True
