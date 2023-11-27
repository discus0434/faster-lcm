from diffusers import StableDiffusionImg2ImgPipeline


def fuse_model():
    """
    Fuse the Stable Diffusion model with the LCM LoRA,
    and save it to the models folder.
    """
    lcm_pipeline = StableDiffusionImg2ImgPipeline.from_pretrained(
        "stablediffusionapi/anything-v5",
        safety_checker=None,
    )
    lcm_pipeline.load_lora_weights("latent-consistency/lcm-lora-sdv1-5")
    lcm_pipeline.fuse_lora(lora_scale=1.5)

    lcm_pipeline.save_pretrained("models/lcm_anything_v5")
