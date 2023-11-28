import json
import time

import numpy as np
import torch
import torch._dynamo
from diffusers import AutoencoderTiny, StableDiffusionImg2ImgPipeline
from PIL import Image
from tqdm import tqdm

from config import Config
from lib import LCMScheduler

torch._dynamo.config.suppress_errors = True
torch.backends.cuda.matmul.allow_tf32 = True


class FastLCM:
    def __init__(self, config: Config) -> None:
        """
        Constructs a FastLCM object.

        Parameters
        ----------
        config : Config
            The configuration object.
        """
        self.config = config
        self.lcm_pipeline = self._init_lcm_pipeline()
        self._prompt_embedding = self._precompute_prompt_embedding()
        self._image = self.lcm_pipeline.image_processor.preprocess(
            Image.open(self.config.image_path)
            .convert("RGB")
            .resize((self.config.resolution,) * 2, Image.Resampling.LANCZOS)
        )
        self._warm_up()

    def run(self):
        """
        Runs 100 times the faster LCM pipeline, and calculates the
        average time and FPS.
        """
        times = []
        for i in tqdm(range(100)):
            start_time = time.time()
            image = self.lcm_pipeline(
                image=self._image,
                prompt_embeds=self._prompt_embedding,
                num_inference_steps=int(
                    self.config.num_inference_steps / self.config.strength
                )
                + 1,
                strength=self.config.strength,
                guidance_scale=self.config.guidance_scale,
                original_inference_steps=self.config.original_inference_steps,
                output_type="pil",
            ).images[0]
            times.append(time.time() - start_time)

            print("num_inference_steps", self.lcm_pipeline._num_timesteps)
            image.save(f"assets/output/output_{i}.png")

        print(f"Average time: {np.mean(times)}")
        print(f"FPS: {1 / np.mean(times)}")

    def _init_lcm_pipeline(self) -> StableDiffusionImg2ImgPipeline:
        """
        Initializes the LCM pipeline.

        Returns
        -------
        StableDiffusionImg2ImgPipeline
            The LCM pipeline.
        """
        # Initialize the LCM pipeline
        lcm_pipeline = StableDiffusionImg2ImgPipeline.from_pretrained(
            self.config.model_id_or_path,
            safety_checker=None,
            feature_extractor=None,
        )

        # Reload the scheduler with custom config
        lcm_pipeline.scheduler = LCMScheduler.from_config(
            json.loads(open(self.config.config_path).read())
        )

        # Load the tiny VAE
        lcm_pipeline.vae = AutoencoderTiny.from_pretrained(self.config.vae_id)

        # Move the pipeline to the GPU
        lcm_pipeline.to(device=self.config.device, dtype=self.config.dtype)

        # Set the unet to channels last
        lcm_pipeline.unet = lcm_pipeline.unet.to(memory_format=torch.channels_last)

        # Disable the progress bar
        lcm_pipeline.set_progress_bar_config(disable=True)

        # Compile the pipeline
        if self.config.compile:
            lcm_pipeline.unet = torch.compile(
                lcm_pipeline.unet, mode="reduce-overhead", fullgraph=True
            )
            lcm_pipeline.vae = torch.compile(
                lcm_pipeline.vae, mode="reduce-overhead", fullgraph=True
            )

        return lcm_pipeline

    def _warm_up(self):
        """
        Warms up the LCM pipeline.
        """
        for _ in range(3):
            self.lcm_pipeline(
                image=self._image,
                prompt_embeds=self._prompt_embedding,
                num_inference_steps=int(
                    self.config.num_inference_steps / self.config.strength
                )
                + 1,
                strength=self.config.strength,
                guidance_scale=self.config.guidance_scale,
                original_inference_steps=self.config.original_inference_steps,
            )

    def _precompute_prompt_embedding(self) -> torch.FloatTensor:
        """
        Precomputes the prompt embedding to speed up the inference.

        Returns
        -------
        torch.FloatTensor
            The prompt embedding.
        """
        prompt_embedding, _ = self.lcm_pipeline.encode_prompt(
            device=self.config.device,
            prompt=self.config.prompt,
            do_classifier_free_guidance=False,
            num_images_per_prompt=1,
            clip_skip=2,
        )
        return prompt_embedding


if __name__ == "__main__":
    config = Config()

    lcm = FastLCM(config)
    lcm.run()
