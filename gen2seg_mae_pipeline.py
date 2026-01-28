# gen2seg official inference pipeline code for Stable Diffusion model
# 
# Please see our project website at https://reachomk.github.io/gen2seg
#
# Additionally, if you use our code please cite our paper. 

from dataclasses import dataclass
from typing import Union, List, Optional

import torch
import numpy as np
from PIL import Image
from einops import rearrange

from diffusers import DiffusionPipeline
from diffusers.utils import BaseOutput, logging
from transformers import AutoImageProcessor, ViTMAEForPreTraining

logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


@dataclass
class gen2segMAEInstanceOutput(BaseOutput):
    """
    Output class for the ViTMAE Instance Segmentation Pipeline.

    Args:
        prediction (`np.ndarray` or `torch.Tensor`):
            Predicted instance segmentation maps. The output has shape 
            `(batch_size, 3, height, width)` with pixel values scaled to [0, 255].
    """
    prediction: Union[np.ndarray, torch.Tensor]


class gen2segMAEInstancePipeline(DiffusionPipeline):
    r"""
    Pipeline for Instance Segmentation using a fine-tuned ViTMAEForPreTraining model.

    This pipeline takes one or more input images and returns an instance segmentation
    prediction for each image. The model is assumed to have been fine-tuned using an instance
    segmentation loss, and the reconstruction is performed by rearranging the model’s
    patch logits into an image.

    Args:
        model (`ViTMAEForPreTraining`):
            The fine-tuned ViTMAE model.
        image_processor (`AutoImageProcessor`):
            The image processor responsible for preprocessing input images.
    """
    def __init__(self, model, image_processor):
        super().__init__()
        if isinstance(model, str):
            model = ViTMAEForPreTraining.from_pretrained(model)
        self.register_modules(model=model, image_processor=image_processor)
        self.model = model
        self.image_processor = image_processor

    def check_inputs(
        self,
        image: Union[Image.Image, np.ndarray, torch.Tensor, List[Union[Image.Image, np.ndarray, torch.Tensor]]]
    ) -> List:
        if not isinstance(image, list):
            image = [image]
        # Additional input validations can be added here if desired.
        return image

    @torch.no_grad()
    def __call__(
        self,
        image: Union[Image.Image, np.ndarray, torch.Tensor, List[Union[Image.Image, np.ndarray, torch.Tensor]]],
        output_type: str = "np",
        **kwargs
    ) -> gen2segMAEInstanceOutput:
        r"""
        The call method of the pipeline.

        Args:
            image (`PIL.Image.Image`, `np.ndarray`, `torch.Tensor`, or a list of these):
                The input image(s) for instance segmentation. For arrays/tensors, expected values are in [0, 1].
            output_type (`str`, optional, defaults to `"np"`):
                The format of the output prediction. Choose `"np"` for a NumPy array or `"pt"` for a PyTorch tensor.
            **kwargs:
                Additional keyword arguments passed to the image processor.

        Returns:
            [`gen2segMAEInstanceOutput`]:
                An output object containing the predicted instance segmentation maps.
        """
        # 1. Check and prepare input images.
        images = self.check_inputs(image)
        inputs = self.image_processor(images=images, return_tensors="pt", **kwargs)
        pixel_values = inputs["pixel_values"].to(self.device)

        # 2. Forward pass through the model.
        outputs = self.model(pixel_values=pixel_values)
        logits = outputs.logits  # Expected shape: (B, num_patches, patch_dim)

        # 3. Retrieve patch size and image size from the model configuration.
        patch_size = self.model.config.patch_size  # e.g., 16
        image_size = self.model.config.image_size    # e.g., 224
        grid_size = image_size // patch_size

        # 4. Rearrange logits into the reconstructed image.
        #    The logits are reshaped from (B, num_patches, patch_dim) to (B, 3, H, W).
        reconstructed = rearrange(
            logits,
            "b (h w) (p1 p2 c) -> b c (h p1) (w p2)",
            h=grid_size,
            p1=patch_size,
            p2=patch_size,
            c=3,
        )

        # 5. Post-process the reconstructed output.
        #    For each sample, shift and scale the prediction to [0, 255].
        predictions = []
        for i in range(reconstructed.shape[0]):
            sample = reconstructed[i]
            min_val = torch.abs(sample.min())
            max_val = torch.abs(sample.max())
            sample = (sample + min_val) / (max_val + min_val + 1e-5)
            # sometimes the image is very dark so we perform gamma correction to "brighten" it
            # in practice we can set this value to whatever we want or disable it entirely. 
            sample = sample**0.7
            sample = sample * 255.0
            predictions.append(sample)
        prediction_tensor = torch.stack(predictions, dim=0).permute(0, 2, 3, 1)

        # 6. Format the output.
        if output_type == "np":
            prediction = prediction_tensor.cpu().numpy()
        else:
            prediction = prediction_tensor
        return gen2segMAEInstanceOutput(prediction=prediction)
