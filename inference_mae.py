# gen2seg official inference pipeline code for Stable Diffusion model
# 
# Please see our project website at https://reachomk.github.io/gen2seg
#
# Additionally, if you use our code please cite our paper, along with the two works above. 


import os
import time
import torch
from gen2seg_mae_pipeline import gen2segMAEInstancePipeline  # Custom pipeline for MAE
from transformers import AutoImageProcessor
from PIL import Image
import numpy as np


# Example usage: Update these paths as needed.
image_path = "/nfs_share3/om/diffusion-e2e-ft/plane1.jpg"       # Path to the input image, change as needed.
output_path = "seg_mae.png"     # Path to save the output image.
device = "cuda:0"  # Change to "cpu" if no GPU is available.

print(f"Loading MAE pipeline on {device} for single image inference...")

# Load the image processor (using a pretrained processor from facebook/vit-mae-huge).
image_processor = AutoImageProcessor.from_pretrained("facebook/vit-mae-huge")

# Instantiate the pipeline and move it to the desired device.
pipe = gen2segMAEInstancePipeline(model="reachomk/gen2seg-mae-h", image_processor=image_processor).to(device)

# Load the image, storing the original size, then resize for inference.
orig_image = Image.open(image_path).convert("RGB")
orig_size = orig_image.size  # (width, height)
image = orig_image.resize((224, 224))

# Run inference.
start_time = time.time()
with torch.no_grad():
    pipe_output = pipe([image])
end_time = time.time()
print(f"Inference completed in {end_time - start_time:.2f} seconds.")
prediction = pipe_output.prediction[0]

# Convert the prediction to an image.
seg = np.array(prediction.squeeze()).astype(np.uint8)
seg_img = Image.fromarray(seg)

# Resize the segmentation output back to the original image size.
seg_img = seg_img.resize(orig_size, Image.LANCZOS)

# Save the output image.
seg_img.save(output_path)
print(f"Saved output image to {output_path}")
