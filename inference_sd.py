# gen2seg official inference pipeline code for Stable Diffusion model
#
# Please see our project website at https://reachomk.github.io/gen2seg
#
# Additionally, if you use our code please cite our paper. 


import torch
from gen2seg_sd_pipeline import gen2segSDPipeline  # Import your custom pipeline
from PIL import Image
import numpy as np
import time

# Load the image
image_path =  '/nfs_share3/om/diffusion-e2e-ft/plane1.jpg' # change this path
image = Image.open(image_path).convert("RGB")
orig_res = image.size
output_path = "seg.png" 

pipe = gen2segSDPipeline.from_pretrained(
    "reachomk/gen2seg-sd",
    use_safetensors=True,         # Use safetensors if available
).to("cuda")  # Ensure the pipeline is moved to CUDA

# Load the pipeline and generate the segmentation map
with torch.no_grad():

    start_time = time.time()
    # Generate segmentation map
    seg = pipe(image).prediction.squeeze()
    end_time = time.time()
    print(f"Inference completed in {end_time - start_time:.2f} seconds.")

seg = np.array(seg).astype(np.uint8)
Image.fromarray(seg).resize(orig_res, Image.LANCZOS).save(output_path)
print(f"Saved output image to {output_path}")