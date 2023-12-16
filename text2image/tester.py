from diffusers import StableDiffusionPipeline
import torch
import argparse

pipeline = StableDiffusionPipeline.from_pretrained("results/", use_safetensors=True).to("cuda")
parser = argparse.ArgumentParser()
parser.add_argument("--prompt", type=str, help="Enter the prompt")
args = parser.parse_args()

prompt = args.prompt
image = pipeline(prompt=prompt,num_inference_steps=400).images[0]
image.save(prompt+".png")