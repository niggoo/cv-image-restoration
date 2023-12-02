import os
import glob
from PIL import Image
from transformers import AutoImageProcessor, AutoModel
import torch
from safetensors.torch import save_file
from multiprocessing import Pool
import tqdm
import argparse


def process_image(image_path, processor, model, size):
    # Open and process the image
    image = Image.open(image_path)
    inputs = processor(images=image, return_tensors="pt")
    inputs.to(device="cuda")

    outputs = model(**inputs)
    last_hidden_states = outputs.last_hidden_state.to(torch.float16)

    # Construct output path including the model size
    base_path = image_path.replace("../data/proc", f"../data/embeddings/full/{size}")
    os.makedirs(os.path.dirname(base_path), exist_ok=True)
    output_path = os.path.splitext(base_path)[0] + ".safetensors"

    # Save the output
    save_file({"last_hidden_states": last_hidden_states}, output_path)


def process_images_on_gpu(gpu_id, image_paths, size):
    # Set the current GPU
    torch.cuda.set_device(gpu_id)

    # Construct the model name using the size parameter
    model_name = f"facebook/dinov2-{size}"

    # Initialize the processor and model
    processor = AutoImageProcessor.from_pretrained(model_name)
    processor.crop_size = {"height": 512, "width": 512}
    processor.size["shortest_edge"] = 512
    model = AutoModel.from_pretrained(model_name).to(f"cuda:{gpu_id}")
    model.to(device=f"cuda:{gpu_id}")

    # Process each image
    for image_path in tqdm.tqdm(image_paths):
        process_image(image_path, processor, model, size)


def main(size):
    # Find all image files
    image_paths = glob.glob("../data/proc/**/*.png", recursive=True)

    # Divide the workload among 4 GPUs
    gpu_workloads = [[] for _ in range(4)]
    for i, image_path in enumerate(image_paths):
        gpu_workloads[i % 4].append(image_path)

    # Use multiprocessing to process images on 4 GPUs
    with Pool(4) as p:
        args = [(gpu_id, gpu_workloads[gpu_id], size) for gpu_id in range(4)]
        p.starmap(process_images_on_gpu, args)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Process images with a specified model size."
    )
    parser.add_argument(
        "size",
        type=str,
        choices=["small", "base", "large", "giant"],
        help="Size of the model to use (e.g., base, large)",
    )
    args = parser.parse_args()

    main(args.size)
