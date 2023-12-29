import json
import numpy as np
from PIL import Image
from collections import defaultdict

def load_json(filename):
    with open(filename, 'r') as file:
        return json.load(file)

def get_fp_value(image_path):
    # Extract the fp value from the image path
    parts = image_path.split('_')
    for part in parts:
        if part.startswith('fp'):
            return part
    return None

def calculate_mean_std(image_paths):
    means, stds = [], []
    for image_path in image_paths:
        image = Image.open(image_path)
        image_array = np.array(image)
        means.append(np.mean(image_array))
        stds.append(np.std(image_array))
    return np.mean(means), np.std(stds)

def process_integral_images(data):
    grouped_images = defaultdict(list)
    for entry in data:
        for image_path in entry['integral_images']:
            fp_value = get_fp_value(image_path)
            if fp_value:
                grouped_images[fp_value].append(image_path)

    results = {}
    for fp_value, image_paths in grouped_images.items():
        mean, std = calculate_mean_std(image_paths)
        results[fp_value] = {'mean': mean, 'std': std}
    return results

# Replace 'your_file.json' with the path to your JSON file
data = load_json('../../train_paths.json')
results = process_integral_images(data)

# Display the results
for fp_value, stats in results.items():
    print(f"fp-{fp_value}: Mean: {stats['mean']}, Std: {stats['std']}")
