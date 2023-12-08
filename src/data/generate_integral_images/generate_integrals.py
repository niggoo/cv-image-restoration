import json
import cv2
from AOS_integrator import create_integral_image
import os
from pathlib import Path

# setup argparse for command line arguments "batch" and "image_id" both string
import argparse

parser = argparse.ArgumentParser()
parser.add_argument(
    "filename", help="filename and path for generated image"
)  # 2019-11-26
parser.add_argument("focal_plane", help="focal plane")  # 0

args, image_list = parser.parse_known_args()  # the rest of the arguments are the images
assert len(image_list) == 11, "no images given"
filename = args.filename
focal_plane = args.focal_plane

# make sure focal_plane is a float between 0 and -3
focal_plane = float(focal_plane)
assert 0.0 >= focal_plane >= -3.0, "focal_plane must be between 0 and -3"

integral_image = create_integral_image(image_list, focal_plane)
# save the integral image into p
cv2.imwrite(filename, integral_image)
