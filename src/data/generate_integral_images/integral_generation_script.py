import json
import os
import multiprocessing
import time
from pathlib import Path
import numpy as np

# import tqdm if installed
try:
    from tqdm import tqdm
except ModuleNotFoundError:
    tqdm = lambda x: x
    print("for a progress bar, install tqdm")


def generate_integrals(args):
    filename, focal_plane, images = args
    os.system(
        f"python generate_integrals.py {filename} {focal_plane} {' '.join(images)}"
    )


if __name__ == "__main__":
    # create integral images folder
    integral_images_path = "../../data/integral_images"
    os.makedirs(integral_images_path, exist_ok=True)

    # load the json file
    with open("data.json") as json_file:
        data = json.load(json_file)

    # create a list of focal planes
    focal_planes = [0.0, -0.5, -1.0, -1.5]
    # focal_planes = list(np.round(np.arange(0.0, -3.1, -0.1), 1))

    # create a list of arguments for generate_integrals (does not add images that already exist)
    args_list = []
    for batch in data:
        for image_id in data[batch]:
            for focal_plane in focal_planes:
                images = data[batch][image_id]["raw_images"]
                batch_folder = Path(batch).parts[
                    -2:
                ]  # e.g. batch_20230912_part1-006\\Part1
                p = os.path.join(integral_images_path, *batch_folder, image_id)
                filename = f"{p}/{image_id}_integral_fp{focal_plane}.png"
                # create folders after batch name and id
                if not os.path.exists(p):
                    os.makedirs(p)
                if not os.path.exists(filename):
                    args_list.append((filename, focal_plane, images))

    start_time = time.time()
    print("Started: ", time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(start_time)))

    # use multiprocessing to run generate_integrals multiple times
    pool = multiprocessing.Pool(processes=50)
    try:
        # apply generate_integrals to each set of arguments
        for _ in tqdm(
            pool.imap_unordered(generate_integrals, args_list), total=len(args_list)
        ):
            pass
    except KeyboardInterrupt:
        print("Caught KeyboardInterrupt, terminating workers")
        pool.terminate()
        pool.join()
    else:
        pool.close()
        pool.join()

    # print the time taken
    end_time = time.time()
    print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(end_time)))
    elapsed_time = end_time - start_time
    print(f"Time taken: {elapsed_time:.2f} seconds or {elapsed_time / 60:.2f} minutes")
