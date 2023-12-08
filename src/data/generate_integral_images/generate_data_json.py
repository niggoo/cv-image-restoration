from pathlib import Path
import json

from typing import List, Tuple


def get_all_files(path: str) -> List[Path]:
    """
    Returns a list of all png and txt files in the given path
    """
    data_files = list(Path(path).rglob("*.png"))
    data_files += list(Path(path).rglob("*.txt"))
    return data_files


def dict_from_data_paths(data_files: List[Path]) -> dict:
    """
    Returns a dict with the following structure:
    {
        "batch_folder": {
            "image_id": {
                "GT": "path/to/image_id_GT_pose_0_thermal.png",
                "raw_images": ["path/to/image_id_pose_0_thermal.png", ..., "path/to/image_id_pose_10_thermal.png"],
                "parameters": "path/to/image_id_parameters.txt"
            }
        }
    }
    data_files: list of pathlib.Path objects

    """
    files = {}
    for file in data_files:
        parent = file.parent.__str__()
        file_name = file.name
        image_id = "_".join(file_name.split("_")[0:2])
        if parent not in files:
            files[parent] = {}
        if image_id not in files[parent]:
            files[parent][image_id] = {}
        if "GT" in file_name:
            files[parent][image_id]["GT"] = str(file)
        elif "pose" in file_name:
            if "raw_images" not in files[parent][image_id]:
                files[parent][image_id]["raw_images"] = []
            files[parent][image_id]["raw_images"].append(str(file))
        elif "Parameters" in file_name:
            if "parameters" not in files[parent][image_id]:
                files[parent][image_id]["parameters"] = str(file)
        else:
            print("unknown file type", file_name)

    return files


def fix_data(files: dict) -> Tuple[dict, List]:
    """
    check for each batch and number if there are 11 raw images and 1 GT image
    files: dict returned by dict_from_data_paths
    returns: dict with only valid data and a list of errors
    """
    errors = []
    keys_to_remove = set()

    for batch in files:
        for image_id in files[batch]:
            # get the number of raw images
            try:
                number_raw_images = len(files[batch][image_id]["raw_images"])
            except KeyError:
                number_raw_images = 0

            if number_raw_images != 11:
                errors.append((batch, image_id, "raw images:", number_raw_images))
                keys_to_remove.add((batch, image_id))

            if "GT" not in files[batch][image_id]:
                errors.append((batch, image_id, "GT not found"))
                keys_to_remove.add((batch, image_id))

    # remove the keys that have errors
    for key in keys_to_remove:
        del files[key[0]][key[1]]

    return files, errors


if __name__ == "__main__":
    data_path = "../../data/download"
    data_files = get_all_files(data_path)
    files = dict_from_data_paths(data_files)
    files, errors = fix_data(files)

    # print some stats
    # number of batches
    print(len(files), "batches found")
    # number of images in each batch
    for batch in files:
        print(len(files[batch]), "images in", batch)
    # number of errors
    print(len(errors), "errors found")

    # save the data
    with open("data.json", "w") as f:
        json.dump(files, f, indent=4)

    print("data.json generated")
