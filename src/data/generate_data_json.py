from pathlib import Path
import json


def generate_data_paths_dict(raw_folder, integral_folder, embedding_folder, focal_planes=None):
    if focal_planes is None:
        focal_planes = [0.0, -0.5, -1, -1.5]

    integral_files = Path(integral_folder).rglob("*.png")
    raw_files = list(Path(raw_folder).rglob("*.png"))
    raw_files += Path(raw_folder).rglob("*.txt")
    embedding_files = Path(embedding_folder).rglob("*.safetensors")

    # setup dict using integral files
    data_paths = {}
    for int_image in integral_files:
        batch_part = Path("/".join(int_image.parent.parts[-3:-1])).__str__()
        file_name = int_image.name
        image_focal_plane = float(int_image.stem.split("_")[-1][2:])
        image_id = "_".join(file_name.split("_")[0:2])
        file_str = int_image.absolute().__str__()

        if batch_part not in data_paths:
            data_paths[batch_part] = {}
        if image_id not in data_paths[batch_part]:
            data_paths[batch_part][image_id] = {}
        if "integral_images" not in data_paths[batch_part][image_id]:
            data_paths[batch_part][image_id]["integral_images"] = []
        if image_focal_plane in focal_planes:
            data_paths[batch_part][image_id]["integral_images"].append(file_str)

    for embedd in embedding_files:
        batch_part = Path("/".join(embedd.parent.parts[-3:-1])).__str__()
        file_name = embedd.name
        image_focal_plane = float(embedd.stem.split("_")[-1][2:])
        image_id = "_".join(file_name.split("_")[0:2])
        file_str = embedd.absolute().__str__()

        if batch_part not in data_paths:
            data_paths[batch_part] = {}
        if image_id not in data_paths[batch_part]:
            data_paths[batch_part][image_id] = {}
        if "embeddings" not in data_paths[batch_part][image_id]:
            data_paths[batch_part][image_id]["embeddings"] = []
        if image_focal_plane in focal_planes:
            data_paths[batch_part][image_id]["embeddings"].append(file_str)

    # add raw images to dict
    for raw_file in raw_files:
        batch_part = Path("/".join(raw_file.parent.parts[-2:])).__str__()
        file_name = raw_file.name
        image_id = "_".join(file_name.split("_")[0:2])

        file_str = raw_file.absolute().__str__()

        try:
            if "GT" in file_name:
                data_paths[batch_part][image_id]["GT"] = file_str
            elif "pose" in file_name:
                if "raw_images" not in data_paths[batch_part][image_id]:
                    data_paths[batch_part][image_id]["raw_images"] = []
                data_paths[batch_part][image_id]["raw_images"].append(file_str)
            elif "Parameters" in file_name:
                data_paths[batch_part][image_id]["parameters"] = file_str
        except KeyError:
            continue

    # check if each sample has 11 raw images, 1 GT image, 1 parameter file and as many integral images as focal planes
    image_ids_to_remove = []
    for batch in data_paths:
        for image_id in data_paths[batch]:
            errors = []
            # check if all the keys are there (raw images, GT, parameters, integral images)
            # are here and only continue if there are 5 keys
            keys = data_paths[batch][image_id].keys()
            if len(keys) == 5:
                if len(data_paths[batch][image_id]["raw_images"]) != 11:
                    errors.append("raw images not 11")
                if "GT" not in data_paths[batch][image_id]:
                    errors.append("GT not found")
                if "parameters" not in data_paths[batch][image_id]:
                    errors.append("parameters not found")
                n_integral_images = len(data_paths[batch][image_id]["integral_images"])
                if n_integral_images < len(focal_planes):
                    errors.append(f"integral images are {n_integral_images} not {len(focal_planes)}")
                n_embeddings = len(data_paths[batch][image_id]["embeddings"])
                if n_embeddings < len(focal_planes):
                    errors.append(f"embeddings are {n_integral_images} not {len(focal_planes)}")
            else:
                errors.append(f"sample is missing something, keys are {keys}")

            if len(errors) > 0:
                print(batch, image_id, errors)
                image_ids_to_remove.append((batch, image_id))

    # remove samples with errors
    for batch, image_id in image_ids_to_remove:
        del data_paths[batch][image_id]
        if len(data_paths[batch]) == 0:
            del data_paths[batch]

    return data_paths


if __name__ == "__main__":

    p = r"C:\Users\pauld\OneDrive - Johannes Kepler Universität Linz\Master\Semester1\Computer Vision\project\data"
    raw_folder = p + r"\download"
    integral_folder = p + r"\integral_images"
    embedding_folder = r"D:\Computer_Vision\small"

    data_paths = generate_data_paths_dict(raw_folder, integral_folder, embedding_folder)

    # print some stats
    print("number of batches:", len(data_paths))
    print("number of integral images:",
          sum([len(data_paths[batch][image_id]["integral_images"]) for batch in data_paths for image_id in
               data_paths[batch]]))
    print("number of embeddings:",
          sum([len(data_paths[batch][image_id]["embeddings"]) for batch in data_paths for image_id in
               data_paths[batch]]))
    print("number of raw images:",
          sum([len(data_paths[batch][image_id]["raw_images"]) for batch in data_paths for image_id in
               data_paths[batch]]))
    print("number of GT images:",
          sum(["GT" in data_paths[batch][image_id] for batch in data_paths for image_id in data_paths[batch]]))
    print("number of parameter files:",
          sum(["parameters" in data_paths[batch][image_id] for batch in data_paths for image_id in data_paths[batch]]))

    # change data_paths to a list of sample dicts -->
    # [{"batch": batch, "image_id": image_id, "GT": GT, "raw_images": raw_images, "parameters": parameters, "integral_images": integral_images}, ...]
    data_paths_list = []
    for batch in data_paths:
        for image_id in data_paths[batch]:
            sample = {"batch": batch,
                      "image_id": image_id,
                      "GT": data_paths[batch][image_id]["GT"],
                      "raw_images": data_paths[batch][image_id]["raw_images"],
                      "parameters": data_paths[batch][image_id]["parameters"],
                      "integral_images": data_paths[batch][image_id]["integral_images"],
                      "embeddings": data_paths[batch][image_id]["embeddings"]
                      }
            data_paths_list.append(sample)

    # save data_paths_list as json
    with open('data_paths.json', 'w') as outfile:
        json.dump(data_paths_list, outfile, indent=4)
