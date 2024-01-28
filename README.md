# Image Reconstruction, Computer Vision, WS2023/24, Group A6

Please follow the steps outlined in this document to generate predictions based on our model.

## Python environment

We recommend using "conda" to create a new environment with the required packages. To do so, run the following commands
in the root directory of this repository:

```
conda env create -f env.yml
conda activate cv-a6
```

This will create a new environment and install all required packages into it. Then, the environment is activated.

If you encounter any issues with torch, torchvision and/or matpotlib, we recommend installing these or all packages via "pip".
If torch/torchvision causes issues, first uninstall both via ```pip uninstall torch torchvision```. Then, reinstall them via pip: ```pip install torch==1.12.1 torchvision==0.13.1 --index-url https://download.pytorch.org/whl/cu102```. You may also adapt this to your local cuda version by changing "cu102" to, e.g., "cu113" or, if only the cpu should be used, to "cpu"

Alternatively, you can install everything via pip:
```
conda env create cv-a6-pip
conda activate cv-a6-pip
pip install torch==1.12.1 torchvision==0.13.1 --index-url https://download.pytorch.org/whl/cu102
pip install -r requirements.txt
```

For the first generation of the integral images, you need to create another environment
which follows the original project instructions for the AOS installation.
Please follow the instructions in the [integral generation README.md](src/data/generate_integral_images/README.md) file.

## Download model

Download the file `model.ckpt` from
this [link](https://drive.google.com/drive/folders/1ueuF1zs5QTb5_t6qXZaQjHwnOwg8Y_6n?usp=sharing) directly into the
folder `weights`.

## Prepare input data

Our `test.py` script can operate in two modes:

- "single": The script takes a single focal stack and generates a single prediction based on it. For this purpose you
  should put your focal stack consisting of four images {0m, 0.5m, 1m, 1.5m} into the `single_input`-folder.
- "testset": The script generates predictions for an entire set of images. For this purpose you should download the
  file `test_data.tar.gz` from
  our [drive](https://drive.google.com/drive/folders/1ueuF1zs5QTb5_t6qXZaQjHwnOwg8Y_6n?usp=sharing) and extract the
  contents directly into the `test_data`-folder. There should be 14145 png-files in total. The
  file `test_data_paths.json` in the root directory is a helper-file for generating the predictions and should be left
  as it is.

## Testing/Inference

To generate predictions for a single focal stack, run
```
python test.py
```
This command takes the focal stack from `single_input` and places its predictions into `results`.

To generate predictions for all the images in `test_data`,
run 
```
python test.py --mode testset --images test_data_paths.json
```
This command takes the images from the folder and
places their predictions into 'results' into subfolders corresponding to the structure provided to us (batches). We save
both a single output image as well as a graph comparing input, output and ground truth. The results should be the same
as in the outputs-folder in
our [drive](https://drive.google.com/drive/folders/1ueuF1zs5QTb5_t6qXZaQjHwnOwg8Y_6n?usp=sharing). Please note that this
testset-mode uses the test set by default. To generate predictions for other inputs please use the
single-mode, or generate paths for training & validation splits via ```src/data/get_splits.py``` and pass the desired json file to our test function.

These parameters can additionally be set for the command:

- --ckpt: Path to the model checkpoint/weights. Default: "./weights/model.ckpt"
- --output: Path to save the predictions at. Default: "./results"
- --mode: Whether to generate predictions for a single focal stack or for an entire set: Options: ["single", "testset"].
  Default: "single".
- --images:
    - In single-mode: A list of four focal stack images or the path to the directory containing them. Default: "
      ./single_input".
    - In testset-mode: Path to the json-file containing the data for the testset.

## Training

### Preparing the Data

If the integral images are not yet generated, please follow the instructions in
the [integral generation README.md](src/data/generate_integral_images/README.md) file in `src/data/generate_integral_images`.
Using our integral generation process makes sure the correct folder and name structure is generated.

When the integral images (focal planes of 0m, -0.5m, -1m and 1.5m) exist and are in the below mentioned folder and name structure,
run the ```src/data/generate_data_json.py``` script on your own machine as it will use absolute paths to the data.

The folders for integral images and the given data from the lecture are given as command line arguments:

```
python generate_data_json.py --raw_dir <path_to_raw_data> --integral_dir <path_to_integral_images>
```

The script will generate a ```data_paths.json``` file in the directory.
This file contains the paths to the data and is used by the DataSet class.

Optionally you can provide a ```--emb_dir``` to add the paths to precomputed DINOV2 embeddings to the json file. These, however, are not needed for our final mdoel.

Folder Structure of the raw data and the integral images must be as follows:

    ../data/download
    ├── batch_20230912_part1-006
    │   ├── Part1
    │   │   ├──  0_0_Parameters.txt
    │   │   ├──  0_0_pose_0_thermal.png
    │   │   ├──  ...
    │   │   ├──  0_0_GT_pose_0_thermal.png
    ├──  ...
    
    ../data/integral_images
    ├── batch_20230912_part1-006
    │   ├── Part1
    │   │   ├──  0_0
    │   │   │   ├──  0_0_integral_fp-0.0.png
    │   │   │   ├──  0_0_integral_fp-0.5.png
    │   │   │   ├──  ...
    │   │   │   ├──  0_0_integral_fp-1.5.png
    │   │   ├──  ...
    ├──  ...

The folder structure for the optional DINOv2 embeddings is not shown but the same as for integral images.

To train our models, we use a random 80/10/10 split for training, validation, and test sets, respectively.

### Training

To train a model run `train.py` - it takes some arguments but the current default should be fine. These are also
the ones for our final model. So, running this is sufficient:
> python3 train.py

You can load configs via hydra:
> python3 train.py config-name=dino-dpt

Loads the configuration as defined in ```configs/dino-dpt.yaml``` and starts training, including wandb logging.

The wandb logging can be setup in the config file, make sure to use your own wandb account infos or disable logging by passing ```logging=False``` flag.

The above command should also reproduce our results. In this case, a focal stack for all provided images should be
produced (0m, -0.5m, -1m, -1.5m). The paths of the files should then be created
via ```src/data/generate_data_json.py```.

You can also pass custom parameters to hydra, e.g.:
> python3 train.py config-name=dino-dpt loss.msge_weight=2.0

We train our models on a cluster of 4 Nvidia GTX2080Ti GPUs. For further details on hyperparameters, please to
the ```configs/``` folder and the respective model configurations.

We also provide some training, validation, and testing metrics of our final model in
our [wandb report](https://api.wandb.ai/links/cv2023-a6/062b67j4).
