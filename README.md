# Image Reconstruction, Computer Vision, WS2023/24, Group A6
We use ````PyTorchLightning````, HuggingFace ````Transformers```` to load the model weights, and ```hydra``` to manage configurations.

## Get started
For inference see the Chapter [Testing/Inference](#testinginference) and for training the model see Chapter [Data](#data) and [Training](#training).

## Directory Structure

    ../src
    ├── data   (place for all data stuff)
    │   ├── base_datamodule   (Parent LightningModule used for all datamodules)
    │   ├──  emb_datamodule   (LightningModule for loading Dinov2 Embeddings, for Conv-Decoder)
    │   ├── image_datamodule   (LightningModule for loading Images (UNet))
    │   ├── dpt_datamodule   (LightningModule for loading Images (DINOv2 + DPT Decoder))
    ├──  model   (place for all model stuff)
    │   ├── unet   (Our baseline model)
    │   ├── dinov2   (Implements DINOv2 model parts & decoders: Simple Convolutional & DPT)
    │   restoration_module.py (Base LightningModule that we use for training all models)
    ├──  utils   (some additional utils)
    test_train.py   (to do the training)
    test.py  (to make inference using a model checkpoint and input image stack)
    ../configs (lists all default configs using hydra)


Use `python generate_data_json.py` to generate the ```data_paths.json```
which saves the paths of the corresponding integral images, ground truth images and other information into a json file.

## Data

Run the ```generate_data_json.py``` on your own machine as it will use absolute paths to the data.

The folders for integral images and the given data from the lecture are defined in the script:

    raw_folder = "../data/download"
    integral_folder = "../data/integral_images"

Folder Structure of the raw data and the integral images:

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

The ````data_paths.json```` is then used by the DataSet class.

To train our models, we use a random 80/10/10 split for training, validation, and test sets, respectively.

## Training

To train a model run `test_train.py` - it takes some arguments but the current default should be fine. These are also the ones for our final model.
You can load configs via hydra:
> python3 test_train.py config-name=dino-dpt

Loads the configuration as defined in ````configs/dino-dpt.yaml`` and starts training, including wandb logging.

The above command should also reproduce our results. In this case, a focal stack for all provided images should be produced (0m, -0.5m, -1m, -1.5m). The paths of the files should then be created via ```src/data/generate_data_json.py```.

You can also pass custom parameters to hydra, e.g.:
> python3 test_train.py config-name=dino-dpt loss.msge_weight=2.0

We train our models on a cluster of 4 Nvidia GTX2080Ti GPUs. For further details on hyperparameters, please to the ```configs/``` folder and the respective model configurations.

We also provide some training, validation, and testing metrics in our [wandb report](https://api.wandb.ai/links/cv2023-a6/062b67j4).

## Testing/Inference
You can make inference on an already trained model using ````test.py``. Example usage:
> python test.py --model dpt --ckpt /path/to/file/dpt.ckp
You can also pass a folder path to the script with 4 integral images (0m, -0.5m, -1m, -1.5m in that order and 512x512 pixels).

You can find model the final model checkpoint on [Our Drive](https://drive.google.com/drive/folders/1ueuF1zs5QTb5_t6qXZaQjHwnOwg8Y_6n?usp=sharing). The config corresponds to ```configs/dino-dpt.yaml```.

The script then saves the output of the selected model and input as a .png files.

With the --testset flag it is possible to produce images on all the test set images. To replicate our test set, please see ```src/data/get_splits.py```.

For further details on our results, please refer to our report and presentation slides.

Our best model is DINOv2 (small) in combination with the DPT decoder, using the MSGE with a weight of 2 (so the final loss becomess MSE + 2*MSGE) and no oversampling, comprising roughly 24.4M parameters, of which are only 3.1M parameters are trainable - most of them (i.e., from the DINOv2 backbone) remain frozen. For the full config, please see ```configs/dino-dpt.yaml```.

Some plots of training can be found [here](https://api.wandb.ai/links/cv2023-a6/062b67j4).

The model can be downloaded on [Our Drive](https://drive.google.com/drive/folders/1ueuF1zs5QTb5_t6qXZaQjHwnOwg8Y_6n?usp=sharing). In this folder, you can also find all the test set outputs by the model.

## Testing/Inference
### Test Set
- Download the model (```model.ckpt```) from [Our Drive](https://drive.google.com/drive/folders/1ueuF1zs5QTb5_t6qXZaQjHwnOwg8Y_6n?usp=sharing) to the ````models/`` folder.H
- Download the input focal stack images and ground truth (for plotting) images from [Our Drive](https://drive.google.com/drive/folders/1ueuF1zs5QTb5_t6qXZaQjHwnOwg8Y_6n?usp=sharing) and unpack the file to the folder ```test_data_folder```. In this folder, there should then be 14145 .png files in total.
- Run ```python test.py --ckpt models/model.ckpt --mode testset --images test_data_paths.json```. By default, it uses the image paths from the downloaded test set, specified via ```used_test_data_paths.json```.
- The outputs are in ```output/``` and in the respective subdirectories, corresponding to the provided folder structure (batches/parts). We show both single output images (just the model output) as well as full output images (focal stack/input, ground truth, and model output) to get a better understanding.
- Running this should get the same results as in ```model_outputs.tar.gz``` and the ``outputs/`` folder from [Our Drive](https://drive.google.com/drive/folders/1ueuF1zs5QTb5_t6qXZaQjHwnOwg8Y_6n?usp=sharing). We put a subset of outputs into ```sample_outputs/```.

### Real Focal Stack
- Download the model (```model.ckpt```) from [Our Drive](https://drive.google.com/drive/folders/1ueuF1zs5QTb5_t6qXZaQjHwnOwg8Y_6n?usp=sharing) to the ````models/`` folder (if not already done; same as for Test Set).
- Run ```python test.py --ckpt models/model.ckpt --mode single```. By default, it uses the single provided real focal stack image (as seen in the folder ```real_focal_stack```).
- Similarly, we save both single and full output images as .png files to the current directory (output.png & output_single.png).