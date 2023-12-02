Directory Structure:

    ../src
    ├── data   (place for all data stuff)
    │   ├──  emb_datamodule   (LightningModule for loading Dinov2 Embeddings)
    │   ├── image_datamodule   (LightningModule for loading Images)
    ├──  model   (place for all model stuff)
    │   ├── unet   (Our baseline model)
    │   │   ├──  unet.py   (implements the unet model)
    │   ├── dinov2   (Our hopefully better than baseline model)
    │   │   ├──  something.py   (should implement dinov2 so we can import it as a torch.nn.Module)
    │   ├── ...   (go ahead an implement any model you would like to try) 
    test_train.py   (the place to dry run your codes and test the implementation)

Use `python generate_data_json.py` to generate the ```data_paths.json```
which saves the paths of the corresponding integral images, ground truth images and other information into a json file.

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

!Disclaimer: If you make any changes to `test_train.py`, you should add it to gitignore as changes may break stuff!

To train your model run `test_train.py` - it takes some arguments but the current default should be fine. To try your model
implement a fitting(if not already existing) DataModule in ../src/data and a Model in ../src/model and import them in
`test_train.py` as DataModule and Model. You may need to adapt the model init further down the code.
