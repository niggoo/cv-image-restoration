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