## Generating the integral images

First follow the environment setup and other instructions in the [README_AOS.md](README_AOS.md) file,
which is from the original project instructions.

### Create helper json file with paths to the data

In the `generate_data_json.py` file, change the

```python
raw_folder = "../../data/download"
```

to the path of the data on your machine.

The image folder structure should be as follows:

```
../data/download
├── batch_20230912_part1-006
│   ├── Part1
│   │   ├──  0_0_Parameters.txt
│   │   ├──  0_0_pose_0_thermal.png
│   │   ├──  ...
│   │   ├──  0_0_GT_pose_0_thermal.png
├── batch_20230912_part2-003
│   ├── Part1
│   │   ├──  0_5500_Parameters.txt
│   │   ├──  ...
├── ...
```

Then run

```
> python generate_data_json.py
```

This will generate a `data.json` file with the paths to the data that is then used to generate the integral images.

### Create integral images

In the integral generation process the ``AOS_integrator.py`` provides the main function to generate the integral images.
Here original script that was provided, is adapted to work as a function that can be imported into other scripts.
In this file the right functions from the LFR folder must be imported (again see [README_AOS.md](README_AOS.md) for
installation instructions of the AOS dependencies) that are not included here.

After that, the integral images are generated using a script that uses multiprocessing to speed up the process:

In the `integral_generation_script.py` file, you can change the folder where the integral images are saved to:

```python
integral_folder = "../../data/integral_images"
```

And make sure the `data.json` file is in the same folder as the script or referenced correctly.

```python
data_json_path = "data.json"
```

Here you can also select which focal planes to generate. In our case we used 0m, -0.5m, -1m and 1.5m.

```python
focal_planes = [0.0, -0.5, -1.0, -1.5]
```

Then run

```
> python integral_generation_script.py
```

This will generate the integral images and save them to the folder specified in the script.
The resulting folder structure looks like this:

```
../data/integral_images
├── batch_20230912_part1-006
│   ├── Part1
│   │   ├──  0_0
│   │   │   ├──  0_0_integral_fp-0.0.png
│   │   │   ├──  0_0_integral_fp-0.5.png
│   │   │   ├──  ...
│   │   │   ├──  0_0_integral_fp-1.5.png
│   │   ├──  ...
├── batch_20230912_part2-003
│   ├── Part1
│   │   ├──  0_5500
│   │   │   ├──  0_5500_integral_fp-0.0.png
│   │   │   ├──  0_5500_integral_fp-0.5.png
│   │   │   ├──  ...
│   │   │   ├──  0_5500_integral_fp-1.5.png
│   │   ├──  ...
├── ...
```
