# Mip-NeRF++
This repository contains the code release for [Reconstruction and Rendering of Buildings as Radiance Fields for View Synthesis](http://resolver.tudelft.nl/uuid:87d5d228-e00d-4cea-9e70-985315956556). This implementation combines [NeRF++](https://github.com/Kai-46/nerfplusplus) and [mip-NeRF](https://github.com/google/mipnerf). Specifically, the foreground network of NeRF++ is replaced with mip-NeRF and the background scene still represents by NeRF. 

## Abstract
In inspection and display scenarios, reconstructing and rendering the entire surface of a building is a critical step in presenting the overall condition of the building. In building reconstruction, most works are based on point clouds because of their enhanced availability. In recent years, neural radiance fields (NeRF) have become a common function for implementing novel view synthesis. Compared to other traditional 3D graphic methods, NeRF-based models have a solid ability to produce photorealistic images with rich details that point clouds based methods cannot offer. As a result, we decided to investigate the performance of this technique in architectural scenes and look for ways to improve it for more significant scenes.

This thesis explores the ability to reconstruct large-field scenes with NeRF-based models. NeRF introduced a fully-connected network to predict the volume density and view-dependent emitted radiance at the special location, which will be projected into an image through classic volume rendering techniques. Due to the limitation of near-field ambiguity and parameterization of unbounded scenes, the original NeRF does not perform well on 360° input view, especially when the inputs are sparse. An inverted sphere parameterization that facilitates free view synthesis is introduced to address this limitation so that the foreground and background views can be trained separately. Besides that, we also compare the performance of tracing different light geometries, ray and cone, respectively. Meanwhile, to generate the reconstructed scene precisely, raw RGB images should be pre-processed to estimate the corresponding camera parameters. Finally, customized camera paths should be prepared to generate the final rendered video.

According to our experiments, training foreground and background separately is a promising method to solve practical large-scale scene reconstruction problems. A complete wrap-around view of the target building can be obtained using adjusted camera path parameters. Furthermore, introducing conical frustum casting into the original model also provides an alternative method to implement reconstruction. We named this method mip-NeRF++, which can contribute to the final results to some extent.

## Demo
<img src="https://github.com/147-Enpu/mipnerfplusplus/blob/master/demo/buildings.gif" width="320" alt="buildings">  <img src="https://github.com/147-Enpu/mipnerfplusplus/blob/master/demo/train.gif" width="320" alt="train">

## Data
Download our pre-processed large-scale scene [buildings](https://drive.google.com/drive/folders/1SO6ku2NWfjezbLM8tZ28KmCSeTcW_-OH?usp=sharing), or the pre-processed [tanks_and_temples](https://drive.google.com/file/d/11KRfN91W1AxAW6lOFs4EeYDbeoQZCi87/view?usp=sharing). And put the data in the sub-folder data/ of this code directory.
### Prepare your own dataset
To prepare a dataset with your images, [COLMAP](https://colmap.github.io/) needs to be installed firstly. Then follow the instruction from [NeRF++](https://github.com/Kai-46/nerfplusplus#generate-camera-parameters-intrinsics-and-poses-with-colmap-sfm) to generate JSON files. Finally, use the scripts in 'dataset_construct.py' to generate the dataset as following structure:

```bash
<dataset_name>
|-- train
    |-- rgb
        |-- 0.png        # target image for each view
        |-- 1.png
        ...
    |-- pose
        |-- 0.txt        # camera pose for each view (4x4 matrices)
        |-- 1.txt
        ...
    |-- intrinsic
        |-- 0.txt        # camera intrinsics for each view (4x4 matrices)
        |-- 1.txt
        ...
|-- test
    ...
|-- validation
    ...
|-- camera path          # camera path for rendering purpose      
    |-- pose
        |-- 0.txt        # camera pose for each view (4x4 matrices)
        |-- 1.txt
        ...
    |-- intrinsic
        |-- 0.txt        # camera intrinsics for each view (4x4 matrices)
        |-- 1.txt
        ...
```
## Create environment
```bash
conda env create --file environment.yml
conda activate mipnerfplus
```
## Quick start
The training script is in 'ddp_train_nerf.py', to train a mip-NeRF++:
```python
python ddp_train_nerf.py --config configs/tanks_and_temples/tat_training_truck.txt
```

## Test and render
```python
python ddp_test_nerf.py --config configs/tanks_and_temples/tat_training_truck.txt \
                        --render_splits test,camera_path
```

## Camera path preparation
You can use the script in `camera_path` to define the locations of cameras which are used to observe the reconstructed target and generate videos as shown in the demo.

* Use 'generate_camera_path.py' to define a batch of cameras on a circle. 
* Run 'normailze.py' to move the average camera center to origin, and put all the camera centers inside the unit sphere.
* 'visualiza_camera.py' can compare the relative location of two sets of camera path, it may be helpful to adjust the parameters in previous two steps. 
