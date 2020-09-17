# YOLO

Custom implementation of Yolo in Python among Darknet framework.
The module was written based on original available [here](https://github.com/AlexeyAB/darknet/blob/master/darknet.py).

## Prerequisites

This software works only on Linux systems.
These instructions will get you a copy of the project up and running on your local machine for development and testing purposes.

### Download

First of all, you need to download the software through cloning.

```sh
git clone https://github.com/gmurro/darknet.git
cd darknet
```

### Configuration

To run the detector, you need to compile [Darknet](https://github.com/AlexeyAB/darknet) on your pc to build a library libdarknet.so.
See the [instructions](https://github.com/gmurro/darknet/INSTRUCTIONS.md) to compile Darknet.

After doing this, copy on 'darknet' folder the library libdarknet.so.
In the same directory there are also: 
1. `cfg-files` - are structures of neural networks 
    * COCO DATASET: coco.data, coco.names, yolov4-tiny.cfg
    * VISDRONE-2019 DATASET: obj.data, obj.names, yolov3tiny.cfg
2. `weights-files` - are weights for correspond cfg-file
    * COCO DATASET: yolov4-tiny.weights
    * VISDRONE-2019 DATASET: yolov3tiny.weights
        
Inside the [darknet.py](https://github.com/gmurro/darknet/darknet.py) file you can set within the global variables the path of the weights, libdarknet.so and configuration files to use :  
```python
LIBDARKNET = os.path.dirname(os.path.realpath(__file__)) + "/libdarknet.so"
CFG_FILE = os.path.dirname(os.path.realpath(__file__)) + "/cfg/yolov4-tiny.cfg"
DATA_FILE = os.path.dirname(os.path.realpath(__file__)) + "/cfg/coco.data"
NAMES_FILE = os.path.dirname(os.path.realpath(__file__)) + "/cfg/coco.names"
WEIGHTS_FILE = os.path.dirname(os.path.realpath(__file__)) + "/yolov4-tiny.weights"
```

The script will detect object within the images saved in "[frame](https://github.com/gmurro/darknet/frame)" folder.
You can replace them with others images on witch you want perform detection.

## Run

To lauch the script, from the terminal execute `python3 darknet.py`. 
The results are saved in the of images and annotations within "frame/res" folder.
For each images, annotations of bounding boxes are saved in this format:

```sh
<label> <top_left_x> <top_left_y> <height> <width> <confidence>

    <label>: label index of the object detected inside the bounding box
    <top_left_x>: ordinate position of the point at the top left of the bounding box
    <top_left_y>: abscissa position of the point at the top left the bounding box
    <height>: height of the bounding box
    <width>: width of the bounding box
    <confidence>: it reflects how likely the box contains an object and how accurate is the boundary box
```

### Usage

You can easily use this in you project. You just need to import darknet file and than call the following functions:
* loadConfiguration()
* performDetect()

Consult their documentation to learn about their use.

## Authors

* **[Giuseppe Murro](https://github.com/gmurro)** - _Customization of the darknet.py module implemented by [AlexeyAB](https://github.com/AlexeyAB)_ 

## License

This project is licensed under the MIT License - see the [LICENSE](https://github.com/gmurro/darknet/blob/master/LICENSE.md) file for details.