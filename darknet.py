#!python3
"""
Python 3 wrapper for identifying objects in images using Darknet (https://github.com/AlexeyAB/darknet).
It run only on Linux systems and libdarknets.so file is required.
Windows version requires dll and code to manage it (see original https://github.com/AlexeyAB/darknet/blob/master/darknet.py)

To use, is necessary call loadConfiguration() and performDetect() after import.
See the docstring of loadConfiguration() and performDetect() for parameters.

You can try this detector using main() funcion, that run detection on images in 'frame' folder.
It's necessary that the paths of files definited in variables LIBDARKNET, CFG_FILE, DATA_FILE,
WEIGHTS_FILE and NAMES_FILE are correct.
Modify variables LIBDARKNET, CFG_FILE, DATA_FILE, WEIGHTS_FILE, NAMES_FILE to change configuration files of neural network.

The module was written based on original available here: https://github.com/AlexeyAB/darknet/blob/master/darknet.py

@author: Giuseppe Murro
"""

from ctypes import *
import random
import os
import cv2
import time
import base64
import numpy as np
from PIL import Image
import io

# change these if path of configuration files is different
LIBDARKNET = os.path.dirname(os.path.realpath(__file__)) + "/libdarknet.so"
CFG_FILE = os.path.dirname(os.path.realpath(__file__)) + "/cfg/yolov4-tiny.cfg"
DATA_FILE = os.path.dirname(os.path.realpath(__file__)) + "/cfg/coco.data"
NAMES_FILE = os.path.dirname(os.path.realpath(__file__)) + "/cfg/coco.names"
WEIGHTS_FILE = os.path.dirname(os.path.realpath(__file__)) + "/yolov4-tiny.weights"


def editDataFile(dataFile, namesFile):
    """
    Function that insert namesFile in "names" row of dataFile.
    If dataFile or namesFile isn't a valid file, raise ValueError.

    Parameters
    ----------------
    dataFile: str
        Path of data file for Darknet (es. coco.data)
    namesFile: str
        Path of names file for Darknet (es. coco.names)
    """

    try:
        with open(dataFile, 'r') as file:
            lines = file.readlines()

        for i in range(0, len(lines)):
            if ("names" in lines[i]):
                lines[i] = "names = " + namesFile + "\n"

        with open(dataFile, 'w') as file:
            file.writelines(lines)
    except:
        raise ValueError("Error during editing dataFile: `" + dataFile + "`")


def sample(probs):
    s = sum(probs)
    probs = [a / s for a in probs]
    r = random.uniform(0, 1)
    for i in range(len(probs)):
        r = r - probs[i]
        if r <= 0:
            return i
    return len(probs) - 1


def c_array(ctype, values):
    arr = (ctype * len(values))()
    arr[:] = values
    return arr


class BOX(Structure):
    _fields_ = [("x", c_float),
                ("y", c_float),
                ("w", c_float),
                ("h", c_float)]


class DETECTION(Structure):
    _fields_ = [("bbox", BOX),
                ("classes", c_int),
                ("prob", POINTER(c_float)),
                ("mask", POINTER(c_float)),
                ("objectness", c_float),
                ("sort_class", c_int),
                ("uc", POINTER(c_float)),
                ("points", c_int)]


class IMAGE(Structure):
    _fields_ = [("w", c_int),
                ("h", c_int),
                ("c", c_int),
                ("data", POINTER(c_float))]


class METADATA(Structure):
    _fields_ = [("classes", c_int),
                ("names", POINTER(c_char_p))]


# darknet is execute with GPU is hasGPU=True
hasGPU = True

# load libdarknet.so librady of darknet wrote in c
lib = CDLL(LIBDARKNET, RTLD_GLOBAL)
lib.network_width.argtypes = [c_void_p]
lib.network_width.restype = c_int
lib.network_height.argtypes = [c_void_p]
lib.network_height.restype = c_int

copy_image_from_bytes = lib.copy_image_from_bytes
copy_image_from_bytes.argtypes = [IMAGE, c_char_p]


def network_width(net):
    return lib.network_width(net)


def network_height(net):
    return lib.network_height(net)


predict = lib.network_predict_ptr
predict.argtypes = [c_void_p, POINTER(c_float)]
predict.restype = POINTER(c_float)

if hasGPU:
    set_gpu = lib.cuda_set_device
    set_gpu.argtypes = [c_int]

init_cpu = lib.init_cpu

make_image = lib.make_image
make_image.argtypes = [c_int, c_int, c_int]
make_image.restype = IMAGE

get_network_boxes = lib.get_network_boxes
get_network_boxes.argtypes = [c_void_p, c_int, c_int, c_float, c_float, POINTER(c_int), c_int, POINTER(c_int), c_int]
get_network_boxes.restype = POINTER(DETECTION)

make_network_boxes = lib.make_network_boxes
make_network_boxes.argtypes = [c_void_p]
make_network_boxes.restype = POINTER(DETECTION)

free_detections = lib.free_detections
free_detections.argtypes = [POINTER(DETECTION), c_int]

free_ptrs = lib.free_ptrs
free_ptrs.argtypes = [POINTER(c_void_p), c_int]

network_predict = lib.network_predict_ptr
network_predict.argtypes = [c_void_p, POINTER(c_float)]

reset_rnn = lib.reset_rnn
reset_rnn.argtypes = [c_void_p]

load_net = lib.load_network
load_net.argtypes = [c_char_p, c_char_p, c_int]
load_net.restype = c_void_p

load_net_custom = lib.load_network_custom
load_net_custom.argtypes = [c_char_p, c_char_p, c_int, c_int]
load_net_custom.restype = c_void_p

do_nms_obj = lib.do_nms_obj
do_nms_obj.argtypes = [POINTER(DETECTION), c_int, c_int, c_float]

do_nms_sort = lib.do_nms_sort
do_nms_sort.argtypes = [POINTER(DETECTION), c_int, c_int, c_float]

free_image = lib.free_image
free_image.argtypes = [IMAGE]

letterbox_image = lib.letterbox_image
letterbox_image.argtypes = [IMAGE, c_int, c_int]
letterbox_image.restype = IMAGE

load_meta = lib.get_metadata
lib.get_metadata.argtypes = [c_char_p]
lib.get_metadata.restype = METADATA

load_image = lib.load_image_color
load_image.argtypes = [c_char_p, c_int, c_int]
load_image.restype = IMAGE

rgbgr_image = lib.rgbgr_image
rgbgr_image.argtypes = [IMAGE]

predict_image = lib.network_predict_image
predict_image.argtypes = [c_void_p, IMAGE]
predict_image.restype = POINTER(c_float)

predict_image_letterbox = lib.network_predict_image_letterbox
predict_image_letterbox.argtypes = [c_void_p, IMAGE]
predict_image_letterbox.restype = POINTER(c_float)


def array_to_image(arr):
    import numpy as np
    # need to return old values to avoid python freeing memory
    arr = arr.transpose(2, 0, 1)
    c = arr.shape[0]
    h = arr.shape[1]
    w = arr.shape[2]
    arr = np.ascontiguousarray(arr.flat, dtype=np.float32) / 255.0
    data = arr.ctypes.data_as(POINTER(c_float))
    im = IMAGE(w, h, c, data)
    return im, arr


def classify(net, meta, im, altNames):
    out = predict_image(net, im)
    res = []
    for i in range(meta.classes):
        if altNames is None:
            nameTag = meta.names[i]
        else:
            nameTag = altNames[i]
        res.append((nameTag, out[i]))
    res = sorted(res, key=lambda x: -x[1])
    return res


def detect(net, meta, altNames, im, thresh=.5, hier_thresh=.5, nms=.45, debug=False):
    """
    Performs the meat of the detection
    """
    ret = detect_image(net, meta, altNames, im, thresh, hier_thresh, nms, debug)
    free_image(im)
    if debug: print("freed image")
    return ret


def detect_image(net, meta, altNames, im, thresh=.5, hier_thresh=.5, nms=.45, debug=False):
    # import cv2
    # custom_image_bgr = cv2.imread(image) # use: detect(,,imagePath,)
    # custom_image = cv2.cvtColor(custom_image_bgr, cv2.COLOR_BGR2RGB)
    # custom_image = cv2.resize(custom_image,(lib.network_width(net), lib.network_height(net)), interpolation = cv2.INTER_LINEAR)
    # import scipy.misc
    # custom_image = scipy.misc.imread(image)
    # im, arr = array_to_image(custom_image)		# you should comment line below: free_image(im)
    num = c_int(0)
    if debug: print("Assigned num")
    pnum = pointer(num)
    if debug: print("Assigned pnum")
    predict_image(net, im)
    letter_box = 0
    # predict_image_letterbox(net, im)
    # letter_box = 1
    if debug: print("did prediction")
    # dets = get_network_boxes(net, custom_image_bgr.shape[1], custom_image_bgr.shape[0], thresh, hier_thresh, None, 0, pnum, letter_box) # OpenCV
    dets = get_network_boxes(net, im.w, im.h, thresh, hier_thresh, None, 0, pnum, letter_box)
    if debug: print("Got dets")
    num = pnum[0]
    if debug: print("got zeroth index of pnum")
    if nms:
        do_nms_sort(dets, num, meta.classes, nms)
    if debug: print("did sort")
    res = []
    if debug: print("about to range")
    for j in range(num):
        if debug: print("Ranging on " + str(j) + " of " + str(num))
        if debug: print("Classes: " + str(meta), meta.classes, meta.names)
        for i in range(meta.classes):
            if debug: print("Class-ranging on " + str(i) + " of " + str(meta.classes) + "= " + str(dets[j].prob[i]))
            if dets[j].prob[i] > 0:
                b = dets[j].bbox
                if altNames is None:
                    nameTag = meta.names[i]
                else:
                    nameTag = altNames[i]
                if debug:
                    print("Got bbox", b)
                    print(nameTag)
                    print(dets[j].prob[i])
                    print((b.x, b.y, b.w, b.h))
                res.append((nameTag, dets[j].prob[i], (b.x, b.y, b.w, b.h)))
    if debug: print("did range")
    res = sorted(res, key=lambda x: -x[1])
    if debug: print("did sort")
    free_detections(dets, num)
    if debug: print("freed detections")
    return res


def loadConfiguration(configPath=CFG_FILE, weightPath=WEIGHTS_FILE, metaPath=DATA_FILE, namesPath=NAMES_FILE):
    """
    Function that load cfg, weights and names_list for Darknet.

    Parameters
    ----------------
    configPath: str
        (default= CFG_FILE)
        Path to the configuration file. Raises ValueError if not found

    weightPath: str
        (default= WEIGHT_FILE)
        Path to the weights file. Raises ValueError if not found

    metaPath: str
        (default= DATA_FILE)
        Path to the data file. Raises ValueError if not found

    namesPath: str
        (default= NAMES_FILE)
        Path to the labels file. Raises ValueError if not found

    Returns
    ----------------------
    return: tuple
        Return tuple where:
            [0] = neural network loaded
            [1] = meta data loaded
            [2] = names readed from namesPath
    """

    if not os.path.exists(configPath):
        raise ValueError("Invalid config path `" + os.path.abspath(configPath) + "`")
    if not os.path.exists(weightPath):
        raise ValueError("Invalid weight path `" + os.path.abspath(weightPath) + "`")
    if not os.path.exists(metaPath):
        raise ValueError("Invalid data file path `" + os.path.abspath(metaPath) + "`")
    if not os.path.exists(namesPath):
        raise ValueError("Invalid names file path `" + os.path.abspath(namesPath) + "`")

    # write NAMES_FILE in DATA_FILE as default
    editDataFile(metaPath, namesPath)

    netMain = load_net_custom(configPath.encode("ascii"), weightPath.encode("ascii"), 0, 1)  # batch size = 1
    metaMain = load_meta(metaPath.encode("ascii"))

    # Read the names file and create a list to feed to detect
    altNames = None
    try:
        with open(metaPath) as metaFH:
            metaContents = metaFH.read()
            import re
            match = re.search("names *= *(.*)$", metaContents, re.IGNORECASE | re.MULTILINE)
            if match:
                result = match.group(1)
            else:
                result = None
            try:
                if os.path.exists(result):
                    with open(result) as namesFH:
                        namesList = namesFH.read().strip().split("\n")
                        altNames = [x.strip() for x in namesList]
            except TypeError:
                pass
    except Exception:
        pass

    return netMain, metaMain, altNames


def performDetect(netMain, metaMain, altNames, image, base64EncodedeImage=True, returnImage=False, thresh=0.25):
    """
    Convenience function to handle the detection and returns image detection results.

    Parameters
    ----------------
    netMain: object
        Neural network loaded with loadConfiguration()[0]

    metaMain: object
        Meta data loaded with loadConfiguration()[1]

    altNames: list
        Names readed from namesPath with loadConfiguration()[2]

    image: str
        If base64EncodedeImage=False, it's the path to the image to evaluate. Raises ValueError if not found
        If base64EncodedeImage=True, it's a base64 encoded image. Raises ValueError if base64 encoded isn't correct

    base64EncodedeImage: bool
        (default= True)
        It determines if the first parameter is a path or a base64 image

    returnImage: bool
        (default= False)
        Compute (and draw) bounding boxes on image. Changes return.

    thresh: float
        (default= 0.25)
        The detection threshold
        Raises ValueError if isn't 0<thresh<1


    Returns
    ----------------------
    return:
        When returnImage is False, a list of bbox dict like as :
            [{
            	"label" : label of the object detected within bbox,
                "top_left_x": ordinate position of the point at the top left of the bbox,
                "top_left_y": abscissa position of the point at the top left the bbox,
                "height": height of the bbox,
                "width": width of the bbox,
                "confidence": confidence
            }]

        Otherwise, a tuple with:
            [0] = list of bbox dict,
            [1] = image where bboxes have been drawn n numpy.ndarray format
    """

    if (not (0 < thresh < 1)):
        raise ValueError("Threshold should be a float between zero and one (non-inclusive)")

    # if base64EncodedeImage=True, image passed as argument is encoded in base64 format
    if (base64EncodedeImage):
        try:
            # the base64 format accepted is without header (es. data:image/png;base64,)
            if "base64," in image:
                image = image.split("base64,")[1]
            # image is converted in numpy format array
            imageArray = np.array(Image.open(io.BytesIO(base64.b64decode(image))))
        except Exception:
            raise ValueError("Invalid base64 encoded image")

        imgHeight, imgWidth, _ = imageArray.shape
        darknetImage = make_image(imgWidth, imgHeight, 3)
        copy_image_from_bytes(darknetImage, imageArray.tobytes())
        detections = detect_image(netMain, metaMain, altNames, darknetImage, thresh)
    else:
        # if base64EncodedeImage=False, image passed as argument is a path
        if not os.path.exists(image):
            raise ValueError("Invalid image path `" + os.path.abspath(image) + "`")
        # Do the detection
        im = load_image(image.encode('ascii'), 0, 0)
        detections = detect(netMain, metaMain, altNames, im, thresh)

    bboxes = list()

    for detection in detections:
        # label is the class detected by the neural network inside the bounding box
        label = detection[0]

        # conficence is accuracy of the class detected by the neural network inside the bounding box
        confidence = detection[1]

        # bounds is a list that represent the bounding box in format (x_center, y_center, width, height)
        bounds = detection[2]

        bboxes.append({
            "label": label,
            "top_left_x": bounds[0] - bounds[2] / 2,
            "top_left_y": bounds[1] - bounds[3] / 2,
            "height": bounds[3],
            "width": bounds[2],
            "confidence": confidence
        })

    if (returnImage):
        # return list of bboxes and cv2 image
        if (not (base64EncodedeImage)):
            imageArray = cv2.imread(image)
        for box in bboxes:
            label = box['label'] + ": " + str(round(box['confidence'], 2)) + "%"
            left = int(box['top_left_x'])
            top = int(box['top_left_y'])
            right = int(box['top_left_x']) + int(box['width'])
            bottom = int(box['top_left_y']) + int(box['height'])
            imgHeight, imgWidth, _ = imageArray.shape
            thick = 2

            if box['confidence'] < 0.25:
                color = (0, 0, 255)
            elif 0.25 <= box['confidence'] < 0.50:
                color = (0, 102, 255)
            elif 0.50 <= box['confidence'] < 0.75:
                color = (0, 255, 255)
            elif box['confidence'] >= 0.75:
                color = (0, 255, 0)

            labelSize = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
            _y1 = top
            _x1 = left

            _x2 = _x1 + labelSize[0][0]
            _y2 = top - int(labelSize[0][1])-5
            cv2.rectangle(imageArray, (_x1, _y1), (_x2, _y2), color, cv2.FILLED)

            cv2.rectangle(imageArray, (left, top), (right, bottom), color, thick)
            cv2.putText(imageArray, label, (left, top-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), thick)

        return bboxes, imageArray

    else:
        # return only list of bboxes
        return bboxes


def save_annotations(imageName, bboxes, altNames):
    """
    Files saved with imageName.txt and absolute coordinates in format <label, top_left_x, top_left_y, height, width, confidence>

    Parameters
    ----------------
    imageName: str 
        Name of the image on which the detection was performed

    bboxes: list
        A list of bbox dict like as :
            [{
                "label" : label of the object detected within bbox,
                "top_left_x": ordinate position of the point at the top left of the bbox,
                "top_left_y": abscissa position of the point at the top left the bbox,
                "height": height of the bbox,
                "width": width of the bbox,
                "confidence": confidence
            }]

    altNames: list
        Names readed from namesPath with loadConfiguration()[2]
    """
    file_name = imageName.split(".")[:-1][0] + ".txt"
    with open(file_name, "w") as f:
        for box in bboxes:
            label = altNames.index(box['label'])
            top_left_x = box['top_left_x']
            top_left_y = box['top_left_y']
            height = box['height']
            width = box['width']
            confidence = box['confidence']
            f.write("{} {:.5f} {:.5f} {:.5f} {:.5f} {:.5f}\n".format(label, top_left_x, top_left_y, height, width,
                                                                     confidence))


def main():
    pathFolder = "frame/"
    saveTo = "res/"

    # load configuration of neural network with default files (to modify see loadConfiguration() documentation)
    netMain, metaMain, altNames = loadConfiguration()

    # select only original image in pathFolder
    images = os.listdir(pathFolder)
    imagesToDetect = list()
    for image in images:
        if ((".jpg" in image) or (".jpeg" in image) or (".png" in image)):
            imagesToDetect.append(image)

    for filename in imagesToDetect:
        start = time.time()
        imgPath = pathFolder + filename

        bboxes, img = performDetect(netMain, metaMain, altNames, imgPath, False, True)

        newImageName = pathFolder + saveTo + filename
        cv2.imwrite(newImageName, img)
        save_annotations(newImageName, bboxes,altNames)
        timer = (time.time() - start) * 1000
        print("Image " + imgPath + " predicted! (" + str(timer) + " ms)")

    print("Results saved in "+pathFolder + saveTo)

if __name__ == "__main__":
    main()
