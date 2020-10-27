# Compile Darknet

### **1. Prerequisites**

- OS == Ubuntu 18.04

- GPU with CC >= 3.0: [Check it](https://en.wikipedia.org/wiki/CUDA#GPUs_supported)

- CMAKE >= 3.13

    ```console
    sudo apt install gcc g++ make libssl-dev
    wget https://github.com/Kitware/CMake/releases/download/v3.17.2/cmake-3.17.2.tar.gz
    tar -zxvf cmake-3.17.2.tar.gz
    cd cmake-3.17.2
    ./bootstrap
    make
    sudo make install
    cmake --version (to verify installation)
    ```

- OPENCV >= 2.4
 ([online tutorial](https://linuxize.com/post/how-to-install-opencv-on-ubuntu-18-04/))

    ```console
    sudo apt update
    sudo apt install python3-opencv
    sudo apt install build-essential cmake git pkg-config libgtk-3-dev \
        libavcodec-dev libavformat-dev libswscale-dev libv4l-dev \
        libxvidcore-dev libx264-dev libjpeg-dev libpng-dev libtiff-dev \
        gfortran openexr libatlas-base-dev python3-dev python3-numpy \
        libtbb2 libtbb-dev libdc1394-22-dev
    mkdir ~/opencv_build && cd ~/opencv_build
    git clone https://github.com/opencv/opencv.git
    git clone https://github.com/opencv/opencv_contrib.git
    cd ~/opencv_build/opencv
    mkdir build && cd build
    cmake -D CMAKE_BUILD_TYPE=RELEASE \
        -D CMAKE_INSTALL_PREFIX=/usr/local \
        -D INSTALL_C_EXAMPLES=ON \
        -D INSTALL_PYTHON_EXAMPLES=ON \
        -D OPENCV_GENERATE_PKGCONFIG=ON \
        -D OPENCV_EXTRA_MODULES_PATH=~/opencv_build/opencv_contrib/modules \
        -D BUILD_EXAMPLES=ON ..
    make -j8
    sudo make install
    pkg-config --modversion opencv4
    ```

    (you can try opencv running python3:
    ```python
    import cv2
    print(cv2.__version__)
    ```
    and you can show path of install with "locate opencv")
    
#### **1.1 Install Nvidia Driver, CUDA & CUDNN**

- NVIDIA DRIVER (CMD Terminal)

    ```console
    sudo apt update & sudo apt upgrade
    ubuntu-drivers devices (to check which nvidia driver is recommended, in my case 440)
    sudo apt install nvidia-driver-440
    sudo reboot
    nvidia-smi
    ```

- NVIDIA DRIVER (Graphical alternative)

> 1) Open the ubuntu 'Software updates' tool
> 2) Click on settings
> 3) Select the menu item 'Additional drivers'
> 4) Select nvidia-driver-440
> 5) Click on 'Apply Changes'
> 6) sudo reboot
> 7) nvidia-smi

- CUDA 10.0 ([online tutorial](https://medium.com/repro-repo/install-cuda-10-1-and-cudnn-7-5-0-for-pytorch-on-ubuntu-18-04-lts-9b6124c44cc))

    > Download the local runfile for cuda 10.0 from [CUDA Toolkit site](https://developer.nvidia.com/cuda-10.0-download-archive?target_os=Linux&target_arch=x86_64&target_distro=Ubuntu&target_version=1804&target_type=runfilelocal)
    ```console
    wget https://developer.nvidia.com/compute/cuda/10.0/Prod/local_installers/cuda_10.0.130_410.48_linux
    sudo sh cuda_10.0.130_410.48_linux.run (Answer yes to everything except the request to install nvidia drivers)
    reboot
    nano ~/.bashrc
    (write at the end:
    #export CUDA
    export PATH=/usr/local/cuda-10.0/bin:/usr/local/cuda-10.0/NsightCompute-1.0${PATH:+:${PATH}}${PATH:+:${PATH}}
    export LD_LIBRARY_PATH=/usr/local/cuda-10.0/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}
    export CUDADIR=/usr/local/cuda-10.0
    )
    source ~/.bashrc
    ```

- CUDNN 7.6.6

    >Login to [Cudnn site](https://developer.nvidia.com/rdp/cudnn-download) and click on download cudnn for cuda 10.0
    (Download all 3 .deb files: the runtime library, the developer library, and the code samples library for Ubuntu 18.04)
    
    ```console
    sudo dpkg -i libcudnn7_7.6.5.32-1+cuda10.0_amd64.deb 
    sudo dpkg -i libcudnn7-dev_7.6.5.32-1+cuda10.0_amd64.deb 
    sudo dpkg -i libcudnn7-doc_7.6.5.32-1+cuda10.0_amd64.deb
    sudo apt-get install libcupti-dev
    nano ~/.bashrc
    (write at end:
    export LD_LIBRARY_PATH=/usr/local/cuda/extras/CUPTI/lib64:$LD_LIBRARY_PATH
    )
    ```
### **2. Compile darknet**

- DARKNET ([git repository](https://github.com/AlexeyAB/darknet))

    ```console
    git clone https://github.com/AlexeyAB/darknet.git
    cd darknet
    nano Makefile (Change the following parameters, se sono necessari:
        - GPU=1 to build with CUDA to accelerate by using GPU (CUDA should be in /usr/local/cuda)
        - CUDNN=1 to build with cuDNN v5-v7 to accelerate training by using GPU (cuDNN should be in /usr/local/cudnn)
        - OPENCV=1 to build with OpenCV 4.x/3.x/2.4.x - allows to detect on video files and video streams from network cameras or web-cams
        - LIBSO=1 to build a library darknet.so and binary runable file uselib that uses this library
    )
    make
    ```
    (if give error `cannot find -lcuda`, search "libcuda.so" and copy it into "/usr/local/cuda-10.0/lib64")

### **3. Run darknet**
- Try to run:
    ```console
    wget https://pjreddie.com/media/files/yolov3-tiny.weights
    ./darknet detector test cfg/coco.data cfg/yolov3-tiny.cfg yolov3-tiny.weights -ext_output data/dog.jpg
    ```
    
    if the execution gives the error `error while loading shared libraries: libopencv_highgui.so.4.3: cannot open shared object file: No such file or directory`, run the following commands:
    ```console
    sudo /bin/bash -c 'echo "/usr/local/lib" > /etc/ld.so.conf.d/opencv.conf'
    sudo ldconfig
    ```
##Author 

- **Giuseppe Murro** - _Writing the tutorial to install Darknet on Ubuntu_ - [GitHub](https://github.com/gmurro) 

## License

This project is licensed under the MIT License - see the [LICENSE](https://github.com/gmurro/darknet/blob/master/LICENSE.md) file for details.
