# Deep learning using Tensorflow Lite on Raspberry Pi

![alt text](https://github.com/Zaheer505/Deep-learning-using-Tensorflow-Lite-on-Raspberry-Pi/blob/main/images/thumbnail.png)

<details open="open">
  <summary>Table of Contents</summary>
  <ol>
    <li><a href="#About-this-Repository">About This Repository</a></li>
    <li><a href="#Features">Features</a></li>
    <li><a href="#Installations">Installations</a></li>
    <li><a href="#Using-this-Repository">Using this Repository</a></li>
    <li><a href="#Pre-Course-Requirments">Pre-Course Requirments</a></li>
    <li><a href="#Link-to-the-Course">Link to the Course</a></li>
    <li><a href="#Instructors">Instructors</a></li>
    <li><a href="#License">License</a></li>
  </ol>
</details>


## About this Repository
This course is focused on Embedded Deep learning in Python . Raspberry PI 4 is utilized as a main hardware and we will be building practical projects with custom data .

- We will start with trigonometric functions approximation . In which we will generate random data and produce a model for Sin function approximation

- Next is a calculator that takes images as input and builds up an equation and produces a result .This Computer vision based project is going to be using convolution network architecture for Categorical classification

- Another amazing project is focused on convolution network but the data is custom voice recordings . We will involve a little bit of electronics to show the output by controlling our multiple LEDs using own voice .

- Unique learning point in this course is Post Quantization applied on Tensor flow models trained on Google Colab . Reducing size of models to 3 times and increasing inferencing speed up to 0.03 sec per input .

Note: This repo contains step by step approach to teach different things to students of our course. You may find some raw data / codes which are meant for learning purposes of students.

---
## Features
- **Non Linear Trignometric Approximation**
    - ![alt text](https://github.com/Zaheer505/Deep-learning-using-Tensorflow-Lite-on-Raspberry-Pi/blob/main/images/sign_function.gif)
- **Real Time Number Detection**
    - ![alt text](https://github.com/Zaheer505/Deep-learning-using-Tensorflow-Lite-on-Raspberry-Pi/blob/main/images/vc_extraction.gif)
- **Visual Calculator Equation Solving**
    - ![alt text](https://github.com/Zaheer505/Deep-learning-using-Tensorflow-Lite-on-Raspberry-Pi/blob/main/images/vc_equation.gif)
- **Voice Controlled LEDs**
    - ![alt text](https://github.com/Zaheer505/Deep-learning-using-Tensorflow-Lite-on-Raspberry-Pi/blob/main/images/voice_control.gif)
---
## Installations
- Laptop/PC Installations
    - Rpi-Imager for installing RPI OS on SD CARD
        ```
        sudo apt install rpi-imager
        ```
    - Tensorflow
        ```
        pip install tensorflow
        ```

- Raspberry PI 4 installations
    - Tensorflow Lite Interpreter
        ```
        python3 -m pip install tflite-runtime
        ```
    - Install tightvnc server
        ```
        sudo apt-get install tightvncserver
        ```
- Common Installations
    - OPENCV
        ```
        pip3 install opencv-python
        sudo apt-get install libcblas-dev
        sudo apt-get install libhdf5-dev
        sudo apt-get install libhdf5-serial-dev
        sudo apt-get install libatlas-base-dev
        sudo apt-get install libjasper-dev
        sudo apt-get install libqtgui4
        sudo apt-get install libqt4-test
        sudo apt-get install libatlas-base-dev
        ```
    - Upgrade Numpy
        ```
        pip install -U numpy
        ```
    - Audio processing Dependencies
        ```
        pip install sounddevice
        sudo apt-get install libportaudio2
        pip install scipy
        ```
----
## Using Repository
- Obtain the code using Git
    ```
    git clone https://github.com/Zaheer505/Tiny-ML --single-branch development
    ```
- SSH into your RPI
    ```
    ssh pi@<IP_of_RPI>
    ```
- Turn on the Tightvnc Server to enable screen sharing
    ```
    tightvncserver :1
    ```
- Access RPI through VNC-Viwer on PC
---
## Pre-Course Requirments
- PC   : Ubuntu 22.04
- RPI4 : RPI Full OS
    - SD-CARD 16GB
    - RPI Camera V2
    - Power Bank with Type C cable
    - 3D printed Parts for Camera Holding
    - Fan on RPI for better thermals

## Link to the Course
If you want to take video lecture explaination of how this repository is built . You can check out this course .
**[[Discounted Link]](https://www.udemy.com/course/deep-learning-using-tensorflow-lite-on-raspberry-pi/?couponCode=LAUNCH)**

----

## Instructors
Muhammad Luqman - [Profile Link](https://www.linkedin.com/in/muhammad-luqman-9b227a11b/)
Zaheer Ahmed - [Profile Link](https://www.linkedin.com/in/zaheer-ahmed505/)

----
## License

Distributed under the MIT License. See `LICENSE` for more information.
