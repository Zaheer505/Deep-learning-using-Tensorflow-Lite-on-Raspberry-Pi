# Tiny-ML
This Branch of repository is build live during course creation and commits are source students to reference the development .

### Projects
- **Project 1 : Sin Function Approximation**
    - Notebooks : Contains Python Notebooks for model training on Colab.
- **Project 2 : Visual Calculator**
    - **Notebooks** : Contains Python Notebooks for model .training on Colab
    - **Scripts** : Step Wise python Scripts for Model Testing and Implementation.
    - **Data** : Custom Data for printing and Video Recording.
- **Project 3 : Audio Led Control**
    - **Notebooks** : Contains Python Notebooks for model training on Colab.
    - **Scripts** : Step Wise python Scripts for Model Testing and Implementation.
    - **Data** : Pre Trained model for inferencing.
### System Requirments
- PC   : Ubuntu 22.04
- RPI4 : RPI Full OS
    - SD-CARD 16GB
    - RPI Camera V2
    - Power Bank with Type C cable
    - 3D printed Parts for Camera Holding
    - Fan on RPI for better thermals

### Installations
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
