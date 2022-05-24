# Tiny-ML

## Installations
- Raspian OS
    - RPI-IMAGER
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
- Tensorflow Lite Interpreter
```
python3 -m pip install tflite-runtime
```
- Install tightvnc server
```
sudo apt-get install tightvncserver
```
## Using Repository
- SSH into your RPI
```
ssh rpi@<IP_of_RPI>
```
- Turn on the Tightvnc Server to enable screen sharing
```
tightvncserver
```
- Access RPI through vnc
