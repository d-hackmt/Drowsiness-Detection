# Drowsiness and Yawn detection with voice alert using Dlib

![Microsoft Game DVR - Frame - VLC media player 3_13_2024 8_18_36 PM](https://github.com/d-hackmt/Drowsiness-Detection/assets/113240252/a597cd1b-c067-4bb6-888d-bc44ab45bec5)
![Microsoft Game DVR - Frame - VLC media player 3_13_2024 8_18_51 PM](https://github.com/d-hackmt/Drowsiness-Detection/assets/113240252/5502b309-f286-4033-89c6-72b989df9e6a)


Download dlib 68 landmarks detector dat file . 

Simple code in python to detect Drowsiness and Yawn and alert the user using Dlib.

## Dependencies

1. Python 3
2. opencv
3. dlib
4. imutils
5. scipy
6. numpy
7. argparse

## Run 

```
Python3 drowsiness_yawn.py -- webcam 0		//For external webcam, use the webcam number accordingly
```

## Setups

Change the threshold values according to your need
```
EYE_AR_THRESH = 0.3
EYE_AR_CONSEC_FRAMES = 30
YAWN_THRESH = 10`	//change this according to the distance from the camera
```





