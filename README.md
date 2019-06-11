

# multi-object-tracking:yolo3/ssd+sort/deep_sort

A multi-object-tracking system which use tracking-by-detection method:yolo3/ssd(detection model)+sort/deep sort(tracking model)

# Introduction

use:
[keras-yolo3](https://github.com/qqwweee/keras-yolo3)
[deep_sort](https://github.com/nwojke/deep_sort)
[deep_sort_yolov3](ht  tps://github.com/zgcr/deep_sort_yolov3)

# Quick Start

1. Download YOLOv3 weights from [YOLO website](https://pjreddie.com/media/files/yolov3.weights).
2. get yolov3.weights into ./yolo3/model_data
2. Convert the Darknet YOLO model to a Keras model.
```
wget https://pjreddie.com/media/files/yolov3.weights
cd ./yolo3/
python convert.py yolov3.cfg yolov3.weights ./model_data/yolo.h5
```
4. download pre-generated detections and the CNN checkpoint file from[here](https://drive.google.com/open?id=18fKzfqnqhqW3s9zwsCbnVJ5XF2JFeqMp).
5. get resources file(contains two files named:detections and networks) into ./deep_sort/model_data/
6. Then generate file:mars-small128.pb
```
cd ./deep_sort/tools/
python freeze_model.py
```
7. download MOT16 datasets from[here](https://motchallenge.net/data/MOT16.zip)
8. uzip MOT16.zip,put MOT16 file into ./
9. Convert MOT16 image to video.
```

```