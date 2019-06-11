

# multi-object-tracking:yolo3/ssd+sort/deep_sort

A multi-object-tracking system which use tracking-by-detection method:yolo3/ssd(detection model)+sort/deep sort(tracking model)

# Introduction

The modification of this repository is based on the following repositories:
[keras-yolo3](https://github.com/qqwweee/keras-yolo3)

[SSD-Tensorflow](https://github.com/balancap/SSD-Tensorflow)

[sort](https://github.com/abewley/sort)

[deep_sort](https://github.com/nwojke/deep_sort)
[deep_sort_yolov3](ht  tps://github.com/zgcr/deep_sort_yolov3)

[centernet_tensorflow_wilderface_voc](https://github.com/xggIoU/centernet_tensorflow_wilderface_voc)

# preparation

1. Download this repository.
2. download YOLOv3 weights from [YOLO website ](https://pjreddie.com/media/files/yolov3.weights)and put them in `Repository_ROOT/yolo3_tf/`.

3. Convert the Darknet YOLO model to a Keras model.

```
python convert.py yolov3.cfg yolov3.weights ./model_data/yolo.h5
```
4. download deep_sort pretrained weights from [here](https://drive.google.com/open?id=18fKzfqnqhqW3s9zwsCbnVJ5XF2JFeqMp) and put them in  `Repository_ROOT/deep_sort/model_data/`.
5. Then generate file:mars-small128.pb.
```
cd Repository_ROOT/deep_sort/deep_sort/tools/
python freeze_model.py
```
6. download SSD weights from [here](https://drive.google.com/file/d/0B0qPCUZ-3YwWUXh4UHJrd1RDM3c/view),uzip it and put them in `Repository_ROOT/SSD_tf/model_data/`.
7. download file:yolo3_centernet_voc from [here](https://pan.baidu.com/s/1VrHv5U1wF1UP_r7JICbeZA#list/path=%2F) ,password:qqwx.Then  put them in  `Repository_ROOT/yolo3_centernet_tf/model_data/`.
8. download MOT16 datasets from [here](https://motchallenge.net/data/MOT16.zip) ,uzip MOT16.zip，and put file: MOT16 in `Repository_ROOT/`

8. Convert MOT16 image to video.

```
python Repository_ROOT/MOT16/convert_image_to_video.py
```

# test

run SSD/YOLO3+sort:

```
python detect_and_sort_video.py
# just modify the parameter detection_mode="SSD"or"YOLO3" to choose different detection model
```

run SSD/YOLO3+deep_sort:

```
python detect_and_deep_sort_video.py
# just modify the parameter detection_mode="SSD"or"YOLO3" to choose different detection model
```

