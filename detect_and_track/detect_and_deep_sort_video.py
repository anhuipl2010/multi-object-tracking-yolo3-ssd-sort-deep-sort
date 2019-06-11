#! /usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division, print_function, absolute_import

from timeit import time
import os
import warnings
import cv2
import numpy as np
from PIL import Image
from yolo3_tf.yolo_detection import YOLO
from SSD_tf.test.detect_and_visualization_image import SSD

from deep_sort import preprocessing
from deep_sort import nn_matching
from deep_sort.detection import Detection
from deep_sort.tracker import Tracker
from deep_sort.tools import generate_detections as gdet

warnings.filterwarnings('ignore')
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def detect_and_track(file_path, save_path, detection_mode="YOLO3"):
	# Definition of the parameters
	max_cosine_distance = 0.3
	nn_budget = None
	nms_max_overlap = 1.0
	# 如果要保存视频,定义视频size
	size = (640, 480)
	save_fps = 24

	# use deep_sort tracker
	model_filename = '../deep_sort/model_data/resources/networks/mars-small128.pb'
	encoder = gdet.create_box_encoder(model_filename, batch_size=1)
	metric = nn_matching.NearestNeighborDistanceMetric("cosine", max_cosine_distance, nn_budget)
	tracker = Tracker(metric)

	write_video_flag = True
	# 假设图中最多300个目标,生成300种随机颜色
	colours = np.random.rand(300, 3) * 255
	video_capture = cv2.VideoCapture(file_path)

	if write_video_flag:
		output_video = cv2.VideoWriter(save_path + 'output.mp4', cv2.VideoWriter_fourcc('m', 'p', '4', 'v'), save_fps,
		                               size)
		object_list_file = open(save_path + 'detection.txt', 'w')
		frame_index = -1

	if detection_mode == "YOLO3":
		yolo = YOLO()
	elif detection_mode == "SSD":
		ssd = SSD()

	# appear记录每个出现过的目标存在的帧数量,number记录所有出现过的目标(不重复)
	appear = {}
	number = 0

	while True:
		ret, frame = video_capture.read()
		if ret is not True:
			break
		frame = cv2.resize(frame, size)
		# 记录每一帧开始处理的时间
		start_time = time.time()
		if detection_mode == "YOLO3":
			image = Image.fromarray(frame[..., ::-1])
			# boxes为[x,y,w,h]形式坐标,detect_scores为目标分数,origin_boxes为左上角+右下角坐标形式
			boxes, detect_scores, origin_boxes = yolo.detect_image(image)
		elif detection_mode == "SSD":
			rclasses, rscores, rbboxes = ssd.process_image(frame)
			height, width = frame.shape[0], frame.shape[1]
			boxes = []
			# 遍历一帧图片中每个目标的(对应classes)
			for i in range(rclasses.shape[0]):
				# rbboxes原始形式为0-1范围的左上角和右下角坐标
				xmin, ymin = int(rbboxes[i, 1] * width), int(rbboxes[i, 0] * height)
				xmax, ymax = int(rbboxes[i, 3] * width), int(rbboxes[i, 2] * height)
				# 转换为x,y,w,h形式的坐标
				x, y, w, h = int(xmin), int(ymin), int(xmax - xmin), int(ymax - ymin)
				if x < 0:
					w = w + x
					x = 0
				if y < 0:
					h = h + y
					y = 0
				boxes.append([x, y, w, h])
			boxes = np.array(boxes)

		features = encoder(frame, boxes)
		# score to 1.0 here
		detections = [Detection(bbox, 1.0, feature) for bbox, feature in zip(boxes, features)]
		# 非极大值抑制
		boxes = np.array([d.tlwh for d in detections])
		scores = np.array([d.confidence for d in detections])
		indices = preprocessing.non_max_suppression(boxes, nms_max_overlap, scores)
		detections = [detections[i] for i in indices]
		# 追踪器预测和更新
		tracker.predict()
		tracker.update(detections)

		for track in tracker.tracks:
			if not track.is_confirmed() or track.time_since_update > 1:
				continue
			bbox = track.to_tlbr()
			color = (int(colours[track.track_id % 300, 0]), int(colours[track.track_id % 300, 1]),
			         int(colours[track.track_id % 300, 2]))
			# (int(bbox[0]), int(bbox[1]))为左上角坐标,(int(bbox[2]), int(bbox[3]))为右下角坐标
			cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), color, 2)
			cv2.putText(frame, str(track.track_id), (int(bbox[0]), int(bbox[1])), 0, 5e-3 * 200, color, 2)
			if track.track_id in appear.keys():
				appear[track.track_id] += 1
			else:
				number += 1
				appear[track.track_id] = 1

		show_fps = 1. / (time.time() - start_time)
		cv2.putText(frame, text="FPS: " + str(int(show_fps)), org=(3, 15), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
		            fontScale=0.50, color=(0, 255, 0), thickness=2)
		cv2.putText(frame, text="number: " + str(number), org=(3, 30), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
		            fontScale=0.50, color=(0, 255, 0), thickness=2)
		cv2.imshow('result', frame)

		if write_video_flag:
			# 保存视频每一帧
			output_video.write(frame)
			# 更新视频帧编号
			frame_index = frame_index + 1
			# detection.txt写入下一帧的编号
			object_list_file.write(str(frame_index) + ' ')
			# 写入每一帧探测到的目标的框四个点坐标
			if len(boxes) != 0:
				for i in range(0, len(boxes)):
					object_list_file.write(
						str(boxes[i][0]) + ' ' + str(boxes[i][1]) + ' ' + str(boxes[i][2]) + ' ' + str(
							boxes[i][3]) + ' ')
			object_list_file.write('\n')

		# 按q可退出
		if cv2.waitKey(1) & 0xFF == ord('q'):
			break

	video_capture.release()
	if write_video_flag:
		output_video.release()
		object_list_file.close()
	cv2.destroyAllWindows()


if __name__ == '__main__':
	detect_file = "../MOT16/video/train/MOT16-11.mp4"
	detected_results_save_path = "../results/"
	detect_and_track(detect_file, detected_results_save_path, detection_mode="SSD")
