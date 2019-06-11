import cv2
import random
import os
import sys
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.cm as mpcm

sys.path.append('../')
import numpy as np
import tensorflow as tf
from SSD_tf.nets import ssd_vgg_300, np_methods
from SSD_tf.preprocessing import ssd_vgg_preprocessing
slim = tf.contrib.slim

# added 20180516#####
def num2class(n):
	import SSD_tf.datasets.pascalvoc_2007 as pas
	x = pas.pascalvoc_common.VOC_LABELS.items()
	for name, item in x:
		if n in item:
			# print(name)
			return name


# =========================================================================== #
# Some colormaps.
# =========================================================================== #
def colors_subselect(colors, num_classes=21):
	dt = len(colors) // num_classes
	sub_colors = []
	for i in range(num_classes):
		color = colors[i * dt]
		if isinstance(color[0], float):
			sub_colors.append([int(c * 255) for c in color])
		else:
			sub_colors.append([c for c in color])
	return sub_colors


colors_plasma = colors_subselect(mpcm.plasma.colors, num_classes=21)
colors_tableau = [(255, 255, 255), (31, 119, 180), (174, 199, 232), (255, 127, 14), (255, 187, 120),
                  (44, 160, 44), (152, 223, 138), (214, 39, 40), (255, 152, 150),
                  (148, 103, 189), (197, 176, 213), (140, 86, 75), (196, 156, 148),
                  (227, 119, 194), (247, 182, 210), (127, 127, 127), (199, 199, 199),
                  (188, 189, 34), (219, 219, 141), (23, 190, 207), (158, 218, 229)]


# =========================================================================== #
# OpenCV drawing.
# =========================================================================== #
def draw_lines(img, lines, color=[255, 0, 0], thickness=2):
	"""Draw a collection of lines on an image.
	"""
	for line in lines:
		for x1, y1, x2, y2 in line:
			cv2.line(img, (x1, y1), (x2, y2), color, thickness)


def draw_rectangle(img, p1, p2, color=[255, 0, 0], thickness=2):
	cv2.rectangle(img, p1[::-1], p2[::-1], color, thickness)


def draw_bbox(img, bbox, shape, label, color=[255, 0, 0], thickness=2):
	p1 = (int(bbox[0] * shape[0]), int(bbox[1] * shape[1]))
	p2 = (int(bbox[2] * shape[0]), int(bbox[3] * shape[1]))
	cv2.rectangle(img, p1[::-1], p2[::-1], color, thickness)
	p1 = (p1[0] + 15, p1[1])
	cv2.putText(img, str(label), p1[::-1], cv2.FONT_HERSHEY_DUPLEX, 0.5, color, 1)


# 绘制每一帧图像的检测框/类别框
def bboxes_draw_on_img(img, classes, scores, bboxes, colors, thickness=2):
	shape = img.shape
	for i in range(bboxes.shape[0]):
		bbox = bboxes[i]
		color = colors[classes[i]]
		# Draw bounding box...
		p1 = (int(bbox[0] * shape[0]), int(bbox[1] * shape[1]))
		p2 = (int(bbox[2] * shape[0]), int(bbox[3] * shape[1]))
		cv2.rectangle(img, p1[::-1], p2[::-1], color, thickness)
		# Draw text...
		s = '%s/%.3f' % (num2class(classes[i]), scores[i])
		p1 = (p1[0] - 5, p1[1])
		cv2.putText(img, s, p1[::-1], cv2.FONT_HERSHEY_DUPLEX, 0.4, color, 1)


def plt_bboxes(img, classes, scores, bboxes, figsize=(10, 10), linewidth=1.5):
	"""Visualize bounding boxes. Largely inspired by SSD-MXNET!
	"""
	fig = plt.figure(figsize=figsize)
	plt.imshow(img)
	height = img.shape[0]
	width = img.shape[1]
	colors = dict()
	# print(classes.shape)
	# classes.shape  (9,)
	# 创建用于传参的result
	result = []
	for i in range(classes.shape[0]):  # 遍历一帧图片中每个目标的(对应classes)
		cls_id = int(classes[i])
		if cls_id >= 0:
			score = scores[i]
			if cls_id not in colors:
				colors[cls_id] = (random.random(), random.random(), random.random())
				# 对应目标类别的颜色随机选取
			ymin = int(bboxes[i, 0] * height)  # 一帧中对应目标的矩形框参数
			xmin = int(bboxes[i, 1] * width)
			ymax = int(bboxes[i, 2] * height)
			xmax = int(bboxes[i, 3] * width)

		result.append([xmin, ymin, xmax, ymax, score])
	return np.array(result)


class SSD(object):
	def __init__(self):
		# TensorFlow session: grow memory when needed. TF, DO NOT USE ALL MY GPU MEMORY!!!
		self.gpu_options = tf.GPUOptions(allow_growth=True)
		self.config = tf.ConfigProto(log_device_placement=False, gpu_options=self.gpu_options)
		self.isess = tf.InteractiveSession(config=self.config)

		self.ckpt_filename = '../SSD_tf/model_data/ssd_300_vgg.ckpt'
		self.net_shape = (300, 300)
		self.data_format = 'NHWC'
		self.img_input = tf.placeholder(tf.uint8, shape=(None, None, 3))
		self.image_pre, self.labels_pre, self.bboxes_pre, self.bbox_img = ssd_vgg_preprocessing.preprocess_for_eval(
			self.img_input, None, None, self.net_shape, self.data_format,
			resize=ssd_vgg_preprocessing.Resize.WARP_RESIZE)
		self.image_4d = tf.expand_dims(self.image_pre, 0)

		self.reuse = True if 'ssd_net' in locals() else None
		self.ssd_net = ssd_vgg_300.SSDNet()
		with slim.arg_scope(self.ssd_net.arg_scope(data_format=self.data_format)):
			self.predictions, self.localisations, _, _ = self.ssd_net.net(self.image_4d, is_training=False,
			                                                              reuse=self.reuse)

		self.isess.run(tf.global_variables_initializer())
		self.saver = tf.train.Saver()
		self.saver.restore(self.isess, self.ckpt_filename)

		# SSD default anchor boxes.
		self.ssd_anchors = self.ssd_net.anchors(self.net_shape)

	def process_image(self, img, select_threshold=0.5, nms_threshold=.45, net_shape=(300, 300)):
		# 分类预测rpredictions,坐标预测rlocalisations
		rimg, rpredictions, rlocalisations, rbbox_img = self.isess.run(
			[self.image_4d, self.predictions, self.localisations, self.bbox_img],
			feed_dict={self.img_input: img})
		# Get classes and bboxes from the net outputs.
		rclasses, rscores, rbboxes = np_methods.ssd_bboxes_select(
			rpredictions, rlocalisations, self.ssd_anchors,
			select_threshold=select_threshold, img_shape=net_shape, num_classes=21, decode=True)

		rbboxes = np_methods.bboxes_clip(rbbox_img, rbboxes)
		rclasses, rscores, rbboxes = np_methods.bboxes_sort(rclasses, rscores, rbboxes, top_k=400)
		rclasses, rscores, rbboxes = np_methods.bboxes_nms(rclasses, rscores, rbboxes, nms_threshold=nms_threshold)
		# Resize bboxes to original image shape. Note: useless for Resize.WARP!
		rbboxes = np_methods.bboxes_resize(rbbox_img, rbboxes)
		return rclasses, rscores, rbboxes
