import tensorflow as tf
import numpy as np
import cv2
import os


class CenterNet(object):
	def __init__(self):
		self.input_img_size = 512
		self.class_names = ('aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car',
		                    'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike',
		                    'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor')
		# os.path.abspath(os.path.split(os.path.split(__file__)[0])[0])获取detect_image.py文件的上一级目录路径
		# .replace('\\','/')将路径中的反斜杠\("\\"表示转义字符\)替代为正斜杠/
		self.model_path = os.path.abspath(os.path.split(os.path.split(__file__)[0])[0]).replace('\\',
		                                                                                        '/') + "/model_data/yolo3_centernet_voc.ckpt-70000"
		self.down_ratio = 8.0
		self.class_prob_thresh = 0.44
		self.num_classes = len(self.class_names)

		self.sess = tf.Session()
		self.saver = tf.train.import_meta_graph(self.model_path + ".meta")
		self.saver.restore(self.sess, self.model_path)
		self.input_tensor = self.sess.graph.get_tensor_by_name('inputs:0')
		self.input_training = self.sess.graph.get_tensor_by_name('is_training:0')

		self.output_center = self.sess.graph.get_tensor_by_name('yolo3_centernet/detector/Conv_1/Sigmoid:0')
		self.output_offset = self.sess.graph.get_tensor_by_name('yolo3_centernet/detector/Conv_3/BiasAdd:0')
		self.output_size = self.sess.graph.get_tensor_by_name('yolo3_centernet/detector/Conv_5/BiasAdd:0')

		self.output_center_peak = tf.layers.max_pooling2d(self.output_center, 3, 1, padding='same')
		self.peak_mask = tf.cast(tf.equal(self.output_center, self.output_center_peak), tf.float32)
		self.thresh_mask = tf.cast(tf.greater(self.output_center, self.class_prob_thresh), tf.float32)
		self.obj_mask = self.peak_mask * self.thresh_mask
		self.output_center = self.output_center * self.obj_mask

	def py_nms(self, boxes, scores, max_boxes=50, iou_thresh=0.5):
		"""
		Pure Python NMS baseline.
		Arguments: boxes: shape of [-1, 4], the value of '-1' means that dont know the
						  exact number of boxes
				   scores: shape of [-1,]
				   max_boxes: representing the maximum of boxes to be selected by non_max_suppression
				   iou_thresh: representing iou_threshold for deciding to keep boxes
		"""
		assert boxes.shape[1] == 4 and len(scores.shape) == 1

		x1 = boxes[:, 0]
		y1 = boxes[:, 1]
		x2 = boxes[:, 2]
		y2 = boxes[:, 3]

		areas = (x2 - x1) * (y2 - y1)
		order = scores.argsort()[::-1]

		keep = []
		while order.size > 0:
			i = order[0]
			keep.append(i)
			xx1 = np.maximum(x1[i], x1[order[1:]])
			yy1 = np.maximum(y1[i], y1[order[1:]])
			xx2 = np.minimum(x2[i], x2[order[1:]])
			yy2 = np.minimum(y2[i], y2[order[1:]])

			w = np.maximum(0.0, xx2 - xx1 + 1)
			h = np.maximum(0.0, yy2 - yy1 + 1)
			inter = w * h
			ovr = inter / (areas[i] + areas[order[1:]] - inter)

			inds = np.where(ovr <= iou_thresh)[0]
			order = order[inds + 1]

		return keep[:max_boxes]

	def detect_image(self, img):
		height, width = img.shape[0:2]
		max_size = max(height, width)
		scale = 1.0
		if max_size <= self.input_img_size:
			top = (self.input_img_size - height) // 2
			bottom = self.input_img_size - top - height
			left = (self.input_img_size - width) // 2
			right = self.input_img_size - left - width
		else:
			if height >= width:
				scale = self.input_img_size / height
				height = self.input_img_size
				width = int(width * scale)
				top, bottom = 0, 0
				left = (self.input_img_size - width) // 2
				right = self.input_img_size - left - width
			else:
				scale = self.input_img_size / width
				width = self.input_img_size
				height = int(height * scale)
				top = (self.input_img_size - height) // 2
				bottom = self.input_img_size - top - height
				left, right = 0, 0
		img_resize = cv2.resize(img, (width, height))
		img_resize = cv2.copyMakeBorder(img_resize, top, bottom, left, right, cv2.BORDER_CONSTANT, value=0)
		img_resize = img_resize.astype(np.float32) / 255.0 - 0.5
		input_img = np.expand_dims(img_resize, axis=0)
		center, offset, size = self.sess.run([self.output_center, self.output_offset, self.output_size],
		                                     feed_dict={self.input_tensor: input_img, self.input_training: False})

		all_boxes, all_scores, all_class = [], [], []
		for i in range(self.num_classes):
			cords = np.argwhere(center[:, :, :, i] > 0.0)
			boxes, scores, = [], []
			for cord in cords:
				w = size[cord[0], cord[1], cord[2], 0] * self.down_ratio
				h = size[cord[0], cord[1], cord[2], 1] * self.down_ratio
				offset_x = offset[cord[0], cord[1], cord[2], 0]
				offset_y = offset[cord[0], cord[1], cord[2], 1]
				center_x = (cord[2] + offset_x) * self.down_ratio
				center_y = (cord[1] + offset_y) * self.down_ratio

				x1 = int(center_x - w / 2.)
				y1 = int(center_y - h / 2.)
				x2 = int(center_x + w / 2.)
				y2 = int(center_y + h / 2.)
				score = center[cord[0], cord[1], cord[2], i]

				if top == 0 and bottom == 0:
					x1_src = int((x1 - left) / scale)
					y1_src = int(y1 / scale)
					x2_src = int((x2 - left) / scale)
					y2_src = int(y2 / scale)
				elif left == 0 and right == 0:
					x1_src = int(x1 / scale)
					y1_src = int((y1 - top) / scale)
					x2_src = int(x2 / scale)
					y2_src = int((y2 - top) / scale)
				else:
					x1_src = x1 - left
					y1_src = y1 - top
					x2_src = x2 - left
					y2_src = y2 - top

				boxes.append([x1_src, y1_src, x2_src, y2_src])
				scores.append(score)

			if boxes:
				boxes = np.asarray(boxes)
				scores = np.asarray(scores)
				inds = self.py_nms(boxes, scores, max_boxes=50, iou_thresh=0.5)
				for ind in inds:
					x1_src = boxes[ind][0]
					y1_src = boxes[ind][1]
					x2_src = boxes[ind][2]
					y2_src = boxes[ind][3]

					all_boxes.append([x1_src, y1_src, x2_src, y2_src])
					all_scores.append(scores[ind])
					all_class.append(i)

		return all_boxes, all_scores, all_class

	def show_image_detection_results(self, img, all_boxes, all_scores, all_class):
		# 假设图中最多300个目标,生成300种随机颜色
		colours = np.random.rand(300, 3) * 255
		for i in range(len(all_boxes)):
			x1_src, y1_src, x2_src, y2_src = all_boxes[i][0], all_boxes[i][1], all_boxes[i][2], all_boxes[i][3]
			center_x_src, center_y_src = (x1_src + x2_src) // 2, (y1_src + y2_src) // 2
			class_name = self.class_names[all_class[i]]
			score = all_scores[i]
			txt = class_name + ":" + str(round(score, 2))
			color = (int(colours[all_class[i] % 300, 0]), int(colours[all_class[i] % 300, 1]),
			         int(colours[all_class[i] % 300, 2]))
			cv2.putText(img, txt, (x1_src, y1_src - 2), 0, 0.5, color, 2)
			cv2.circle(img, (center_x_src, center_y_src), 2, (0, 0, 255), 2)
			cv2.rectangle(img, (x1_src, y1_src), (x2_src, y2_src), (0, 255, 0), 1)
		cv2.imshow('result', img)
		cv2.waitKey(0)


if __name__ == "__main__":
	# 检测图片
	img_file = "../image/2.jpg"
	image = cv2.imread(img_file)
	# bgr或rgb格式输入来检测目标结果一致
	center_net = CenterNet()
	boxes_results, scores_results, class_results = center_net.detect_image(image)
	center_net.show_image_detection_results(image, boxes_results, scores_results, class_results)
