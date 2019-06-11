import os
import numpy as np
import tensorflow as tf
import cv2
import time
import warnings
from PIL import Image
from sort.sort import Sort

from SSD_tf.test.detect_and_visualization_image import SSD
from SSD_tf.test import detect_and_visualization_image
from yolo3_tf.yolo_detection import YOLO
from yolo3_centernet_tf.test.detect_image import CenterNet

warnings.filterwarnings('ignore')
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def detect_and_track(file_path, save_path, detection_mode="SSD"):
	# 如果要保存视频,定义视频size
	size = (640, 480)
	save_fps = 24
	# 假设图中最多300个目标,生成300种随机颜色
	colours = np.random.rand(300, 3) * 255
	# 为True保存检测后视频
	write_video_flag = True
	video_capture = cv2.VideoCapture(file_path)
	mot_tracker = Sort()

	if write_video_flag:
		output_video = cv2.VideoWriter(save_path + 'output.mp4', cv2.VideoWriter_fourcc('m', 'p', '4', 'v'), save_fps,
		                               size)
		object_list_file = open(save_path + 'detection.txt', 'w')
		frame_index = -1

	if detection_mode == "SSD":
		ssd = SSD()
	elif detection_mode == "YOLO3":
		yolo = YOLO()
	elif detection_mode == "CENTERNET":
		centernet = CenterNet()

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
		if detection_mode == "SSD":
			image = frame
			classes, scores, bboxes = ssd.process_image(image)
			# 获得检测到的每个目标的左上角和右下角坐标
			result = np.array(detect_and_visualization_image.plt_bboxes(image, classes, scores, bboxes))
			rbboxes = []
			for object in result:
				rbboxes.append([object[0], object[1], object[2], object[3]])

		elif detection_mode == "YOLO3":
			image = Image.fromarray(frame[..., ::-1])
			# bboxes为[x,y,w,h]形式坐标,score为目标分数,rbboxes为左上角+右下角坐标形式
			bboxes, scores, rbboxes = yolo.detect_image(image)
			result = []
			for box, score in zip(rbboxes, scores):
				# 使用目标左上角和右下角坐标用于追踪,注意图像的左上角为原点,x轴向右为正,y轴向下为正
				ymin, xmin, ymax, xmax = box
				xmin, ymin = max(0, np.floor(xmin + 0.5).astype('int32')), max(0, np.floor(ymin + 0.5).astype('int32'))
				xmax, ymax = min(image.size[0], np.floor(xmax + 0.5).astype('int32')), min(image.size[1],
				                                                                           np.floor(ymax + 0.5).astype(
					                                                                           'int32'))
				result.append([xmin, ymin, xmax, ymax, score])
			result = np.array(result)
		elif detection_mode == "CENTERNET":
			image = frame
			# 这里的boxes_results是左上角和右下角坐标
			rbboxes, scores, classes = centernet.detect_image(image)
			result = []
			for i in range(len(rbboxes)):
				result.append([rbboxes[i][0], rbboxes[i][1], rbboxes[i][2], rbboxes[i][3],
				               scores[i]])
			result = np.array(result)

		if len(result) != 0:
			# 调用目标检测结果
			det = result[:, 0:5]
		else:
			det = result
		# 调用sort进行数据关联追踪
		trackers = mot_tracker.update(det)
		for object in trackers:
			xmin, ymin, xmax, ymax, index = int(object[0]), int(object[1]), int(object[2]), int(object[3]), int(
				object[4])
			color = (int(colours[index % 300, 0]), int(colours[index % 300, 1]), int(colours[index % 300, 2]))
			cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), color, 2)
			cv2.putText(frame, str(index), (xmin, ymin), 0, 5e-3 * 200, color, 2)
			if index in appear.keys():
				appear[index] += 1
			else:
				number += 1
				appear[index] = 1

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
			# 写入每一帧探测到的目标位置,即目标狂的左上角和右下角坐标
			if len(rbboxes) != 0:
				for i in range(0, len(rbboxes)):
					object_list_file.write(
						str(rbboxes[i][0]) + ' ' + str(rbboxes[i][1]) + ' ' + str(rbboxes[i][2]) + ' ' + str(
							rbboxes[i][3]) + ' ')
			object_list_file.write('\n')

		# 按q可退出
		if cv2.waitKey(1) & 0xFF == ord('q'):
			break

	video_capture.release()
	if write_video_flag:
		output_video.release()
		object_list_file.close()
	cv2.destroyAllWindows()


if __name__ == "__main__":
	detect_file = "../MOT16/video/train/MOT16-11.mp4"
	detected_results_save_path = "../results/"
	# detection_mode="SSD"/"YOLO3"/"CENTERNET"
	detect_and_track(detect_file, detected_results_save_path, detection_mode="YOLO3")
