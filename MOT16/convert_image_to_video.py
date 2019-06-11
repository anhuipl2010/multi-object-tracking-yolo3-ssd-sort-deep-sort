"""
将测试集和训练集中的连续帧图片合成为原始视频
"""
import os
import cv2

# videowriter可以保存的格式:
# 'I', '4', '2', '0'->.avi;'m', 'p', '4', 'v'->.mp4;

test_picture_path = './test/'
train_picture_path = './train/'
test_video_save_path = './video/test/'
train_video_save_path = './video/train/'

# 每秒24帧
fps = 24
# 视频尺寸,注意图片尺寸一定要用resize转换成和视频尺寸一致再写入视频,否则生成的视频不能正常打开
size = (640, 480)

# 合成测试集视频
test_video_list = os.listdir(test_picture_path)
for per_video in test_video_list:
	per_video_picture_path = test_picture_path + per_video + "/img1/"
	save_path = test_video_save_path + per_video + ".mp4"
	video = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc('m', 'p', '4', 'v'), fps, size)
	picture_list = os.listdir(per_video_picture_path)
	for item in picture_list:
		if item.endswith('.jpg'):
			# 找到目录中所有后缀名为.jpg的文件
			item = per_video_picture_path + item
			img = cv2.imread(item)
			img = cv2.resize(img, size)
			video.write(img)

	print("complete:{}".format(save_path))
	video.release()
	cv2.destroyAllWindows()

# 合成训练集视频
train_video_list = os.listdir(train_picture_path)
for per_video in train_video_list:
	per_video_picture_path = train_picture_path + per_video + "/img1/"
	save_path = train_video_save_path + per_video + ".mp4"
	video = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc('m', 'p', '4', 'v'), fps, size)
	picture_list = os.listdir(per_video_picture_path)
	for item in picture_list:
		if item.endswith('.jpg'):
			# 找到目录中所有后缀名为.jpg的文件
			item = per_video_picture_path + item
			img = cv2.imread(item)
			img = cv2.resize(img, size)
			video.write(img)

	print("complete:{}".format(save_path))
	video.release()
	cv2.destroyAllWindows()
