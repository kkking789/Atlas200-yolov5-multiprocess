import cv2
import time
import os
import sys
import importlib
import time
import shutil
import argparse
import yaml
from tqdm import tqdm
BASE_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
print(BASE_PATH)
sys.path.append(BASE_PATH)
importlib.reload(sys)
from YoloV5Detector.V5Detector import Detector

def inference_single_image(weights_path, thresh, src_img, dst_img, yaml, cls, colors=None, device ='cpu'):
	#实例化V5检测类，可指定检测阈值，输入图片，输出图片，需要检测类别，画框颜色，以及使用的gpuid
	det = Detector(weights_path, yaml, colors=colors, device=device)
	t1 = time.time()
	img = cv2.imread(src_img)
	#模型推理
	img_res, det_res = det.detect(img, cls, thresh)
	t2 = (time.time() - t1) * 1000
	print("inference time:{} ms".format(t2))
	#绘制模型检测到的框
	img_res = det.draw_box(img, det_res)
	#打印模型检测到的框信息
	det.print_result(det_res)
	#保存图片
	cv2.imwrite(dst_img, img_res)

def get_cls(yaml_path):
	#获取yaml中的检测类别
	with open(yaml_path,'r') as yaml_file:
		data = yaml.safe_load(yaml_file)
	cls = data.get('names',[])
	return cls

def parse_opt():
	parser = argparse.ArgumentParser()
	parser.add_argument('--weights', type=str, default='./weights/best.pt')
	parser.add_argument('--todetect', type=str, default='./todetect')
	parser.add_argument('--detected', type=str, default='./detected')
	parser.add_argument('--yaml', type=str, default='./yaml/data.yaml')

	return parser.parse_known_args()[0]


def main(opt):
	cls = get_cls(opt.yaml)
	todetect_imgs = os.listdir(opt.todetect)
	for todetect in todetect_imgs:
		todetect_path = os.path.join(opt.todetect, todetect)
		detected_path = os.path.join(opt.detected, todetect)
		inference_single_image(opt.weights, 0.3, todetect_path, detected_path, opt.yaml, cls)

if __name__ == '__main__':
	opt = parse_opt()
	main(opt)


