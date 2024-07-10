import os

from lib import yolo5_detect
import cv2
import time
import argparse
import shutil
import re
import numpy as np
from multiprocessing import Process

start = 0
running = 1


def print_progress_bar(current_value, total_value, bar_length=50):
    progress = int(bar_length * current_value / total_value)
    bar = "[" + "#" * progress + " " * (bar_length - progress) + "]"
    percentage = int(current_value * 100 / total_value)
    print(f"\r{bar} {percentage}%", end=" ", flush=True)
    print(f"{current_value}/{total_value}", end="\n")


def split_array(arr, n):
    # Calculate the length of each group
    k, m = divmod(len(arr), n)
    # Split the array
    return [arr[i * k + min(i, m):(i + 1) * k + min(i + 1, m)] for i in range(n)]


def multi_detect(opt, todetect_imgs):
    det = yolo5_detect.detector(opt.yaml, opt.weights, img_size=(opt.size, opt.size))
    cnt = 0
    for todetect in todetect_imgs:
        if todetect.endswith(('.jpg', '.jpeg', '.png')):
            cnt += 1
            starttime = time.time()
            src_mat = cv2.imread(opt.todetect + todetect)
            dst_mat, det_res = det.detecting(src_mat)
            endtime = time.time()
            print(f"运算时间:{(endtime - starttime) * 1000:.2f}ms")
            cv2.imwrite(opt.detected + todetect, dst_mat)
            print_progress_bar(cnt, len(todetect_imgs))
    print("该线程完成运算")




def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, default='./weights/temp.om')#权重路径
    parser.add_argument('--multiprocess', type=int, default=1)#是否开启多线程
    parser.add_argument('--processnum', type=int, default=3)#线程数量
    parser.add_argument('--todetect', type=str, default='./todetect/images/')#输入图片路径
    parser.add_argument('--detected', type=str, default='./detected/')#输出图片路径
    parser.add_argument('--yaml', type=str, default='./yaml/data.yaml')#数据的类型存储路径
    parser.add_argument('--clr', type=int, default=1)#是否清除输出的图片路径的文件夹
    parser.add_argument('--tresh', type=float, default=0.3)#阈值
    parser.add_argument('--colors', type=int, default=None)#输出图片框的颜色，为none时为随机颜色
    parser.add_argument('--size', type=int, default=640)#网络需要的输入图片的大小

    return parser.parse_known_args()[0]


def main(opt):
    if not os.path.exists(opt.detected):
        os.mkdir(opt.detected)
    if opt.clr:
        shutil.rmtree(opt.detected)
        os.mkdir(opt.detected)
    print(opt.block)
    todetect_imgs = os.listdir(opt.todetect)
    if not opt.multiprocess:#非多线程检测
        chair_starttime = time.time()
        det = yolo5_detect.detector(opt.yaml, opt.weights, img_size=(opt.size, opt.size))#模型初始化
        cnt = 0
        for todetect in todetect_imgs:#遍历路径下的图片
            if todetect.endswith(('.jpg', '.jpeg', '.png')):
                starttime = time.time()
                cnt+=1
                src_mat = cv2.imread(opt.todetect + todetect)
                dst_mat, det_res = det.detecting(src_mat)#图片检测
                endtime = time.time()
                print(f"运算时间:{(endtime - starttime) * 1000:.2f}ms")
                cv2.imwrite(opt.detected + todetect, dst_mat)#图片输出到指定路径
                print_progress_bar(cnt, len(todetect_imgs))
        chair_endtime = time.time()
        print(f"整体的运算时间:{(chair_endtime - chair_starttime) * 1000:.2f}ms")
    else:#多线程检测
        groups = split_array(todetect_imgs, opt.processnum)#输入图片划分为多组
        p_group = [None]*opt.processnum
        chair_starttime = time.time()
        for i, todetect_group in enumerate(groups):
            p_group[i] = Process(target=multi_detect, args=(opt, todetect_group))#线程多开
            p_group[i].daemon = True
            p_group[i].start()
        for i in range(opt.processnum):
            p_group[i].join()

        chair_endtime = time.time()
        print(f"整体的运算时间:{(chair_endtime - chair_starttime) * 1000:.2f}ms")






if __name__ == '__main__':
    opt = parse_opt()
    main(opt)
