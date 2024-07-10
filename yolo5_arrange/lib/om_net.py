import cv2
import numpy as np

from lib.acl_resource import AclResource
from lib.acl_model import Model



class yolo_om:
    def __init__(self, om_path, classes, conf_thres=0, iou_thres=0.5, img_size=(640, 640), color='random'):
        acl_resource = AclResource()
        acl_resource.init()

        self.om_path = om_path
        self.classes = classes
        self.net = Model(acl_resource, self.om_path)
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres
        self.img_size = img_size

        if color == 'random':
            self.color = np.random.randint(0, 255, size=(len(self.classes), 3), dtype=np.uint8)
        else:
            self.color = color
        self.color = [tuple(row.tolist()) for row in self.color]
    def nms(self, dets):
        # dets:x1 y1 x2 y2 score class
        # x[:,n]就是取所有集合的第n个数据
        x1 = dets[:, 0]
        y1 = dets[:, 1]
        x2 = dets[:, 2]
        y2 = dets[:, 3]
        # -------------------------------------------------------
        #   计算框的面积
        #	置信度从大到小排序
        # -------------------------------------------------------
        areas = (y2 - y1 + 1) * (x2 - x1 + 1)
        scores = dets[:, 4]
        # print(scores)
        keep = []
        index = scores.argsort()[::-1]  # np.argsort()对某维度从小到大排序
        # [::-1] 从最后一个元素到第一个元素复制一遍。倒序从而从大到小排序

        while index.size > 0:
            i = index[0]
            keep.append(i)
            # -------------------------------------------------------
            #   计算相交面积
            #	1.相交
            #	2.不相交
            # -------------------------------------------------------
            x11 = np.maximum(x1[i], x1[index[1:]])
            y11 = np.maximum(y1[i], y1[index[1:]])
            x22 = np.minimum(x2[i], x2[index[1:]])
            y22 = np.minimum(y2[i], y2[index[1:]])

            w = np.maximum(0, x22 - x11 + 1)
            h = np.maximum(0, y22 - y11 + 1)

            overlaps = w * h
            # -------------------------------------------------------
            #   计算该框与其它框的IOU，去除掉重复的框，即IOU值大的框
            #	IOU小于thresh的框保留下来
            # -------------------------------------------------------
            ious = overlaps / (areas[i] + areas[index[1:]] - overlaps)
            idx = np.where(ious <= self.iou_thres)[0]
            index = index[idx + 1]
        return keep

    def xywh2xyxy(self, x):
        # [x, y, w, h] to [x1, y1, x2, y2]
        y = np.copy(x)
        y[:, 0] = x[:, 0] - x[:, 2] / 2
        y[:, 1] = x[:, 1] - x[:, 3] / 2
        y[:, 2] = x[:, 0] + x[:, 2] / 2
        y[:, 3] = x[:, 1] + x[:, 3] / 2
        return y

    def filter_box(self, org_box):  # 过滤掉无用的框
        # -------------------------------------------------------
        #   删除为1的维度
        #	删除置信度小于conf_thres的BOX
        # -------------------------------------------------------
        org_box = np.squeeze(org_box)  # 删除数组形状中单维度条目(shape中为1的维度)
        # (25200, 9)
        # […,4]：代表了取最里边一层的所有第4号元素，…代表了对:,:,:,等所有的的省略。此处生成：25200个第四号元素组成的数组
        conf = org_box[..., 4] > self.conf_thres  # 0 1 2 3 4 4是置信度，只要置信度 > conf_thres 的
        box = org_box[conf == True]  # 根据objectness score生成(n, 9)，只留下符合要求的框
        # print('box:符合要求的框')
        # print(box.shape)

        # -------------------------------------------------------
        #   通过argmax获取置信度最大的类别
        # -------------------------------------------------------
        cls_cinf = box[..., 5:]  # 左闭右开（5 6 7 8），就只剩下了每个grid cell中各类别的概率
        cls = []
        for i in range(len(cls_cinf)):
            cls.append(int(np.argmax(cls_cinf[i])))  # 剩下的objecctness score比较大的grid cell，分别对应的预测类别列表
        all_cls = list(set(cls))  # 去重，找出图中都有哪些类别
        # set() 函数创建一个无序不重复元素集，可进行关系测试，删除重复数据，还可以计算交集、差集、并集等。
        # -------------------------------------------------------
        #   分别对每个类别进行过滤
        #   1.将第6列元素替换为类别下标
        #	2.xywh2xyxy 坐标转换
        #	3.经过非极大抑制后输出的BOX下标
        #	4.利用下标取出非极大抑制后的BOX
        # -------------------------------------------------------
        output = []
        for i in range(len(all_cls)):
            curr_cls = all_cls[i]
            curr_cls_box = []
            curr_out_box = []

            for j in range(len(cls)):
                if cls[j] == curr_cls:
                    box[j][5] = curr_cls
                    curr_cls_box.append(box[j][:6])  # 左闭右开，0 1 2 3 4 5

            curr_cls_box = np.array(curr_cls_box)  # 0 1 2 3 4 5 分别是 x y w h score class
            # curr_cls_box_old = np.copy(curr_cls_box)
            curr_cls_box = self.xywh2xyxy(curr_cls_box)  # 0 1 2 3 4 5 分别是 x1 y1 x2 y2 score class
            curr_out_box = self.nms(curr_cls_box)  # 获得nms后，剩下的类别在curr_cls_box中的下标

            for k in curr_out_box:
                output.append(curr_cls_box[k])
        output = np.array(output)
        return output

    def draw(self, image, box_data):
        # -------------------------------------------------------
        #	取整，方便画框
        # -------------------------------------------------------
        if box_data.size == 0:
            return image
        boxes = box_data[..., :4].astype(np.int32)  # x1 x2 y1 y2
        scores = box_data[..., 4]
        classes = box_data[..., 5].astype(np.int32)
        org_img_y, org_img_x, channel = image.shape
        scale_x = (org_img_x / self.img_size[0])
        scale_y = (org_img_y / self.img_size[1])
        for box, score, cl in zip(boxes, scores, classes):
            top, left, right, bottom = box
            print('class: {}, score: {}'.format(self.classes[cl], score))
            print('box coordinate left,top,right,down: [{}, {}, {}, {}]'.format(top, left, right, bottom))
            cv2.rectangle(image, (int(top * scale_x), int(left * scale_y)),
                          (int(right * scale_x), int(bottom * scale_y)), color=self.color[cl], thickness=2)
            cv2.putText(image, '{0} {1:.2f}'.format(self.classes[cl], score),
                        (int(top * scale_x), int(left * scale_y)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, color=self.color[cl], thickness=2)
        return image