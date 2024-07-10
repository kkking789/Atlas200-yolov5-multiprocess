import cv2
import yaml
import numpy as np
from lib import om_net

import acl
from lib.acl_resource import AclResource
from lib.acl_model import Model


def letterbox(img, new_shape=(640, 640), color=(114, 114, 114)):
    shape = img.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)
    dw, dh = 0.0, 0.0
    new_unpad = (new_shape[1], new_shape[0])
    ratio = (new_shape[1] / shape[1], new_shape[0] / shape[0])  # width, height ratios
    if shape[::-1] != new_unpad:  # resize
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    return img, ratio


class detector:
    def __init__(self, yaml_path, om_path, img_size=(640, 640), iou_thres=0.5, conf_thres=0.5):
        self.yaml_path = yaml_path
        self.om_path = om_path
        self.img_size = img_size
        self.iou_thres = iou_thres
        self.conf_thres = conf_thres
        self.classes = get_cls(yaml_path)
        self.det = om_net.yolo_om(om_path, self.classes, conf_thres=conf_thres, iou_thres=iou_thres, img_size=img_size)

    def detecting(self, img_org):
        img, ratio = letterbox(img_org, new_shape=self.img_size)
        img = img[:, :, ::-1].transpose(2, 0, 1)  # bgr2rgb, HWC2CHW
        img = np.expand_dims(img, 0).astype(np.float32)
        data = np.ascontiguousarray(img) / 255.0
        output = self.det.net.execute([data, ])  # 设置模型输入
        outbox = []
        for index, p in enumerate(output):
            outbox = self.det.filter_box(p)
        or_img = self.det.draw(img_org, outbox)
        return or_img, outbox


def get_cls(yaml_path):
    # 获取yaml中的检测类别
    with open(yaml_path, 'r') as yaml_file:
        data = yaml.safe_load(yaml_file)
    cls = data.get('names', [])
    return cls
