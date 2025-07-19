import sys
import torch
import cv2
import numpy as np

sys.path.insert(0, './yolov5')

from models.common import DetectMultiBackend
from utils.general import non_max_suppression, scale_boxes
from utils.torch_utils import select_device

class Detector:
    def __init__(self):
        self.device = select_device('')
        self.model = DetectMultiBackend('yolov5s.pt', device=self.device)
        self.model.warmup(imgsz=(1, 3, 640, 640))

    def detect(self, frame):
        img0 = frame.copy()
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).float()
        img = img.to(self.device) / 255.0

        pred = self.model(img)
        pred = non_max_suppression(pred, 0.3, 0.45, None, False, max_det=1)[0]

        detected_labels = []
        crop_img = None

        if pred is not None and len(pred):
            pred[:, :4] = scale_boxes(img.shape[2:], pred[:, :4], img0.shape).round()
            for *box, conf, cls in pred:
                x1, y1, x2, y2 = map(int, box)
                label = self.model.names[int(cls)]
                detected_labels.append(label)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)
                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,0), 2)
                crop_img = img0[y1:y2, x1:x2] if y2 > y1 and x2 > x1 else None

        return detected_labels, frame, crop_img
