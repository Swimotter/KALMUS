import cv2
import numpy as np

from kalmus.metrics.brightness_metrics.BrightnessMetric import BrightnessMetric


class BrightestBrightnessMetric(BrightnessMetric):
    brightness_metric_type = "brightest"

    @classmethod
    def get_color(cls, frame):
        grey_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        pos = np.argwhere(grey_frame == grey_frame.max())[0]
        return frame[pos[0], pos[1]].copy().astype("uint8")