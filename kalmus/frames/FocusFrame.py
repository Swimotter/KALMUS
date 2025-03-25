import cv2
import numpy as np

from kalmus.frames.Frame import Frame
from kalmus.utils.focus import find_focus


class FocusFrame(Frame):
    frame_type = "focus"

    @classmethod
    def load_video(cls, video_path):
        find_focus(video_path)
        return cv2.VideoCapture("./temp.mp4")

    @classmethod
    def get_frame(cls, frame):
        mask = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY) > 10
        return frame[mask].reshape(-1, 1, 3)
