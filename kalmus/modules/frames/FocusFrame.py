import cv2
import numpy as np

from kalmus.modules.frames.Frame import Frame
from kalmus.utils.focus import find_focus


class FocusFrame(Frame):
    frame = "focus"

    @classmethod
    def load_video(cls, video_path):
        find_focus(video_path)
        return cv2.VideoCapture("./temp.mp4")

    @classmethod
    def get_frame(cls, frame):
        alpha = np.sum(frame, axis=-1) > 0

        # Convert True/False to 0/255 and change type to "uint8" to match "na"
        alpha = np.uint8(alpha * 255)

        # Stack new alpha layer with existing image to go from BGR to BGRA, i.e. 3 channels to 4 channels
        return np.dstack((frame, alpha))
