import numpy as np

from kalmus.modules.frames.Frame import Frame
from kalmus.utils.artist import get_contrast_matrix_and_labeled_image


class HighContrastFrame(Frame):
    frame_type = "high_contrast"

    @classmethod
    def get_frame(cls, frame):
        contrast_matrix, labels = get_contrast_matrix_and_labeled_image(frame)
        highest_contrast_region = np.sum(contrast_matrix, axis=1).argmax()
        return frame[labels == (highest_contrast_region + 1)]
