import numpy as np

from kalmus.modules.frames.Frame import Frame
from kalmus.utils.artist import get_contrast_matrix_and_labeled_image


class LowContrastFrame(Frame):
    frame_type = "low_contrast"

    @classmethod
    def get_frame(cls, frame):
        contrast_matrix, labels = get_contrast_matrix_and_labeled_image(frame)
        lowest_contrast_region = np.sum(contrast_matrix, axis=1).argmin()
        return frame[labels == (lowest_contrast_region + 1)]
