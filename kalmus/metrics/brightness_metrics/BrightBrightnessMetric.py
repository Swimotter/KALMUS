import numpy as np

from kalmus.metrics.brightness_metrics.BrightnessMetric import BrightnessMetric
from kalmus.utils.artist import find_bright_spots, compute_mean_color


class BrightBrightnessMetric(BrightnessMetric):
    """
    BROKEN DO NOT USE
    """

    brightness_metric_type = "bright"

    @classmethod
    def get_color(cls, frame):
        labels, bright_locations, dominance = find_bright_spots(frame, n_clusters=3, return_all_pos=True)
        top_bright = np.argsort(dominance)[-1]
        top_bright_pos = (labels == top_bright)[:, 0]
        pos = bright_locations[top_bright_pos]
        frame = frame[pos[:, 0], pos[:, 1]].reshape(pos.shape[0], 1, 3)
        return compute_mean_color(frame)