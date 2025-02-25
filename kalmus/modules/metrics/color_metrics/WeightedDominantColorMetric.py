import numpy as np

from kalmus.utils.artist import compute_dominant_color

from kalmus.modules.metrics.color_metrics.ColorMetric import ColorMetric


class WeightedDominantColorMetric(ColorMetric):
    color_metric_type = "weighted_dominant"

    @classmethod
    def get_color(cls, frame):
        colors, dominances = compute_dominant_color(frame, n_clusters=3)
        return np.sum(colors * dominances.reshape(dominances.shape[0], 1), axis=0)