import numpy as np

from kalmus.utils.artist import compute_dominant_color

from kalmus.metrics.color_metrics.ColorMetric import ColorMetric


class TopDominantColorMetric(ColorMetric):
    color_metric_type = "top_dominant"

    @classmethod
    def get_color(cls, frame):
        colors, dominances = compute_dominant_color(frame, n_clusters=3)
        pos = np.argsort(dominances)[-1]
        return colors[pos]