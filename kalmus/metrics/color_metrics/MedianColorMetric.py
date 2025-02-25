from kalmus.utils.artist import compute_median_color

from kalmus.metrics.color_metrics.ColorMetric import ColorMetric


class MedianColorMetric(ColorMetric):
    color_metric_type = "median"

    @classmethod
    def get_color(cls, frame):
        return compute_median_color(frame)