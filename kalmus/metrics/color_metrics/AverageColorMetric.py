from kalmus.utils.artist import compute_mean_color

from kalmus.metrics.color_metrics.ColorMetric import ColorMetric


class AverageColorMetric(ColorMetric):
    color_metric_type = "average"

    @classmethod
    def get_color(cls, frame):
        return compute_mean_color(frame)