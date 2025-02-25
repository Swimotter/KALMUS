from kalmus.utils.artist import compute_mean_color

from kalmus.modules.metrics.color_metrics.ColorMetric import ColorMetric


class AverageColorMetric(ColorMetric):
    metric_name = "average"

    @classmethod
    def get_color(cls, frame):
        return compute_mean_color(frame)