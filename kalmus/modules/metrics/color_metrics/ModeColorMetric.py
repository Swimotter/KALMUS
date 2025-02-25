from kalmus.utils.artist import compute_mode_color

from kalmus.modules.metrics.color_metrics.ColorMetric import ColorMetric


class ModeColorMetric(ColorMetric):
    color_metric_type = "mode"

    @classmethod
    def get_color(cls, frame):
        return compute_mode_color(frame)[0]