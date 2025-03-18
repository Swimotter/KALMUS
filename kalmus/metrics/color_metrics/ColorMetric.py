from kalmus.utils.artist import compute_mean_color


class ColorMetric:
    color_metric_type = "none"

    color_metric_types = {}
    """
    Contains references to every color metric class
    String identifiers used as keys
    """

    @classmethod
    def register(cls):
        print(f"Registered Color Metric: {cls.color_metric_type}")

        cls.color_metric_types[cls.color_metric_type] = cls

    @classmethod
    def get_color(cls, frame):
        """
        Finds the color using the color metric type

        :param frame: The input frame
        :type frame: np.ndarray
        :return: The color
        :rtype: np.ndarray
        """
        return compute_mean_color(frame)

    def __init__(self):
        if self.color_metric_type != "none" and self.color_metric_type not in ColorMetric.color_metric_types:
            self.register()
