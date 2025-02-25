from kalmus.utils.artist import compute_mean_color


class BrightnessMetric:
    brightness_metric_type = "none"

    brightness_metric_types = {}
    """
    Contains references to every color metric class
    String identifiers used as keys
    """

    @classmethod
    def register(cls):
        print(f"Registered Brightness Metric: {cls.brightness_metric_type}")

        cls.brightness_metric_types[cls.brightness_metric_type] = cls

    @classmethod
    def get_brightness(cls, frame):
        """
        Finds the brightness using the brightness metric type

        :param frame: The input frame
        :type frame: np.ndarray
        :return: The brightness
        :rtype: np.ndarray
        """
        return compute_mean_color(frame)

    def __init__(self):
        if self.brightness_metric_type != "none":
            self.register()
