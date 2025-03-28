""" Barcode Generator Class"""

import json

import numpy as np

from kalmus.barcodes.Barcode import Barcode
from kalmus.frames.Frame import Frame
from kalmus.metrics.brightness_metrics.BrightnessMetric import BrightnessMetric
from kalmus.metrics.color_metrics.ColorMetric import ColorMetric


def build_barcode_from_json(path_to_json, barcode_type="color"):
    """
    Helper function that build a barcode object from the attributes stored in a json file

    :param path_to_json: Path to the json file
    :type path_to_json: str
    :param barcode_type: Type of the barcode that stored in the json file
    :type barcode_type: str
    :return: The barcode built from the json file at given path
    :rtype: class:`kalmus.barcodes.Barcode.ColorBarcode` or class:`kalmus.barcodes.Barcode.BrightnessBarcode`
    """
    assert barcode_type in Barcode.barcode_types.keys(), "Invalid barcode type. The available types of " \
                                          "the barcode are {:s}".format(str(Barcode.barcode_types.keys()))
    with open(path_to_json, "r") as infile:
        object_dict = json.load(infile)
    infile.close()

    barcode = Barcode.barcode_types[barcode_type](object_dict["metric"], frame_type=object_dict["frame_type"],
                               sampled_frame_rate=object_dict["sampled_frame_rate"],
                               skip_over=object_dict["skip_over"], total_frames=int(object_dict["total_frames"]))
    barcode.load_dict_value(object_dict)

    barcode.set_letterbox_bound(object_dict["low_bound_ver"], object_dict["high_bound_ver"],
                                object_dict["low_bound_hor"], object_dict["high_bound_hor"])

    if ("meta_data" in object_dict.keys()) and (object_dict["meta_data"] is not None):
        barcode.meta_data = object_dict["meta_data"]

    barcode.barcode = np.array(object_dict["barcode"])

    barcode.video = None

    if ("fps" in object_dict.keys()) and (object_dict["fps"] is not None):
        barcode.fps = float(object_dict["fps"])
    else:
        barcode.fps = 30

    barcode.film_length_in_frames = int(object_dict["film_length_in_frames"])

    if "save_frames_in_generation" in object_dict.keys():
        if object_dict["save_frames_in_generation"]:
            barcode.save_frames_in_generation = object_dict["save_frames_in_generation"]
            barcode.saved_frames = np.array(object_dict["saved_frames"])

    return barcode


class BarcodeGenerator:
    """
    Barcode Generator Class

    :param frame_type: The type of the frame sampling
    :type frame_type: str
    :param color_metric: The metric of computing the frame color
    :type color_metric: str
    :param barcode_type: The type of the generated barcode
    :type barcode_type: str
    :param sampled_frame_rate: The frame sample rate \
    (one frame will be sampled from every sampled_frame_rate frames)
    :type sampled_frame_rate: int
    :param skip_over: How many frames to skip with at the beginning of the input video
    :type skip_over: int
    :param total_frames: Total number of frames that will be computed (included in the barcode/sampled frames)
    :type total_frames: int
    """
    def __init__(self, frame_type="whole_frame", color_metric="average", brightness_metric=None, barcode_type="color",
                 sampled_frame_rate=1, skip_over=0, total_frames=10):
        """
        Initialize the parameters for the barcode generator
        """
        assert frame_type in Frame.frame_types.keys(), "Invalid frame acquisition method." \
                                          "Valid frame types are: {:s}".format(str(Frame.frame_types.keys()))
        if brightness_metric is not None:
            assert brightness_metric in BrightnessMetric.brightness_metric_types.keys(), "Invalid brightness metric." \
                                                  "Valid brightness metrics are: {:s}".format(str(BrightnessMetric.brightness_metric_types.keys()))
            self.metric = brightness_metric
        else:
            assert color_metric in ColorMetric.color_metric_types.keys(), "Invalid color metric." \
                                                  "Valid color metrics are: {:s}".format(str(ColorMetric.color_metric_types.keys()))
            self.metric = color_metric
        assert barcode_type in Barcode.barcode_types.keys(), "Invalid barcode type. Two types of barcode are available " \
                                              "Valid barcode types are: {:s}".format(str(Barcode.barcode_types.keys()))

        self.frame_type = frame_type
        self.barcode_type = barcode_type
        self.sampled_frame_rate = sampled_frame_rate
        self.skip_over = skip_over
        self.total_frames = total_frames
        self.barcode = None

    def instantiate_barcode_object(self):
        """
        Instantiate the barcode object using the given generation parameters
        """
        self.barcode = Barcode.barcode_types[self.barcode_type](self.metric, frame_type=self.frame_type,
                                                      sampled_frame_rate=self.sampled_frame_rate,
                                                      skip_over=self.skip_over,
                                                      total_frames=self.total_frames)

    def generate_barcode(self, video_file_path, user_defined_letterbox=False,
                         low_ver=-1, high_ver=-1, left_hor=-1, right_hor=-1,
                         num_thread=None, save_frames=False, rescale_frames_factor=-1,
                         save_frames_rate=4):
        """
        Generate the barcode

        :param video_file_path: The path to the video file
        :type video_file_path: str
        :param user_defined_letterbox: Whether use the user defined the letterbox, or use the \
        automatically found letterbox
        :type user_defined_letterbox: bool
        :param low_ver: The lower vertical letterbox given by user
        :type low_ver: int
        :param high_ver: The higher vertical letterbox given by user
        :type high_ver: int
        :param left_hor: The left horizontal letterbox given by user
        :type left_hor: int
        :param right_hor: The right horizontal letterbox given by user
        :type right_hor: int
        :param num_thread: Number of thread for computation. None == Single thread. num_thread > 1: multi-thread
        :type num_thread: int
        :param save_frames: Whether to save the frames during the barcode generation
        :type save_frames: bool
        :param save_frames_rate: The period of seconds of one frame being saved. In other words, save 1 frame every \
        save_frame_rate seconds in the barcode generation
        :type save_frames_rate: float
        :param rescale_frames_factor: factor to rescale the input frames during the generation
        :type rescale_frames_factor: float
        """
        self.instantiate_barcode_object()
        if user_defined_letterbox:
            self.barcode.set_letterbox_bound(up_vertical_bound=high_ver, down_vertical_bound=low_ver,
                                             left_horizontal_bound=left_hor, right_horizontal_bound=right_hor)
        if save_frames and save_frames_rate > 0:
            self.barcode.enable_save_frames(sampled_rate=save_frames_rate)

        if rescale_frames_factor > 0:
            self.barcode.enable_rescale_frames_in_generation(rescale_frames_factor)

        self.barcode.generate(video_file_path, num_thread)

    def generate_barcode_from_json(self, json_file_path, barcode_type=None):
        """
        Generate the barcode from a json file, which contain a dictionary representation of barcode object

        :param json_file_path: the path to the json file
        :type json_file_path: str
        :param barcode_type: the type of the barcode saved in the json file
        :type barcode_type: str
        """
        if barcode_type is None:
            barcode_type = self.barcode_type
        self.barcode = build_barcode_from_json(json_file_path, barcode_type=barcode_type)

    def get_barcode(self):
        """
        return the barcode object stored in the Barcode generator

        :return: The generated barcode
        :rtype: class:`kalmus.barcodes.Barcode.ColorBarcode` or class:`kalmus.barcodes.Barcode.BrightnessBarcode`
        """
        assert self.barcode is not None, "There is not a generated barcode"
        return self.barcode
