import copy
import json

import cv2
import numpy as np

from kalmus.metrics.color_metrics.ColorMetric import ColorMetric
from kalmus.frames.Frame import Frame
from kalmus.utils.artist import get_letter_box_from_frames



class Barcode:
    """
    Barcode Class. Base class for ColorBarcode and BrightnessBarcode

    :param frame_type: The type of frame sampling
    :type frame_type: str
    :param sampled_frame_rate: Frame sample rate: the frame sampled from every sampled_frame_rate.
    :type sampled_frame_rate: int
    :param skip_over: The number of frames to skip with at the beginning of the video
    :type skip_over: int
    :param total_frames: The total number of frames (computed) included in the barcode
    :type total_frames: int
    :type barcode_type: str
    """

    barcode_type = "none"

    barcode_types = {}
    """
    Contains references to every barcode class
    String identifiers used as keys
    """

    @classmethod
    def register(cls):
        print(f"Registered Barcode: {cls.barcode_type}")

        cls.barcode_types[cls.barcode_type] = cls

    def __init__(self, metric, frame_type, sampled_frame_rate=1, skip_over=0, total_frames=10):
        """
        Initialize the barcode with the given parameters
        """
        self.metric = metric
        self.frame_type = frame_type

        self.meta_data = {}

        self.sampled_frame_rate = sampled_frame_rate
        self.skip_over = skip_over
        self.total_frames = total_frames

        self.video = None
        self.film_length_in_frames = 0

        self.fps = None
        self.scale_factor = 1

        self.user_defined_letterbox = False
        self.low_bound_ver = 0
        self.high_bound_ver = 0
        self.low_bound_hor = 0
        self.high_bound_hor = 0

        self.barcode = None

        self.save_frames_in_generation = False
        self.saved_frames = None
        self.saved_frames_sampled_rate = -1
        self.saved_frame_height = 0

        self.rescale_frames_in_generation = False
        self.rescale_frame_factor = -1

        if self.barcode_type != "none" and self.barcode_type not in self.barcode_types:
            self.register()

    def read_video(self, video_path_name):
        """
        Read in the video from the given path

        :param video_path_name: The path to the video file
        :type video_path_name: str
        """
        # Run saliency detection and use new video
        self.video = Frame.frame_types[self.frame_type].load_video(video_path_name)

        # Get the fps of the video
        self.fps = self.video.get(cv2.CAP_PROP_FPS)

        # Get the length of the video
        self.film_length_in_frames = int(self.video.get(cv2.CAP_PROP_FRAME_COUNT))
        if self.total_frames + self.skip_over > self.film_length_in_frames:
            self.total_frames = self.film_length_in_frames - self.skip_over

        # Find the letter box
        if not self.user_defined_letterbox:
            self.find_film_letterbox()

        if self.save_frames_in_generation:
            self._determine_save_frame_param()

    def find_film_letterbox(self, num_sample=30):
        """
        Automatically find the letter box bounds of the film.
        Function run the get_letter_box_from_frames helper function by num_sample times and take the median of bounds

        :param num_sample: Number of times running the get_letter_box_from_frames
        :type num_sample: int
        """
        possible_indexes = np.arange(self.film_length_in_frames // 6, self.film_length_in_frames * 5 // 6, 1)
        frame_indexes = np.random.choice(possible_indexes, num_sample, replace=True)

        possible_low_bound_ver = []
        possible_high_bound_ver = []
        possible_low_bound_hor = []
        possible_high_bound_hor = []

        for frame_index in frame_indexes:
            self.video.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
            success, frame = self.video.read()
            if success:
                low_bound_v, high_bound_v, low_bound_h, high_bound_h = get_letter_box_from_frames(frame)
                possible_low_bound_ver.append(low_bound_v)
                possible_high_bound_ver.append(high_bound_v)
                possible_low_bound_hor.append(low_bound_h)
                possible_high_bound_hor.append(high_bound_h)

        self.low_bound_ver = int(np.median(possible_low_bound_ver))
        self.high_bound_ver = int(np.median(possible_high_bound_ver))
        self.low_bound_hor = int(np.median(possible_low_bound_hor))
        self.high_bound_hor = int(np.median(possible_high_bound_hor))

    def remove_letter_box_from_frame(self, frame):
        """
        Remove the letter box from the frame using the known letter box bounds

        :param frame: Input original frame with letter box
        :type frame: numpy.ndarray
        :return: Cropped frame without letter box
        :rtype: numpy.ndarray
        """
        frame = frame[self.low_bound_ver: self.high_bound_ver, self.low_bound_hor: self.high_bound_hor, ...]
        return frame

    def process_frame(self, frame):
        """
        Process the original frame by cropping out the letter box and resample frame using the given frame type

        :param frame: Input orignal frame
        :type frame: numpy.ndarray
        :return: The processed and sampled frame
        :rtype: numpy.ndarray
        """
        frame = self.remove_letter_box_from_frame(frame)
        if self.rescale_frames_in_generation:
            frame = self._resize_frame(frame)

        return Frame.frame_types[self.frame_type].get_frame(frame)

    def _resize_frame(self, frame):
        """
        resize the input frame with a factor of self.rescale_frame_factor

        :param frame: Input frame
        :type frame: numpy.ndarray
        :return: resized frame
        :rtype: numpy.ndarray
        """
        frame = cv2.resize(frame, dsize=(0, 0), fx=self.rescale_frame_factor, fy=self.rescale_frame_factor,
                           interpolation=cv2.INTER_NEAREST)
        return frame

    def get_color_from_frame(self, frame):
        """
        Compute the color of the input frame using the known color metric

        :param frame: Input frame
        :type frame: numpy.ndarray
        :return: The color of the frame computed using the known color metric
        :rtype: numpy.ndarray
        """
        return ColorMetric.color_metric_types[self.metric].get_color(frame)

    def get_barcode(self):
        """
        Return the barcode.

        :return: Return the barcode
        :rtype: class:`kalmus.barcodes.Barcode.Barcode`
        """
        return self.barcode

    def set_letterbox_bound(self, up_vertical_bound, down_vertical_bound,
                            left_horizontal_bound, right_horizontal_bound):
        """
        Manually set up the letter box bound of the film

        :param up_vertical_bound: The lower vertical bound
        :type up_vertical_bound: int
        :param down_vertical_bound: The higher vertical bound
        :type down_vertical_bound: int
        :param left_horizontal_bound: The left vertical bound
        :type left_horizontal_bound: int
        :param right_horizontal_bound: The right vertical bound
        :type right_horizontal_bound: int
        """
        self.enable_user_defined_letterbox()
        self.low_bound_ver = up_vertical_bound
        self.high_bound_ver = down_vertical_bound
        self.low_bound_hor = left_horizontal_bound
        self.high_bound_hor = right_horizontal_bound

    def enable_user_defined_letterbox(self):
        """
        Use the user defined letter box
        """
        self.user_defined_letterbox = True

    def automatic_find_letterbox(self):
        """
        Automatically find the letter box
        """
        self.find_film_letterbox()

    def enable_rescale_frames_in_generation(self, rescale_factor=1):
        """
        Rescale frames with a factor of rescale_factor for all frames processed in barcode generation

        :param rescale_factor: rescale factor
        :type rescale_factor: float
        """
        assert rescale_factor > 0, "Rescale factor must be Positive"
        self.rescale_frames_in_generation = True
        self.rescale_frame_factor = np.sqrt(rescale_factor)

    def add_meta_data(self, key, value):
        """
        Add the meta information that describes the barcode

        :param key: The key for the meta information
        :type key: str
        :param value: The value stored in that key
        :type value: str
        """
        assert key is not None, "The key for the adding data cannot be None"
        if self.meta_data is None:
            self.meta_data = {}
        self.meta_data[key] = value

    def enable_save_frames(self, sampled_rate=4):
        """
        Set the save frame in the generation of barcode to be True.
        This attribute, saved_frames_in_generation, should only be modified before the generation of barcode.
        Once the barcode is generated, this attribute should not be changed.

        :param sampled_rate: Save 1 frame every sampled_rate seconds
        :type sampled_rate: float
        """
        self.save_frames_in_generation = True
        self.saved_frames = []

        self.saved_frames_sampled_rate = sampled_rate

    def _determine_save_frame_param(self):
        """
        Private method
        Determine the parameters of saving frame during the generation process.
        At most 900 Frames will be saved for each barcode
        Save frame rate is, by default, saving one frame every 4 seconds
        Frame will resized to the width of 100 pixels with the same aspect ratio
        """
        assert self.video is not None, "Video must be read before determining the save frame rate"
        assert self.fps is not None, "FPS must be determined before determining the save frame rate"
        self.saved_frames_sampled_rate = round(self.fps * self.saved_frames_sampled_rate / self.sampled_frame_rate)
        sampled_rate_upper_bound = round(self.total_frames / (self.sampled_frame_rate * 900))
        if self.saved_frames_sampled_rate < sampled_rate_upper_bound:
            self.saved_frames_sampled_rate = sampled_rate_upper_bound

        # If the barcode is too short
        if self.saved_frames_sampled_rate <= 0:
            self.saved_frames_sampled_rate = 1

        height = self.high_bound_ver - self.low_bound_ver
        width = self.high_bound_hor - self.low_bound_hor

        aspect_ratio = height / width
        self.saved_frame_height = int(100 * aspect_ratio)

    def save_frames(self, cur_used_frame, frame, frame_arr=None):
        """
        Private method
        Save the frame during the generation process.
        This functions should only be invoked during the generation process.

        :param cur_used_frame: How many frames have been read in
        :type cur_used_frame: int
        :param frame: Current frame (original unprocessed frame)
        :type frame: numpy.ndarray
        :param frame_arr: Array that stored the saved frames
        :type frame_arr: list
        """
        if cur_used_frame % self.saved_frames_sampled_rate == 0:
            frame = self.remove_letter_box_from_frame(frame)
            resized_frame = cv2.resize(frame, dsize=(100, self.saved_frame_height))
            if frame_arr is not None:
                frame_arr.append(resized_frame)
            else:
                self.saved_frames.append(resized_frame)

    def set_copy_value(self, deepcopy):
        pass

    def save_as_json(self, filename=None):
        """
        Save the barcode into the json file

        :param filename: The name of the saved json file
        :type filename: str
        """
        # This cv2 captured video is not pickled in the json
        # therefore it is not able to be pickled for deepcopy
        # Delete it from the object first
        self.video = None
        barcode_dict = copy.deepcopy(self.__dict__)
        barcode_dict['barcode'] = barcode_dict['barcode'].tolist()
        if self.save_frames_in_generation:
            barcode_dict['saved_frames'] = barcode_dict['saved_frames'].tolist()
        self.set_copy_value(barcode_dict)
        barcode_dict["video"] = None

        if filename is None:
            filename = "saved_{:s}_barcode_{:s}_{:s}.json" \
                .format(self.barcode_type, self.frame_type, self.metric)

        with open(filename, "w") as file:
            json.dump(barcode_dict, file)
        file.close()