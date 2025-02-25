import threading

import cv2
import numpy as np

from kalmus.barcodes.InstancableBarcode import InstancableBarcode


class ColorBarcode(InstancableBarcode):
    """
    Color barcode

    :param color_metric: The metric for computing the color of the frame
    :type color_metric: str
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

    barcode_type = "color"

    def __init__(self, color_metric, frame_type, sampled_frame_rate=1, skip_over=0, total_frames=10):
        """
        Initialize the barcode with the given parameters
        """
        super().__init__(color_metric, frame_type, sampled_frame_rate, skip_over, total_frames)
        self.colors = None

    def set_copy_value(self, deepcopy):
        deepcopy['colors'] = deepcopy['colors'].tolist()

    def load_dict_value(self, object_dict):
        self.colors = np.array(object_dict["colors"]).astype("uint8")

    def generate(self, video_path, num_threads):
        if num_threads is not None:
            self.multi_thread_collect_colors(video_path, num_threads)
        else:
            self.collect_colors(video_path)

    def collect_colors(self, video_path_name):
        """
        Collect the colors of frames from the video

        :param video_path_name: The path to the video file
        :type video_path_name: str
        """
        self.read_video(video_path_name)

        success, frame = self.video.read()

        used_frames = 0
        cur_frame_idx = 0 + self.skip_over

        self.video.set(cv2.CAP_PROP_POS_FRAMES, cur_frame_idx)

        colors_sequence = []

        while success and used_frames < self.total_frames and cur_frame_idx < self.film_length_in_frames:
            if (cur_frame_idx % self.sampled_frame_rate) == 0:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                if self.save_frames_in_generation:
                    self.save_frames(used_frames, frame)
                frame = self.process_frame(frame)
                if len(frame.shape) <= 2:
                    frame = frame.reshape(-1, 1, 3)

                color = self.get_color_from_frame(frame)
                colors_sequence.append(color)

                used_frames += 1

            cur_frame_idx += 1

            success, frame = self.video.read()

        self.colors = np.array(colors_sequence).astype("uint8")
        if self.save_frames_in_generation:
            self.saved_frames = np.array(self.saved_frames)

    def multi_thread_collect_colors(self, video_path_name, num_thread=4):
        """
        Collect the color of the input video using Multi-thread method

        :param video_path_name: The path to the input video
        :type video_path_name: str
        :param num_thread: Number of threads to collect the brightness
        :type num_thread: int
        """
        # Correct the total frames temporarily for the multi-thread generation in order to
        # be according with the definition of total frames in single thread generation
        # where total frames == self.colors.size / 3 (channels)
        self.total_frames *= self.sampled_frame_rate
        self.read_video(video_path_name)

        thread_videos = [None] * num_thread
        thread_videos[0] = self.video
        for i in range(1, num_thread):
            thread_videos[i] = cv2.VideoCapture(video_path_name)

        threads = [None] * num_thread
        thread_results = [None] * num_thread
        step = self.total_frames // num_thread

        if self.save_frames_in_generation:
            saved_frame_results = [[]] * num_thread
        else:
            saved_frame_results = None

        for i, tid in zip(range(self.skip_over, self.skip_over + self.total_frames, step),
                          range(num_thread)):
            if tid == num_thread - 1:
                start_point = i
                break
            threads[tid] = threading.Thread(target=self.thread_collect_color_start_to_end,
                                            args=(thread_videos[tid], i, step, thread_results, tid,
                                                  saved_frame_results))
            threads[tid].start()

        threads[num_thread - 1] = threading.Thread(target=self.thread_collect_color_start_to_end,
                                                   args=(thread_videos[tid], start_point,
                                                         self.total_frames + self.skip_over - start_point,
                                                         thread_results, tid, saved_frame_results))
        threads[num_thread - 1].start()

        for i in range(num_thread):
            threads[i].join()

        # Now change the total frames back to the original input
        self.total_frames = int(self.total_frames / self.sampled_frame_rate)

        colors_sequence = [thread_results[0]]
        for i in range(1, num_thread):
            colors_sequence.append(thread_results[i])

        self.colors = np.concatenate(colors_sequence).astype("uint8")

        if self.save_frames_in_generation:
            for frame_arry in saved_frame_results:
                self.saved_frames += frame_arry
            self.saved_frames = np.array(self.saved_frames)

    def thread_collect_color_start_to_end(self, video, start_point, num_frames, results, tid, frame_saved=None):
        """
        Collect the colors from the video using the multi-threads

        :param video: The video object
        :type video: class:`cv2.VideoCapture`
        :param start_point: Start point for collecting the colors
        :type start_point: int
        :param num_frames: The number of frames to collect
        :type num_frames: int
        :param results: The placeholder for saving the results
        :type results: list
        :param tid: The id of the thread
        :type tid: int
        :param frame_saved: The placeholder for the saved frames
        :type frame_saved: list
        """
        assert self.video is not None, "No video is read in to the barcode for analysis."
        cur_frame_idx = start_point
        video.set(cv2.CAP_PROP_POS_FRAMES, cur_frame_idx)

        colors_sequence = []
        frame_sequence = []

        success, frame = video.read()
        used_frames = 0
        while success and used_frames < num_frames and cur_frame_idx < (start_point + num_frames):
            if (cur_frame_idx % self.sampled_frame_rate) == 0:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                if self.save_frames_in_generation:
                    self.save_frames(used_frames, frame, frame_sequence)
                frame = self.process_frame(frame)
                if len(frame.shape) <= 2:
                    frame = frame.reshape(-1, 1, 3)

                color = self.get_color_from_frame(frame)
                colors_sequence.append(color)

                used_frames += 1

            cur_frame_idx += 1

            success, frame = video.read()

        results[tid] = colors_sequence
        if self.save_frames_in_generation:
            frame_saved[tid] = frame_sequence

    def reshape_barcode(self, frames_per_column=160):
        """
        Reshape the barcode (2 dimensional with 3 channels)

        :param frames_per_column: Number of frames per column in the reshaped barcode
        :type frames_per_column: int
        """
        if len(self.colors) % frames_per_column == 0:
            self.barcode = self.colors.reshape(frames_per_column, -1, self.colors.shape[-1], order='F')
        elif len(self.colors) < frames_per_column:
            self.barcode = self.colors.reshape(-1, 1, self.colors.shape[-1], order='F')
        else:
            truncate_bound = int(len(self.colors) / frames_per_column) * frames_per_column
            self.barcode = self.colors[:truncate_bound].reshape(frames_per_column, -1,
                                                                self.colors.shape[-1], order='F')
