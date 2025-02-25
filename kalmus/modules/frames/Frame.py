import cv2


class Frame:
    frame_type = "none"

    frame_types = {}
    """
    Contains references to every frame class
    String identifiers used as keys
    """

    @classmethod
    def register(cls):
        print("Registered: %s".format(cls.frame_type))

        cls.frame_types[cls.frame_types] = cls

    @classmethod
    def load_video(cls, video_path):
        """
        Loads the video from given path

        :param video_path: The path to the video file
        :type video_path: str
        :return: The loaded video
        :rtype: cv2.VideoCapture
        """
        return cv2.VideoCapture(video_path)

    @classmethod
    def get_frame(cls, frame):
        """
        Returns the relevant part of the frame

        :param frame: The whole frame
        :type frame: np.array
        :return: The relevant part of the frame
        :rtype: np.array
        """
        return frame

    def __init__(self):
        if self.frame_type != "none":
            self.register()
