from kalmus.utils import artist

from kalmus.frames.Frame import Frame


def foreback_segmentation(frame):
    """
    Helper function
    Segmented the input frame into two parts: foreground and background, using the GrabCut

    :param frame: Input frame
    :type frame: numpy.ndarray
    :return: 1D image of the foreground part of the image, and 1D image of the background part of the image \
             Expected shape== Number of pixels x channels
    :rtype: (numpy.ndarray, numpy.ndarray)
    """
    fore_frame, back_frame = artist.grabcut_foreback_segmentation(frame, start_row=0, row_size=frame.shape[0] - 1,
                                                                  start_col=frame.shape[1] // 6,
                                                                  col_size=frame.shape[1] * 2 // 3)
    return fore_frame, back_frame

class BackgroundFrame(Frame):
    frame_type = "background"

    @classmethod
    def get_frame(cls, frame):
        _, back_frame = foreback_segmentation(frame)
        if back_frame.size == 0:
            # Empty background part use the whole frame instead
            back_frame = frame
        return back_frame
