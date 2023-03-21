import abc

import cv2


class FramePostProcessor:
    """
    Abstract baseclass for preprocessing strategy. Children of this class should implement call and call
    should take a single unprocessed frame, the label of the frame and return a single post_processed frame
    """

    def __init__(self):
        pass

    @abc.abstractmethod
    def __call__(self, frame, label, *args, **kwargs):
        return frame


class RedGreenTingePost(FramePostProcessor):
    def __init__(self):
        super(RedGreenTingePost, self).__init__()

    def __call__(self, frame, label, *args, **kwargs):
        if label.item() > 0.5:
            frame = cv2.applyColorMap(frame, cv2.COLORMAP_SPRING)
        else:
            frame = cv2.applyColorMap(frame, cv2.COLORMAP_AUTUMN)
        print(label.item())
        return frame

