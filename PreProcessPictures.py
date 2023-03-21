import abc

import cv2
import numpy as np
import matplotlib.pyplot as plt

class FramePreProcessor:
    """
    Abstract baseclass for preprocessing strategy. Children of this class should implement call and call
    should take a single unprocessed frame and return a single pre_processed frame
    """

    def __init__(self):
        pass

    @abc.abstractmethod
    def __call__(self, frame, *args, **kwargs):
        return frame


class GreyscaleCropPreprocess(FramePreProcessor):
    def __init__(self, box=(150, 300, 150, 400)):
        super(GreyscaleCropPreprocess, self).__init__()
        self.box = box
        self.called = False

    def __call__(self, frame, *args, **kwargs):

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        frame = frame.astype(np.uint8)
        frame = frame.astype('float')
        frame = frame[self.box[0]:self.box[1], self.box[2]:self.box[3]]
        if self.called:
            return frame
        else:
            self.called = True
            plt.imshow(frame[:,:,0], interpolation='nearest')
            plt.show()
            return frame


class GreyscaleCropResizePreprocess(FramePreProcessor):
    def __init__(self, box=(150, 300, 150, 400)):
        super(GreyscaleCropResizePreprocess, self).__init__()
        self.box = box
        self.called = False

    def __call__(self, frame, *args, **kwargs):
        if not self.called:
            plt.imshow(frame[:, :, 0], interpolation='nearest')
            plt.show()
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        frame = frame.astype(np.uint8)
        frame = frame.astype('float')
        if not self.called:
            plt.imshow(frame[:, :, 0], interpolation='nearest')
            plt.show()
        frame = frame[self.box[0]:self.box[1], self.box[2]:self.box[3]]
        frame = cv2.resize(frame, (250, 150), interpolation=cv2.INTER_AREA)
        if self.called:
            return frame
        else:
            self.called = True
            plt.imshow(frame[:, :, 0], interpolation='nearest')
            plt.show()
            return frame
