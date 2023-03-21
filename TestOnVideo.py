import numpy as np
import cv2
import abc
from PreProcessPictures import FramePreProcessor, GreyscaleCropPreprocess, GreyscaleCropResizePreprocess
from PostProcessPictures import FramePostProcessor, RedGreenTingePost
from EvaluatePictures import FrameEvaluator, DenseNetFrameEvaluator


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


def test_video(input_path, output_path, interval=None, flip=False,
               pre_processing_method=FrameEvaluator(),
               evaluation_method=FramePreProcessor(),
               post_processing_method=FramePostProcessor()):
    # Load video for testing
    input_video = cv2.VideoCapture(input_path)

    # Find relevant parameters for encoding output video
    input_video_width = int(input_video.get(cv2.CAP_PROP_FRAME_WIDTH))
    input_video_height = int(input_video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    input_video_fps = input_video.get(cv2.CAP_PROP_FPS)

    # Find number of frames to process
    if interval is not None:
        number_of_frames_to_process = int((interval[1] - interval[0]) * input_video_fps)
        start = int(interval[0] * input_video_fps)
        stop = int(interval[1] * input_video_fps)
    else:
        number_of_frames_to_process = np.inf
        start = 0
        stop = np.inf
    # output_video = cv2.VideoWriter(filename=output_path,
    #                                fourcc=-1,
    #                                fps=input_video_fps,
    #                                frameSize=(input_video_width, input_video_height))
    # output_video = cv2.VideoWriter(filename=output_path,
    #                                fourcc=cv2.VideoWriter_fourcc('m','p','4','v'),
    #                                fps=input_video_fps,
    #                                frameSize=(input_video_width, input_video_height))
    output_video = cv2.VideoWriter(filename=output_path,
                                   fourcc=cv2.VideoWriter_fourcc(*'avc1'),
                                   fps=input_video_fps,
                                   frameSize=(input_video_width, input_video_height))
    current_frame = 0
    while True:
        success, unprocessed_frame = input_video.read()
        if success and current_frame < stop:
            if current_frame > start:
                if flip:
                    unprocessed_frame = cv2.flip(unprocessed_frame, 0)
                    # unprocessed_frame = cv2.flip(unprocessed_frame, 1)
                pre_processed_frame = pre_processing_method(unprocessed_frame)
                label = evaluation_method(pre_processed_frame)
                post_processed_frame = post_processing_method(unprocessed_frame, label)
                output_video.write(post_processed_frame)
            current_frame += 1
        else:
            break
    input_video.release()
    output_video.release()


if __name__ == "__main__":
    # test_video("PeterWarmUp.mp4", "PeterWarmUpProcessed.mp4",
    #            pre_processing_method=GreyscaleCropResizePreprocess(),
    #            post_processing_method=RedGreenTingePost(),
    #            evaluation_method=DenseNetFrameEvaluator(),
    #            interval=(8, 12)
    #            )
    # test_video("GavinHigh.mp4", "GavinHighProcessed.mp4",
    #            pre_processing_method=GreyscaleCropPreprocess(),
    #            post_processing_method=RedGreenTingePost(),
    #            evaluation_method=DenseNetFrameEvaluator(),
    #            interval=(0, 4)
    #            )
    # test_video("RecordHigh.mp4", "RecordHighSecondDataProcessed.mp4",
    #            pre_processing_method=GreyscaleCropPreprocess(),
    #            post_processing_method=RedGreenTingePost(),
    #            evaluation_method=DenseNetFrameEvaluator(),
    #            interval=(0, 4)
    #            )
    test_video("Inference/MajaTest.mp4", "Inference/MajaTestSecondDataProcessed.mp4",
               pre_processing_method=GreyscaleCropPreprocess(),
               post_processing_method=RedGreenTingePost(),
               evaluation_method=DenseNetFrameEvaluator(),
               interval=(0, 6)
               )
    # test_video("PovertySquat.mp4", "PovertySquatProcessed.mp4",
    #            pre_processing_method=GreyscaleCropPreprocess(box=(330, 480, 250, 500)),
    #            post_processing_method=RedGreenTingePost(),
    #            evaluation_method=DenseNetFrameEvaluator(),
    #            interval=(18, 27)
    #            )
    # test_video("AnneSofieTredjeJM22.mp4", "AnneSofieTredjeJM22Processed.mp4", flip = True,
    #            pre_processing_method=GreyscaleCropResizePreprocess(box=(700, 1150, 300, 1050)),
    #            post_processing_method=RedGreenTingePost(),
    #            evaluation_method=DenseNetFrameEvaluator(),
    #            interval=(14, 20)
    #            )
    # test_video("AnneSofieTredjeJM22.mp4", "AnneSofieTredjeJM22Processed.mp4", flip = True,
    #            pre_processing_method=GreyscaleCropResizePreprocess(box=(700, 1150, 150, 900)),
    #            post_processing_method=RedGreenTingePost(),
    #            evaluation_method=DenseNetFrameEvaluator(),
    #            interval=(17, 19)
    #            )
    # test_video("SigurdSquats.mp4", "SigurdSquatsProcessed.mp4", flip=True,
    #            pre_processing_method=GreyscaleCropResizePreprocess(box=(400, 680, 50, 350)),
    #            post_processing_method=RedGreenTingePost(),
    #            evaluation_method=DenseNetFrameEvaluator(),
    #            interval=(10, 13)
    #            )
