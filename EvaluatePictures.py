import abc

import torch
import torchvision
from torchvision.transforms import transforms


class FrameEvaluator:
    """
    Abstract baseclass for evaluation strategy. Children of this class should implement call and call should return
    the label of a single pre_processed frame
    """

    def __init__(self):
        pass

    @abc.abstractmethod
    def __call__(self, frame, *args, **kwargs):
        return 0


class DenseNetFrameEvaluator(FrameEvaluator):
    def __init__(self, model_path="best_model.pt"):
        super(DenseNetFrameEvaluator, self).__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = torchvision.models.densenet121()
        self.model.classifier = torch.nn.Sequential(
            torch.nn.Linear(1024, 1),
            torch.nn.Sigmoid()
        )
        self.model.load_state_dict(torch.load(model_path))
        self.model.to(self.device)

    def __call__(self, frame, *args, **kwargs):
        frame = torch.from_numpy(frame)
        frame = torch.reshape(frame, (3, 150, 250))
        frame = frame.unsqueeze(0)
        frame = frame.to(self.device)
        frame = frame.float()
        self.model.train(False)
        label = self.model(frame)
        return label
