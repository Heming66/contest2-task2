import matplotlib.pyplot as plt

# helper function for data visualization
import numpy as np
import torch
import torchmetrics
import cv2
import torch.nn as nn
import random
import segmentation_models_pytorch as smp

from scipy import ndimage
from segmentation_models_pytorch.utils import base,functional
from segmentation_models_pytorch.base.modules import Activation
from segmentation_models_pytorch.utils.functional import _take_channels, _threshold
from sklearn.metrics import roc_auc_score
from torch.nn import functional as F

class AUC(base.Metric):

    def __init__(self, threshold=0.5, activation=None, ignore_channels=None, **kwargs):
        super().__init__(**kwargs)
        self.threshold = threshold
        self.activation = Activation(activation)
        self.ignore_channels = ignore_channels

    def forward(self, y_pr, y_gt):
        y_pr = self.activation(y_pr)
        return roc_auc_score(
            y_pr, y_gt,
            threshold=self.threshold,
            ignore_channels=self.ignore_channels,
        )

def specificity(pr, gt, eps=1e-7, threshold=None, ignore_channels=None):

    pr = _threshold(pr, threshold=threshold)
    pr, gt = _take_channels(pr, gt, ignore_channels=ignore_channels)

    tp = torch.sum(gt * pr)
    fp = torch.sum(pr) - tp
    fn = torch.sum(gt) - tp
    tn = gt.view(-1).shape[0] - tp - fp -fn

    score = (tn + eps) / (tn + fp + eps)

    return score


class Specificity(base.Metric):

    def __init__(self, eps=1e-7, threshold=0.5, activation=None, ignore_channels=None, **kwargs):
        super().__init__(**kwargs)
        self.eps = eps
        self.threshold = threshold
        self.activation = Activation(activation)
        self.ignore_channels = ignore_channels

    def forward(self, y_pr, y_gt):
        y_pr = self.activation(y_pr)
        return specificity(
            y_pr, y_gt,
            eps=self.eps,
            threshold=self.threshold,
            ignore_channels=self.ignore_channels,
        )

def dice(pr, gt, eps=1e-7, threshold=0.5, ignore_channels=None):
    pr = _threshold(pr, threshold=threshold)
    pr, gt = _take_channels(pr, gt, ignore_channels=ignore_channels)

    intersection = torch.sum(gt * pr)
    dice_eff = ((2. * intersection) + eps) / (torch.sum(gt) + torch.sum(pr) + eps)
    return dice_eff

class Dice(base.Metric):

    def __init__(self, eps=1e-7, threshold=0.5, activation=None, ignore_channels=None, **kwargs):
        super().__init__(**kwargs)
        self.eps = eps
        self.threshold = threshold
        self.activation = Activation(activation)
        self.ignore_channels = ignore_channels

    def forward(self, y_pr, y_gt):
        y_pr = self.activation(y_pr)
        return dice(
            y_pr, y_gt,
            eps=self.eps,
            threshold=self.threshold,
            ignore_channels=self.ignore_channels,
        )


def set_seed(random_seed):
    random.seed(random_seed)
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def get_model(args):
    if args.model == 'unet++':
        return smp.UnetPlusPlus(
            encoder_name=args.encoder,
            encoder_weights=args.encoder_weights,
            classes=1,
            in_channels=3,
            activation=args.activation,
        )
    elif args.model == 'manet':
        return smp.MAnet(
            encoder_name=args.encoder,
            encoder_weights=args.encoder_weights,
            classes=1,
            in_channels=3,
            activation=args.activation,
        )
    elif args.model == 'deeplabv3+':
        return smp.DeepLabV3Plus(
            encoder_name=args.encoder,
            encoder_weights=args.encoder_weights,
            classes=1,
            in_channels=3,
            activation=args.activation,
        )
