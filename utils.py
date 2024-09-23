import torch
import logging
import os
# import wandb
import re


def atoi(text):
    return int(text) if text.isdigit() else text


def natural_keys(text):
    return [atoi(c) for c in re.split(r'(\d+)', text)]


def init_wandb(args):
    wandb.init(project=args.wandb)

    # Save run name.
    wandb.run.save()
    run_name = wandb.run.name

    # Log args.
    config = wandb.config
    config.update(args)

    return run_name


def set_logger(log_path, file_name):
    os.makedirs(log_path, exist_ok=True)
    path = os.path.join(log_path, file_name)

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    # Log to .txt
    file_handler = logging.FileHandler(path)
    file_handler.setFormatter(logging.Formatter('%(asctime)s:%(levelname)s: %(message)s'))
    logger.addHandler(file_handler)

    # Log to console
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(logging.Formatter('%(message)s'))
    logger.addHandler(stream_handler)

    return logger


class AverageMeter(object):

    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class Scores():
    # Computes and stores measures of inference

    def __init__(self):
        self.TP = 0
        self.TN = 0
        self.FP = 0
        self.FN = 0

        self.patient_dice = []
        self.patient_iou = []
        self.accuracy = []
        self.precision = []

    def record(self, preds, label):
        assert len(torch.unique(preds)) < 3

        tp = torch.sum((label == 1) * (preds == 1))
        tn = torch.sum((label == 0) * (preds == 0))
        fp = torch.sum((label == 0) * (preds == 1))
        fn = torch.sum((label == 1) * (preds == 0))

        self.patient_dice.append(2 * tp / (2 * tp + fp + fn))
        self.patient_iou.append(tp / (tp + fp + fn))
        self.accuracy.append((tp + tn) / (tp + tn + fp + fn))
        prec = tp / (tp + fp)
        if (torch.isnan(prec)):
            self.precision.append(torch.tensor(0))
        else:
            self.precision.append(prec)

        self.TP += tp
        self.TN += tn
        self.FP += fp
        self.FN += fn

    def compute_dice(self):
        return 2 * self.TP / (2 * self.TP + self.FP + self.FN)

    def compute_iou(self):
        return self.TP / (self.TP + self.FP + self.FN)
    
    def compute_accuracy(self):
        return ((self.TP + self.TN) / (self.TP + self.TN + self.FP + self.FN))
    
    def compute_precision(self):
        return self.TP / (self.TP + self.FP)

