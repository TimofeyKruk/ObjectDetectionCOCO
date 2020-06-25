import torch
import torchvision
import torch.nn as nn


class modelYOLO(nn.Module):
    def __init__(self) -> None:
        super(modelYOLO, self).__init__()

    def forward(self, x):
        return x


def train_model(model_yolo, train, test, PATH, tensorboard, epoch=10, cuda=True, save=True) -> modelYOLO:
    '''
    :param model_yolo: object of class modelYOLO
    :param train: train_loader <- data to train the model
    :param test:test_loader <- data to test on
    :param PATH: string <- directory to save model
    :param epoch: int <- number of epoch for training
    :param cuda: bool <- whether to train on GPU or not
    :param save:bool <- variable for saving model weights or not
    :return: model_yolo: modelYOLO object <- trained model
    '''
    
    return model_yolo
