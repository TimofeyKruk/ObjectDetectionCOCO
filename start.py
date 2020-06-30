import torch
import torchvision
import yolo  # Own class of model implementation
from torch.utils.tensorboard import SummaryWriter
import data_preparation
import yolo

if __name__ == '__main__':
    print("Image Segmentation started!")

    train=data_preparation.loadCOCO(train_bool=True)
    test=data_preparation.loadCOCO(train_bool=False)

    model=yolo.modelYOLO(80)
    model=yolo.train_model(model,train,test,num_classes=80, PATH="KILLME",tensorboard=True)
