import torch
import torchvision
import yolo  # Own class of model implementation
from torch.utils.tensorboard import SummaryWriter
import data_preparation
import yolo

if __name__ == '__main__':
    print("YOLO Object Detection. Start.py")
    img_size_transform = 448
    batch_size = 16
    num_classes=90
    train = data_preparation.loadCOCO(img_size_transform, train_bool=True, batch_size=batch_size)
    test = data_preparation.loadCOCO(img_size_transform, train_bool=False, batch_size=batch_size)

    model = yolo.modelYOLO(num_classes=num_classes)
    print("Training started:")
    model = yolo.train_model(model, train, test, num_classes=num_classes, PATH="KILLME", tensorboard=True,cuda=False)
