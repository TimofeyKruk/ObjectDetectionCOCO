from torch.utils.tensorboard import SummaryWriter
import data_preparation
import os
import yolo
import argparse
import torch

## SMALL/SMALL

if __name__ == '__main__':

    CLASSES = {1: "person",
               2: "car",
               3: "bird",
               4: "cat",
               5: "dog"}

    print("YOLO Object Detection. Start.py")
    print("CWD: ")
    print("__CWD: ", os.getcwd())
    parser = argparse.ArgumentParser()
    parser.add_argument("--saveName", help="Name how to save model weights file", default="SMALL_SavedModelWeights13")
    parser.add_argument("--dataset_path", help="PATH to dataset location",
                        default="//media//cuda//HDD//Internship//Kruk//COCO//")
    parser.add_argument("--tensorboard", help="Name how to save tensorboard logs",
                        default="SMALL_yolov2_training13")
    parser.add_argument("--img_size", help="Images will be scaled to img_size*img_size", default="416")
    parser.add_argument("--batch", help="Batch size", default="32")
    parser.add_argument("--num_classes", help="Int number of classes", default="5")
    parser.add_argument("--epochs", help="Number of epochs to train", default="20")
    parser.add_argument("--epoch_start", help="Number of epoch to continue to train", default="0")
    parser.add_argument("--cuda", help="Bool. Whether to train on CUDA or not", default="True")
    parser.add_argument("--continue_training",
                        help="Whether to download weights and continue to train or start from the beginning",
                        default="False")
    parser.add_argument("--lr_start", help="Learning rate to start this training with", default="0.0001")
    parser.add_argument("--save_every", help="Number of every epochs to save model weights", default="5")
    parser.add_argument("--no_person", help="Boolean variable whether to train on images with person or not",
                        default="False")

    args = parser.parse_args()

    save_every = int(args.save_every)
    lr_start = float(args.lr_start)
    continue_training = True if args.continue_training == "True" else False
    epoch_start = int(args.epoch_start)

    saveName = args.saveName
    tensorboard_name = args.tensorboard
    datasetPATH = args.dataset_path
    img_size_transform = int(args.img_size)
    batch_size = int(args.batch)
    num_classes = int(args.num_classes)
    epochs = int(args.epochs)
    print("Train model for: ", epochs, " epochs")

    cuda = args.cuda
    cuda = False if cuda == "False" else True

    no_person = args.no_person
    no_person = False if no_person == "False" else True
    print("___No person arg is: ", no_person)

    model = yolo.modelYOLO(num_classes=num_classes)

    if continue_training is True:
        print("Loading model for future training from file: ", saveName)
        model.load_state_dict(torch.load(saveName))
        print("Model was loaded!")

    tensorboard = SummaryWriter(tensorboard_name)

    tensorboard.add_graph(model, torch.rand(batch_size, 3, img_size_transform, img_size_transform))

    print("___Train loadCOCO started!")
    train = data_preparation.loadCOCO(datasetPATH, img_size_transform, train_bool=True, batch_size=batch_size,
                                      no_person=no_person)
    print("___Test loadCOCO started!")
    test = data_preparation.loadCOCO(datasetPATH, img_size_transform, train_bool=False, batch_size=batch_size)

    print("___Training started:")
    model = yolo.train_model(model,
                             train,
                             test,
                             num_classes=num_classes,
                             saveName=saveName,
                             tensorboard=tensorboard,
                             cuda=cuda,
                             lr_start=lr_start,
                             epoch_start=epoch_start,
                             epochs=epochs,
                             save=True,
                             save_every=save_every)

    print("___Model trained!!!")
    tensorboard.close()
