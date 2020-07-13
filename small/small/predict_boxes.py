import torch
import yolo
import data_preparation
import matplotlib.pyplot as plt
import numpy as np
import yolo_loss
import cv2
from src import post_processing


#___SMALL___
def load_model(PATH, class_number=95):
    model = yolo.modelYOLO(num_classes=class_number)
    model.load_state_dict(torch.load(PATH))
    print("Model was loaded from file: ", PATH)
    return model


if __name__ == '__main__':
    print("Predicting object boxes started!__SMALL__")
    PATH = "F:\WORK_Oxagile\INTERN\ImageSegmentation\small\SMALL_SavedModelWeights6_after15_after20"
    num_classes = 5
    batch_size = 2
    img_size_transform = 32 * 13

    # Loading model from memory
    model = load_model(PATH, num_classes)

    dataset = data_preparation.loadCOCO("F:\WORK_Oxagile\INTERN\Datasets\COCO\\", img_size_transform,
                                        train_bool=True,
                                        batch_size=batch_size)

    for data in dataset:
        images, targets = data[0], data[1]

        with torch.no_grad():
            outputs = model(images)
            criterion = yolo_loss.yoloLoss(num_classes, device="cpu", cuda=False)
            loss_total, loss_coordinates, loss_confidence, loss_classes = criterion(outputs, targets)

        post_images = post_processing.draw_predictions(images, outputs)

        if len(targets) != 0:
            post_images = post_processing.draw_gt_boxes(post_images, targets)

        for post_image in post_images:
            # cv2.imshow("Post", post_image)
            plt.imshow(post_image)
            plt.show()

        # out = outputs[1].view(5, 100, 13, 13)
        # print("Outputs[1].view.shape", out.shape)
        # for i in range(13):
        #     for j in range(13):
        #         print("Position: i", i, " j", j)
        #         for anchor in range(5):
        #             if out[anchor, 4, i, j].sigmoid() > 0.1:
        #                 print("Anchor: ", anchor, ", confidence: ", out[anchor, 4, i, j].sigmoid(), ", class: ",
        #                       torch.argmax(torch.softmax(out[anchor, 5:, i, j], dim=0)))

        break