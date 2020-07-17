import torch
import yolo
import data_preparation
import matplotlib.pyplot as plt
import numpy as np
import yolo_loss
import cv2
from src import post_processing


# ___SMALL___
def load_model(PATH, class_number=95):
    model = yolo.modelYOLO(num_classes=class_number)
    model.load_state_dict(torch.load(PATH))
    print("Model was loaded from file: ", PATH)
    return model


if __name__ == '__main__':
    print("Predicting object boxes started!__SMALL__")
    PATH = "F:\WORK_Oxagile\INTERN\ImageSegmentation\small\SMALL_SavedModelWeights6_after15_after20_after30_after35_after44_after60_after70"
    print("Model PATH: ", PATH)
    num_classes = 5
    batch_size = 2
    img_size_transform = 32 * 13

    # Loading model from memory
    model = load_model(PATH, num_classes)

    dataset = data_preparation.loadCOCO("F:\WORK_Oxagile\INTERN\Datasets\COCO\\", img_size_transform,
                                        train_bool=False,
                                        shuffle_test=True,
                                        batch_size=batch_size)

    for i, data in enumerate(dataset):
        images, targets = data[0], data[1]

        with torch.no_grad():
            outputs = model(images)
            criterion = yolo_loss.yoloLoss(num_classes, device="cpu", cuda=False)
            loss_total, loss_coordinates, loss_confidence, loss_classes = criterion(outputs, targets)

        gt_dict = {0: "Person",
                   1: "Car",
                   2: "Bird",
                   3: "Cat",
                   4: "Dog"}
        color_dict = {0: (0.3, 0.7, 0.7),
                      1: (0.6, 0.2, 0.3),
                      2: (0.3, 0.3, 0.5),
                      3: (0.5, 0.5, 0.2),
                      4: (0.1, 0.7, 0.1)}

        conf = 0.10
        nms = 0.5
        print("Confidence threshold: ", conf)
        print("NMS threshold: ", nms)
        post_images = post_processing.draw_predictions(images, outputs,
                                                       gt_classes_dict=gt_dict,
                                                       color_dict=color_dict,
                                                       confidence_threshold=conf,
                                                       nms_threshold=nms)

        if len(targets) != 0:
            post_images = post_processing.draw_gt_boxes(post_images, targets)

        for j, post_image in enumerate(post_images):
            # cv2.imshow("Post", post_image)
            cv2.imwrite("F:\WORK_Oxagile\INTERN\ImageSegmentation\small\predicted_after70//"+"no_person_SCORE_lessthresh2_rgb_" + str(i) + "_" + str(j) + ".jpg",
                        cv2.cvtColor(post_image * 255, cv2.COLOR_RGB2BGR))

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

        if i >= 20:
            break
