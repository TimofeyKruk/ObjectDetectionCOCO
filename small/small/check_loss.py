import yolo
import numpy
import data_preparation
import yolo_loss
from src import post_processing
import torch
import cv2
import matplotlib.pyplot as plt

# ___SMALL___
def load_model(PATH, class_number=95):
    model = yolo.modelYOLO(num_classes=class_number)
    model.load_state_dict(torch.load(PATH))
    print("Model was loaded from file: ", PATH)
    return model


if __name__ == '__main__':
    print("Checking loss function!")
    PATH = "F:\WORK_Oxagile\INTERN\ImageSegmentation\small\SMALL_SavedModelWeights15no_residual_after40"
    print("Model PATH: ", PATH)
    num_classes = 5
    batch_size = 2
    img_size_transform = 32 * 13

    # Loading model from memory
    model = load_model(PATH, num_classes)
    dataset = data_preparation.loadCOCO("F:\WORK_Oxagile\INTERN\Datasets\COCO\\", img_size_transform,
                                        train_bool=False,
                                        shuffle_test=False,
                                        batch_size=batch_size,
                                        no_person=True)

    for i, data in enumerate(dataset):
        images, targets = data[0], data[1]

        model.eval()
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

        conf = 0.2
        nms = 0.6
        print("Confidence threshold: ", conf)
        print("NMS threshold: ", nms)

        if len(targets) != 0:
            post_images = post_processing.draw_gt_boxes(images, targets, gt_classes_dict=gt_dict)

        post_images = post_processing.draw_predictions(post_images, outputs,
                                                       gt_classes_dict=gt_dict,
                                                       color_dict=color_dict,
                                                       confidence_threshold=conf,
                                                       nms_threshold=nms)

        for j, post_image in enumerate(post_images):
            # cv2.imshow("Post", post_image)
            cv2.imwrite("F:\WORK_Oxagile\INTERN\ImageSegmentation\small//v12_predicted_after20//"
                        + "5rgb_" + str(i) + "_" + str(j) + ".jpg",
                        cv2.cvtColor(post_image * 255, cv2.COLOR_RGB2BGR))

            plt.imshow(post_image)
            plt.show()


        if i>5:
            break
