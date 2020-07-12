import torch
import yolo
import data_preparation
import matplotlib.pyplot as plt
import numpy as np
import yolo_loss
import cv2
import post_processing


def draw_boxes(img, target):
    image = np.transpose(img.numpy(), (1, 2, 0))
    image = image.astype("float32")
    print(image.shape)
    for object in target:
        x1 = int(object[0])
        y1 = int(object[1])
        x2 = int(object[0] + object[2])
        y2 = int(object[1] + object[3])
        im = np.zeros((215, 215, 3))
        cv2.rectangle(im, (40, 40), (130, 130), color=(0, 0, 0), thickness=5)
        plt.imshow(image)
        plt.show()
        cv2.imshow("Img", image)
        k = cv2.waitKey(0)  # 0==wait forever
        print("Object : ", object[4].item(), ", coordinates: ", x1, " ", y1, " ", x2, " ", y2)

    return image


def load_model(PATH, class_number=95):
    model = yolo.modelYOLO(num_classes=class_number)
    model.load_state_dict(torch.load(PATH))
    print("Model was loaded from file: ", PATH)
    return model


if __name__ == '__main__':
    print("Predicting object boxes started!")
    PATH = "F:\WORK_Oxagile\INTERN\ImageSegmentation\SavedModelWeights5_after14_after28"
    num_classes = 95
    batch_size =4
    img_size_transform = 32 * 13

    # Loading model from memory
    model = load_model(PATH, num_classes)

    train = data_preparation.loadCOCO("F:\WORK_Oxagile\INTERN\Datasets\COCO\\", img_size_transform,
                                      train_bool=False,
                                      batch_size=batch_size)

    for data in train:
        images, targets = data[0], data[1]

        with torch.no_grad():
            outputs = model(images)
            criterion = yolo_loss.yoloLoss(num_classes, device="cpu", cuda=False)
            loss_total, loss_coordinates, loss_confidence, loss_classes = criterion(outputs, targets)

        post_images = post_processing.draw_predictions(images, outputs)

        for post_image in post_images:
            #cv2.imshow("Post", post_image)
            plt.imshow(post_image)
            plt.show()

        # for i,anchor in enumerate(out):
        #     for position in range(196):
        #         #print("Position: ",position," confidence: ",anchor[4,position])
        #         if anchor[4,position]>-10:
        #             print("max class in confident position: ",position, torch.max(anchor[5:,position],dim=0)[1])
        #     #print("Anchor: ",i,", Confidence: ",anchor[4],", class: ",max(anchor[5:]))

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
