import torch
import yolo
import data_preparation
import matplotlib.pyplot as plt
import numpy as np
import yolo_loss
import cv2


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
    return model


if __name__ == '__main__':
    print("Predicting object boxes started!")
    PATH = "F:\WORK_Oxagile\INTERN\ImageSegmentation\SavedModelWeights4_after7_after28_after35"
    num_classes = 95
    batch_size = 2
    img_size_transform = 32 * 13

    # Loading model from memory
    model = load_model(PATH, num_classes)

    train = data_preparation.loadCOCO("F:\WORK_Oxagile\INTERN\Datasets\COCO\\", img_size_transform, train_bool=True,
                                      batch_size=batch_size)

    for data in train:
        images, targets = data[0], data[1]

        np_image = np.transpose(images.numpy()[1, :, :, :], (1, 2, 0))

        cv2.rectangle(np_image, (40, 40), (100, 100), (240, 130, 10), thickness=4)

        plt.imshow(np_image)
        plt.show()
        plt.imshow(np.transpose(images.numpy()[0, :, :, :], (1, 2, 0)))
        plt.show()

        with torch.no_grad():
            outputs = model(images)
            criterion = yolo_loss.yoloLoss(num_classes, device="cpu", cuda=False)
            loss_total, loss_coordinates, loss_confidence, loss_classes = criterion(outputs, targets)

        out = outputs[1].view(5, 100, 13, 13)
        print("Outputs[1].view.shape", out.shape)

        # for i,anchor in enumerate(out):
        #     for position in range(196):
        #         #print("Position: ",position," confidence: ",anchor[4,position])
        #         if anchor[4,position]>-10:
        #             print("max class in confident position: ",position, torch.max(anchor[5:,position],dim=0)[1])
        #     #print("Anchor: ",i,", Confidence: ",anchor[4],", class: ",max(anchor[5:]))

        for i in range(13):
            for j in range(13):
                print("Position: i", i, " j", j)
                for anchor in range(5):
                    if out[anchor, 4, i, j].sigmoid() > 0.1:
                        print("Anchor: ", anchor, ", confidence: ", out[anchor, 4, i, j].sigmoid(), ", class: ",
                              torch.argmax(torch.softmax(out[anchor, 5:, i, j], dim=0)))

        break
