import torch
import yolo
import data_preparation
import matplotlib.pyplot as plt
import numpy as np


def load_model(PATH, class_number=95):
    model = yolo.modelYOLO(num_classes=class_number)
    model.load_state_dict(torch.load(PATH))
    return model


if __name__ == '__main__':
    print("Predicting object boxes started!")
    PATH = "F:\WORK_Oxagile\INTERN\ImageSegmentation\SavedModelWeights3after21"
    num_classes = 95
    batch_size = 4
    img_size_transform = 448

    # Loading model from memory
    model = load_model(PATH, num_classes)

    train = data_preparation.loadCOCO("F:\WORK_Oxagile\INTERN\Datasets\COCO\\", img_size_transform, train_bool=False,
                                      batch_size=batch_size)

    for data in train:
        images, targets = data[0], data[1]
        np_image=np.transpose(images.numpy()[3, :, :, :], (1, 2, 0))
        print(np_image)
        print("Max: ",np.max(np_image))
        plt.imshow(np_image)
        plt.show()
        with torch.no_grad():
            outputs = model(images)

        out = outputs[3].view(5, 100, 14, 14)
        print("Outputs[3].view.shape", out.shape)

        # for i,anchor in enumerate(out):
        #     for position in range(196):
        #         #print("Position: ",position," confidence: ",anchor[4,position])
        #         if anchor[4,position]>-10:
        #             print("max class in confident position: ",position, torch.max(anchor[5:,position],dim=0)[1])
        #     #print("Anchor: ",i,", Confidence: ",anchor[4],", class: ",max(anchor[5:]))

        for i in range(14):
            for j in range(14):
                print("Position: i", i, " j", j)
                for anchor in range(5):
                    if out[anchor, 4, i, j].sigmoid() > 0.02:
                        print("Anchor: ", anchor, ", confidence: ", out[anchor, 4, i, j].sigmoid(), ", class: ",
                              torch.argmax(torch.softmax(out[anchor, 5:, i, j], dim=0)))

        break
