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
    PATH = "SavedModelWeights"
    num_classes = 95
    batch_size = 32
    img_size_transform = 448

    # Loading model from memory
    model = load_model(PATH, num_classes)

    test = data_preparation.loadCOCO("F:\WORK_Oxagile\INTERN\Datasets\COCO\\", img_size_transform, train_bool=False,
                                     batch_size=batch_size)

    for data in test:
        images, targets = data[0], data[1]

        plt.imshow(np.transpose(images.numpy()[6, :, :, :], (1, 2, 0)))
        plt.show()
        with torch.no_grad():
            outputs = model(images)

        out = outputs[6].view(5, 100, -1)
        print("Outputs[0].view.shape", out.shape)

        # for i,anchor in enumerate(out):
        #     for position in range(196):
        #         #print("Position: ",position," confidence: ",anchor[4,position])
        #         if anchor[4,position]>-10:
        #             print("max class in confident position: ",position, torch.max(anchor[5:,position],dim=0)[1])
        #     #print("Anchor: ",i,", Confidence: ",anchor[4],", class: ",max(anchor[5:]))

        for position in range(196):
            #print("Position: ", position)
            for anchor in range(5):
                if out[anchor][4][position]>-3:
                    print("position: ",position,", anchor: ", anchor, ", confidence: ", out[anchor][4][position], "boxes: x",
                      out[anchor][0][position])

        break
