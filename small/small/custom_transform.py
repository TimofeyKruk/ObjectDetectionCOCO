import cv2
import torch
import numpy as np
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt


## SMALL/SMALL


class Resize():
    def __init__(self, img_size) -> None:
        super().__init__()
        self.img_size = img_size

    def __call__(self, image, target):
        width, height = image.size
        # Resizing image to img_size:
        image = image.resize((self.img_size, self.img_size), resample=Image.NEAREST)

        width_ratio = float(self.img_size) / width
        height_ratio = float(self.img_size) / height

        boxes = []
        for annotation in target:
            # left upper corner resizing
            x1_resized = annotation["bbox"][0] * width_ratio
            y1_resized = annotation["bbox"][1] * height_ratio

            # width and height of a box resizing
            w_resized = annotation["bbox"][2] * width_ratio
            h_resized = annotation["bbox"][3] * height_ratio
            object_class = annotation["category_id"]

            # Transforming to 1-5:
            categoryId_change = {1: 0,
                                 3: 1,
                                 16: 2,
                                 17: 3,
                                 18: 4}
            object_class = categoryId_change[object_class]

            boxes.append([x1_resized, y1_resized, w_resized, h_resized, object_class])

        # ToTensor transform
        toTensor = transforms.ToTensor()
        image = toTensor(image)
        boxes = np.array(boxes, dtype=np.float32)

        if len(boxes) != 0:
            temp = []
            for box in boxes:
                temp.append(torch.tensor(box).unsqueeze(dim=0))
            # print("Len temp cust transf: ", len(temp))
            boxes = torch.cat(temp, dim=0)
        else:
            # TODO: What to do when there are no ground boxes?
            boxes = torch.tensor([])
            # print("___Zero number of GrTruth boxes detected:", boxes.size)
        # print("Custom transform boxes shape (cat,unsquize): ",image.shape,type(image))
        # print("After ToTensor() types: ",type(image),type(boxes),boxes.shape)
        return image, boxes
