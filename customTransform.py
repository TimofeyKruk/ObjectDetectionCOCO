import torch
import torchvision
import cv2

class Resize():
    def __init__(self,img_size) -> None:
        super().__init__()
        self.img_size=img_size

    def __call__(self, images, targets):
        print("Resize().__call__ was called!!!!!!!@!@!@!")


