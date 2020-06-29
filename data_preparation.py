from pycocotools.coco import COCO
import torchvision
from torch.utils.data import dataloader
from torchvision import transforms
import matplotlib.pyplot as plt
import numpy as np


def loadCOCO(train_bool=True, batch_size=1):
    """Loading train loaders of COCO detection dataset"""

    # Maybe later will add more transformations
    transform = transforms.Compose([transforms.ToTensor()])
    PATH = "F:\WORK_Oxagile\INTERN\Datasets\COCO\\"

    if train_bool is True:
        train = torchvision.datasets.CocoDetection(root=PATH + "images\\train2014\\train2014",
                                                   annFile=PATH + "annotations\\annotations_trainval2014\\annotations\\instances_train2014.json",
                                                   transform=transform)
        train_l = dataloader.DataLoader(train,
                                        batch_size=batch_size,
                                        shuffle=True,
                                        num_workers=0)
        return train_l
    else:
        test = torchvision.datasets.CocoDetection(root=PATH + "images\\test2014\\test2014",
                                                  annFile=PATH + "annotations\image_info_test2014\\annotations\\image_info_test2014.json",
                                                  transform=transform)
        test_l = dataloader.DataLoader(test,
                                       batch_size=batch_size,
                                       shuffle=True,
                                       num_workers=0)
        return test_l


if __name__ == '__main__':
    print("DATA preparation.py")
    # annFile = "F:\WORK_Oxagile\INTERN\Datasets\COCO\\annotations\image_info_test2014\\annotations\image_info_test2014.json"
    # img_keys = list(COCO(annFile).imgs.keys())
    # print(len(img_keys))

    train_l = loadCOCO(train_bool=True)
    print("Loader starts!")

    for batch in train_l:
        img, lab = batch
        npimg = np.transpose(img.numpy()[0, :, :, :], (1, 2, 0))
        plt.imshow(np.transpose(img.numpy()[0, :, :, :], (1, 2, 0)))
        plt.show()
        print("Label: ", lab[0].keys())
        break

    a = train_l.dataset.coco.getCatIds("banana")
    print(a)
