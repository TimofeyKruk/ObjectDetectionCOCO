from pycocotools.coco import COCO
import torchvision
from torch.utils.data import dataloader
import matplotlib.pyplot as plt
import numpy as np
from custom_transform import Resize
from torch.utils.data.dataloader import default_collate
import my_coco


## SMALL/SMALL

def my_collate(batch):
    items = list(zip(*batch))
    items[0] = default_collate(items[0])
    items[1] = list(items[1])
    return items


def loadCOCO(PATH, img_size=416, train_bool=True, batch_size=32, shuffle_test=False, no_person=False):
    """Loading train loaders of COCO detection dataset"""


    if train_bool is True:
        train = my_coco.CocoDetection(root=PATH + "images//train2014",
                                      annFile=PATH + "annotations//annotations_trainval2014//annotations//instances_train2014.json",
                                      transforms=Resize(img_size=img_size),
                                      no_person=no_person)
        print("___DataLoader started! Batch size: ", batch_size)
        train_l = dataloader.DataLoader(train,
                                        batch_size=batch_size,
                                        shuffle=True,
                                        collate_fn=my_collate,
                                        pin_memory=True,
                                        num_workers=2,
                                        drop_last=True)
        return train_l
    else:
        test = my_coco.CocoDetection(root=PATH + "images//val2014",
                                     annFile=PATH + "//annotations//annotations_trainval2014//annotations//instances_val2014.json",
                                     transforms=Resize(img_size=img_size),
                                     no_person=no_person)
        print("___DataLoader started!")
        test_l = dataloader.DataLoader(test,
                                       batch_size=batch_size,
                                       shuffle=shuffle_test,
                                       collate_fn=my_collate,
                                       pin_memory=True,
                                       num_workers=1,
                                       drop_last=True)
        return test_l


if __name__ == '__main__':
    print("DATA preparation.py")
    # annFile = "F:\WORK_Oxagile\INTERN\Datasets\COCO\\annotations\image_info_test2014\\annotations\image_info_test2014.json"
    # img_keys = list(COCO(annFile).imgs.keys())
    # print(len(img_keys))

    train_l = loadCOCO(train_bool=True)
    print("Loader starts!")

    step = 0
    print("Trying to print boxes:")
    for batch in train_l:
        img, lab = batch
        npimg = np.transpose(img.numpy()[0, :, :, :], (1, 2, 0))
        plt.imshow(npimg)

        coco = train_l.dataset.coco
        # coco.showAnns(lab)
        plt.show()

        # print(coco)
        # print("Label: ", lab[0].keys())

        print("HAHA@@@@@@@@@  ", step, npimg.shape)
        if step == 15:
            break
        else:
            step += 1

    # coco = train_l.dataset.coco
    #
    # catIds=coco.getCatIds(catNms=["person","dog","skateboard"])
    # print("CatIds: ",catIds)
    # imgIds=coco.getImgIds(catIds=catIds)
    # print("ImgIds: ",imgIds)
    # img=coco.loadImgs(imgIds[0])
    # print("Image: ",img)
    # print("Img URL: ",img[0]["coco_url"])
    #
    # print(type(img))
    # plt.imshow(img)
    # plt.show()

    #
    # a = train_l.dataset.coco.getCatIds("umbrella")
    # print(a)
