from torchvision.datasets.vision import VisionDataset
from PIL import Image
import os
import os.path
import matplotlib.pyplot as plt


class CocoCaptions(VisionDataset):
    """`MS Coco Captions <http://mscoco.org/dataset/#captions-challenge2015>`_ Dataset.

    Args:
        root (string): Root directory where images are downloaded to.
        annFile (string): Path to json annotation file.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.ToTensor``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        transforms (callable, optional): A function/transform that takes input sample and its target as entry
            and returns a transformed version.

    Example:

        .. code:: python

            import torchvision.datasets as dset
            import torchvision.transforms as transforms
            cap = dset.CocoCaptions(root = 'dir where images are',
                                    annFile = 'json annotation file',
                                    transform=transforms.ToTensor())

            print('Number of samples: ', len(cap))
            img, target = cap[3] # load 4th sample

            print("Image Size: ", img.size())
            print(target)

        Output: ::

            Number of samples: 82783
            Image Size: (3L, 427L, 640L)
            [u'A plane emitting smoke stream flying over a mountain.',
            u'A plane darts across a bright blue sky behind a mountain covered in snow',
            u'A plane leaves a contrail above the snowy mountain top.',
            u'A mountain that has a plane flying overheard in the distance.',
            u'A mountain view with a plume of smoke in the background']

    """

    def __init__(self, root, annFile, transform=None, target_transform=None, transforms=None):
        super(CocoCaptions, self).__init__(root, transforms, transform, target_transform)
        from pycocotools.coco import COCO
        self.coco = COCO(annFile)
        self.ids = list(sorted(self.coco.imgs.keys()))

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: Tuple (image, target). target is a list of captions for the image.
        """
        coco = self.coco

        img_id = self.ids[index]
        ann_ids = coco.getAnnIds(imgIds=img_id)
        anns = coco.loadAnns(ann_ids)
        target = [ann['caption'] for ann in anns]

        path = coco.loadImgs(img_id)[0]['file_name']

        img = Image.open(os.path.join(self.root, path)).convert('RGB')

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target

    def __len__(self):
        return len(self.ids)


class CocoDetection(VisionDataset):
    """`MS Coco Detection <http://mscoco.org/dataset/#detections-challenge2016>`_ Dataset.

    Args:
        root (string): Root directory where images are downloaded to.
        annFile (string): Path to json annotation file.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.ToTensor``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        transforms (callable, optional): A function/transform that takes input sample and its target as entry
            and returns a transformed version.
    """

    def __init__(self, root, annFile, transform=None, target_transform=None, transforms=None):
        super(CocoDetection, self).__init__(root, transforms, transform, target_transform)
        from pycocotools.coco import COCO
        self.coco = COCO(annFile)
        # @@ My code @@
        self.cat_ids = self.coco.getCatIds(catNms=["person", "car", "bird", "cat", "dog"])
        print("self.cat_ids: ", self.cat_ids)
        # self.ann_ids = self.coco.getAnnIds(catIds=self.cat_ids, iscrowd=None)
        # Set
        img_ids = set()
        for cat_id in self.cat_ids:
            for im_id in self.coco.getImgIds(catIds=cat_id):
                img_ids.add(im_id)

        img_ids = list(img_ids)
        print("self.img_ids: len ", len(img_ids))
        # self.all_ann = self.coco.loadAnns(self.ann_ids)
        # I have changed sorted() argument
        self.ids = list(sorted(img_ids))

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: Tuple (image, target). target is the object returned by ``coco.loadAnns``.
        """
        coco = self.coco
        img_id = self.ids[index]
        ann_ids = coco.getAnnIds(imgIds=img_id, catIds=self.cat_ids)

        target = coco.loadAnns(ann_ids)

        path = coco.loadImgs(img_id)[0]['file_name']

        img = Image.open(os.path.join(self.root, path)).convert('RGB')

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target

    def __len__(self):
        return len(self.ids)
