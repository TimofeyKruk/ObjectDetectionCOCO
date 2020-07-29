import yolo
import numpy as np
import torch.nn as nn
import data_preparation
from matplotlib import cm
from PIL import Image
from src import post_processing
import torch
import math
import cv2
import matplotlib.pyplot as plt
import matplotlib.ticker as plticker


# ___SMALL___
def load_model(PATH, class_number=95):
    model = yolo.modelYOLO(num_classes=class_number)
    model.load_state_dict(torch.load(PATH))
    print("Model was loaded from file: ", PATH)
    return model



def heatmap_on_image(image, cam):
    color_map = cm.get_cmap("hsv")
    heatmap = color_map(cam)

    if image is torch.Tensor:
        image = Image.fromarray((np.transpose(image.numpy() * 255, (1, 2, 0))).astype(np.uint8))
    else:  # numpy array
        image = Image.fromarray((image * 255).astype(np.uint8))

    heatmap_transparent = heatmap.copy()
    heatmap_transparent[:, :, 3] = 0.4

    heatmap = Image.fromarray((heatmap * 255).astype(np.uint8))
    heatmap_transparent = Image.fromarray((heatmap_transparent * 255).astype(np.uint8))

    # Applying heatmap transparent on image
    heatmap_image = Image.new("RGBA", image.size)
    heatmap_image = Image.alpha_composite(heatmap_image, image.convert("RGBA"))
    heatmap_image = Image.alpha_composite(heatmap_image, heatmap_transparent)

    return heatmap, heatmap_image


# __________________________________COPY OF YOLO LOSS_________________________________________
# __________________________________COPY OF YOLO LOSS_________________________________________
class yoloLoss(nn.modules.loss._Loss):
    def __init__(self, num_classes,
                 device,
                 anchors=[(1.3221, 1.73145),
                          (3.19275, 4.00944),
                          (5.05587, 8.09892),
                          (9.47112, 4.84053),
                          (11.2364, 10.0071)],
                 cell_size=32,
                 cuda=True,
                 coord_scale=1.0,
                 noobject_scale=1.0,
                 object_scale=3.0,
                 class_scale=2.0,
                 threshold=0.6) -> None:
        super().__init__()

        self.device = device
        self.num_classes = num_classes
        self.num_anchors = len(anchors)
        self.anchors = torch.Tensor(anchors)
        self.cell_size = cell_size
        self.cuda = cuda

        self.coord_scale = coord_scale
        self.noobject_scale = noobject_scale
        self.object_scale = object_scale
        print("____Object scale in loss: ", self.object_scale)
        self.class_scale = class_scale
        self.threshold = threshold

    def forward(self, output, target, np_image):
        '''
        Calculates loss between model output and ground truth target.
        :param self:
        :param output: the output from model
        :param target: the ground truth for an image
        :return: loss: the calculated loss
        '''
        batch_size = output.data.size(0)
        height = output.data.size(2)
        width = output.data.size(3)

        # Fetching x,y,w,h,confidence,class
        output = output.view(batch_size, self.num_anchors, -1, height * width)
        coordinates = torch.zeros_like(output[:, :, :4, :])

        coordinates[:, :, :2, :] = output[:, :, :2, :].sigmoid()
        coordinates[:, :, 2:4, :] = output[:, :, 2:4, :]
        confidence = output[:, :, 4, :].sigmoid()
        classes = output[:, :, 5:, :].contiguous().view(batch_size * self.num_anchors, self.num_classes, height * width)
        classes = classes.transpose(1, 2).contiguous().view(-1, self.num_classes)

        # Create prediction boxes and cell grid
        predicted_boxes = torch.FloatTensor(batch_size * self.num_anchors * height * width, 4)
        lin_x = torch.arange(0, width, step=1).repeat(height, 1).view(height * width)
        lin_y = torch.arange(0, height, step=1).repeat(width, 1).t().contiguous().view(height * width)
        anchor_width = self.anchors[:, 0].contiguous().view(self.num_anchors, 1)
        anchor_height = self.anchors[:, 1].contiguous().view(self.num_anchors, 1)

        # Switching to CUDA
        if torch.cuda.is_available() and self.cuda:
            predicted_boxes = predicted_boxes.to(self.device)
            lin_x = lin_x.to(self.device)
            lin_y = lin_y.to(self.device)
            anchor_width = anchor_width.to(self.device)
            anchor_height = anchor_height.to(self.device)

        # Calculating precise coordinates of boxes corresponding to the whole image
        # TODO: I have deleted .detach for cpu, but will I need it for GPU?
        predicted_boxes[:, 0] = (coordinates[:, :, 0].detach() + lin_x).view(-1)
        predicted_boxes[:, 1] = (coordinates[:, :, 1].detach() + lin_y).view(-1)
        predicted_boxes[:, 2] = (coordinates[:, :, 2].detach().exp() * anchor_width).view(-1)
        predicted_boxes[:, 3] = (coordinates[:, :, 3].detach().exp() * anchor_height).view(-1)

        # _______________________________Part to visualize predictions:____________________________________

        # COORDINATES
        fig = plt.figure(figsize=(8, 8))
        ax = fig.add_subplot(111)
        fig.subplots_adjust(left=0, right=1, bottom=0, top=1)

        loc = plticker.MultipleLocator(base=self.cell_size)
        ax.xaxis.set_major_locator(loc)
        ax.yaxis.set_major_locator(loc)

        ax.grid(which="major", axis="both", linestyle="-")
        ax.imshow(np_image)

        anchor_n = 2
        plt.title("Predicted boxes coordinates for anchor: " + str(anchor_n))
        print("Number of predicted boxes: ", len(predicted_boxes))
        for box in predicted_boxes[anchor_n * 169:(anchor_n + 1) * 169]:
            x = box[0].item()
            y = box[1].item()
            w = box[2].item()
            h = box[3].item()
            # print("Drawing: ", "({},{},{},{})".format(x, y, w, h))
            ax.text(int(x) * self.cell_size, int(y) * self.cell_size,
                    "({:.1f},{:.1f}\n,{:.1f},{:.1f})".format(x, y, w, h), color="w", ha="left", va="top")
        plt.show()
        # _______________________________Part to visualize predictions____________________________________

        # Receiving target values
        coordinates_mask, confidence_mask, classes_mask, t_coord, t_conf, t_classes = self.build_targets(
            predicted_boxes, target, height, width)

        # ______________________Drawing masks___________________________

        # ______________________________________________________________

        coordinates_mask = coordinates_mask.expand_as(t_coord)

        t_classes = t_classes[classes_mask].view(-1).long()
        classes_mask = classes_mask.view(-1, 1).repeat(1, self.num_classes)

        confidence_mask = confidence_mask.sqrt()
        classes = classes[classes_mask].view(-1, self.num_classes)

        # Losses
        lossMSE = nn.MSELoss()
        lossCE = nn.CrossEntropyLoss()

        self.loss_coordinates = lossMSE(coordinates_mask * coordinates, coordinates_mask * t_coord)
        self.loss_coordinates *= self.coord_scale

        self.loss_confidence = lossMSE(confidence_mask * confidence, confidence_mask * t_conf)
        self.loss_classes = self.class_scale * 2 * lossCE(classes, t_classes)

        self.loss_total = self.loss_confidence + self.loss_coordinates + self.loss_classes

        return self.loss_total, self.loss_coordinates, self.loss_confidence, self.loss_classes

    def build_targets(self, predicted_boxes, target, height, width):
        '''
        #TODO: Write the function description
        :param predicted_boxes: tensor(all_boxes,4) <- boxes predicted by a model
        :param target: ground truth
        :param height: int <-height height
        :param width:
        :return: coordinates_mask, confidence_mask, classes_mask, t_coord, t_conf, t_classes <- masks
        '''
        batch_size = len(target)
        # Masks initialization with ones
        coordinates_mask = torch.zeros(batch_size, self.num_anchors, 1, height * width, requires_grad=False).type(
            torch.ByteTensor)
        confidence_mask = self.noobject_scale * torch.ones(batch_size, self.num_anchors, height * width,
                                                           requires_grad=False)
        classes_mask = torch.zeros(batch_size, self.num_anchors, height * width, requires_grad=False).type(
            torch.BoolTensor)
        t_coord = torch.zeros(batch_size, self.num_anchors, 4, height * width, requires_grad=False)
        t_conf = torch.zeros(batch_size, self.num_anchors, height * width, requires_grad=False)
        t_classes = torch.zeros(batch_size, self.num_anchors, height * width, requires_grad=False).type(
            torch.ByteTensor)

        # Switching again to cuda
        if torch.cuda.is_available() and self.cuda:
            t_coord = t_coord.to(self.device)
            t_classes = t_classes.to(self.device)
            t_conf = t_conf.to(self.device)
            confidence_mask = confidence_mask.to(self.device)
            classes_mask = classes_mask.to(self.device)
            coordinates_mask = coordinates_mask.to(self.device)

        # Adding two zeros for anchors
        extended_anchors = torch.cat([torch.zeros_like(self.anchors), self.anchors], 1)

        for instance in range(batch_size):
            # Checking if there is info for instance
            if len(target[instance]) == 0:
                continue

            current_predicted_boxes = predicted_boxes[instance * (self.num_anchors * height * width):
                                                      (instance + 1) * (self.num_anchors * height * width)]

            gt_boxes = torch.zeros(len(target[instance]), 4)
            if torch.cuda.is_available() and self.cuda:
                gt_boxes.cuda()

            # Getting boxes
            for i, annotation in enumerate(target[instance]):
                # x and y centers
                gt_boxes[i, 0] = (annotation[0] + annotation[2] / 2)
                gt_boxes[i, 1] = (annotation[1] + annotation[3] / 2)
                # height and width
                gt_boxes[i, 2] = (annotation[2])
                gt_boxes[i, 3] = (annotation[3])
            # Important to scale gt_boxes:
            gt_boxes = gt_boxes / self.cell_size

            # Confidence mask elements set to true if predictions are greater than threshold (iou >thresh)
            iou_gt_predicted = boxes_iou(gt_boxes, current_predicted_boxes, self.device)

            temp_mask = (iou_gt_predicted > self.threshold).sum(0) >= 1
            confidence_mask[instance][temp_mask.view_as(confidence_mask[instance])] = 0

            # Searching for the best anchor for each ground truth box
            gt_hw_boxes = gt_boxes.clone().detach()
            gt_hw_boxes[:, :2] = 0
            iou_hwgt_anchors = boxes_iou(gt_hw_boxes, extended_anchors, self.device)
            _, best_anchors = iou_hwgt_anchors.max(1)

            # Setting masks and target value for each ground truth
            for i, annotation in enumerate(target[instance]):
                gi = int(min(width - 1, max(0, gt_boxes[i, 0])))
                gj = int(min(height - 1, max(0, gt_boxes[i, 1])))
                best_anchor = best_anchors[i].item()

                iou = iou_gt_predicted[i][best_anchor * height * width + gj * width + gi]

                coordinates_mask[instance][best_anchor][0][gj * width + gi] = 1
                classes_mask[instance][best_anchor][gj * width + gi] = 1

                confidence_mask[instance][best_anchor][gj * width + gi] = self.object_scale

                t_coord[instance][best_anchor][0][gj * width + gi] = gt_boxes[i, 0] - gi
                t_coord[instance][best_anchor][1][gj * width + gi] = gt_boxes[i, 1] - gj
                t_coord[instance][best_anchor][2][gj * width + gi] = math.log(
                    max(gt_boxes[i, 2], 1.0) / self.anchors[best_anchor, 0])
                t_coord[instance][best_anchor][3][gj * width + gi] = math.log(
                    max(gt_boxes[i, 3], 1.0) / self.anchors[best_anchor, 1])

                t_conf[instance][best_anchor][gj * width + gi] = iou
                t_classes[instance][best_anchor][gj * width + gi] = int(annotation[4])

        return coordinates_mask, confidence_mask, classes_mask, t_coord, t_conf, t_classes


def boxes_iou(boxes1, boxes2, device):
    b1x1, b1y1 = (boxes1[:, :2].detach().cpu() - (boxes1[:, 2:4].detach().cpu() / 2)).split(1, 1)
    b1x2, b1y2 = (boxes1[:, :2].detach().cpu() + (boxes1[:, 2:4].detach().cpu() / 2)).split(1, 1)

    b2x1, b2y1 = (boxes2[:, :2].detach().cpu() - (boxes2[:, 2:4].detach().cpu() / 2)).split(1, 1)
    b2x2, b2y2 = (boxes2[:, :2].detach().cpu() + (boxes2[:, 2:4].detach().cpu() / 2)).split(1, 1)

    dx = (b1x2.min(b2x2.t()) - b1x1.max(b2x1.t())).clamp(min=0)
    dy = (b1y2.min(b2y2.t()) - b1y1.max(b2y1.t())).clamp(min=0)
    intersections = dx * dy

    area1 = (b1x2 - b1x1) * (b1y2 - b1y1)
    area2 = (b2x2 - b2x1) * (b2y2 - b2y1)
    unions = area1 + area2.t() - intersections

    intersections = intersections.to(device)
    unions = unions.to(device)

    return intersections / unions


# __________________________________COPY OF YOLO LOSS_________________________________________
# __________________________________COPY OF YOLO LOSS_________________________________________


if __name__ == '__main__':
    print("Checking loss function!")
    PATH = "F:\WORK_Oxagile\INTERN\ImageSegmentation\small\SMALL_SavedModelWeights6_after15_after20"
    print("Model PATH: ", PATH)
    num_classes = 5
    batch_size = 1
    img_size_transform = 32 * 13
    cell_size = 32

    # Loading model from memory
    model = load_model(PATH, num_classes)
    dataset = data_preparation.loadCOCO("F:\WORK_Oxagile\INTERN\Datasets\COCO\\", img_size_transform,
                                        train_bool=True,
                                        shuffle=False,
                                        batch_size=batch_size,
                                        no_person=False)

    for i, data in enumerate(dataset):
        images, targets = data[0], data[1]

        model.eval()
        with torch.no_grad():
            outputs = model(images)
            criterion = yoloLoss(num_classes, device="cpu", cuda=False)
            np_image = np.transpose(images[0].numpy(), (1, 2, 0))

            # Working with loss
            loss_total, loss_coordinates, loss_confidence, loss_classes = criterion(outputs, targets, np_image)

            print("Total: ", loss_total)
            print("Coord: ", loss_coordinates)
            print("Conf: ", loss_confidence)
            print("Classes: ",loss_classes)
        # Working with loss visualization

        # _______________________________ Drawing predictions part________________________________
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
            cv2.imwrite("F:\WORK_Oxagile\INTERN\ImageSegmentation\small//v15_check_loss//"
                        + "rgb_" + str(i) + "_" + str(j) + ".jpg",
                        cv2.cvtColor(post_image * 255, cv2.COLOR_RGB2BGR))
            plt.imshow(post_image)
            plt.show()
        # _______________________________ Drawing predictions part________________________________

        if i >= 5:
            break
