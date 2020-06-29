import torch
import torch.nn as nn
import math


class yoloLoss(nn.modules.loss._Loss):
    def __init__(self, num_classes,
                 anchors,
                 cell_size=32,
                 coord_scale=0.1,
                 noobject_scale=1.0,
                 object_scale=0.5,
                 class_scale=1.0,
                 threshold=0.6,
                 reduction: str = "mean") -> None:
        super().__init__(reduction)

        self.num_classes = num_classes
        self.num_anchors = len(anchors)
        self.anchors = torch.Tensor(anchors)
        self.cell_size = cell_size
        self.reduction = reduction

        self.coord_scale = coord_scale
        self.noobject_scale = noobject_scale
        self.object_scale = object_scale
        self.class_scale = class_scale
        self.threshold = threshold

    def forward(self, output, target):
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
        classes = output[:, :, 5:, :].contiguous().view(batch_size * self.num_anchors, self.num_classes,
                                                        height * width)
        classes = classes.transpose(1, 2).contiguous().view(-1, self.num_classes)

        # Create prediction boxes and cell grid
        predicted_boxes = torch.FloatTensor(batch_size * self.num_anchors * height * width, 4)
        lin_x = torch.range(0, width - 1).repeat(height, 1).view(height * width)
        lin_y = torch.range(0, height - 1).repeat(width, 1).t().contiguous().view(height * width)
        anchor_width = self.anchors[:, 0].contiguous().view(self.num_anchors, 1)
        anchor_height = self.anchors[:, 1].contiguous().view(self.num_anchors, 1)

        # Switching to CUDA in any case (if possible)
        if torch.cuda.is_available():
            predicted_boxes = predicted_boxes.cuda()
            lin_x = lin_x.cuda()
            lin_y = lin_y.cuda()
            anchor_width = anchor_width.cuda()
            anchor_height = anchor_height.cuda()

        # Calculating precise coordinates of boxes corresponding to the whole image
        predicted_boxes[:, 0] = (coordinates[:, :, 0].detach() + lin_x).view(-1)
        predicted_boxes[:, 1] = (coordinates[:, :, 1].detach() + lin_y).view(-1)
        predicted_boxes[:, 2] = (coordinates[:, :, 2].detach().exp() * anchor_width).view(-1)
        predicted_boxes[:, 3] = (coordinates[:, :, 3].detach().exp() * anchor_height).view(-1)

        predicted_boxes = predicted_boxes.cpu()

        # Receiving target values
        coordinates_mask, confidence_mask, classes_mask, t_coord, t_conf, t_classes = self.build_targets(
            predicted_boxes, target, height, width)

        coordinates_mask.expand_as(t_coord)
        print("Coord mask shape", coordinates_mask.shape)

        t_classes = t_classes[classes_mask].view(-1).long()
        classes_mask = classes_mask.view(-1, 1).repeat(1, self.num_classes)

        # Switching again to cuda
        if torch.cuda.is_available():
            t_coord.cuda()
            t_classes.cuda()
            t_conf.cuda()
            confidence_mask.cuda()
            classes_mask.cuda()
            coordinates_mask.cuda()

        confidence_mask = confidence_mask.sqrt()
        classes = classes[classes_mask].view(-1, self.num_classes)

        #Losses
        lossMSE=nn.MSELoss()
        lossCE=nn.CrossEntropyLoss()

        self.loss_coordinates=self.coord_scale*lossMSE(coordinates_mask*coordinates,coordinates_mask*t_coord)
        self.loss_confidence=lossMSE(confidence_mask*confidence,confidence_mask*t_conf)
        self.loss_classes=self.class_scale*2*lossCE(classes,t_classes)

        self.loss_total=self.loss_confidence+self.loss_coordinates+self.loss_classes

        return self.loss_total, self.loss_coordinates,self.loss_confidence, self.loss_classes












    def build_target(self, predicted_boxes, target, height, width):
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
        coordinates_mask = torch.zeros(batch_size, self.num_anchors, 1, height * width, requires_grad=False)
        confidence_mask = self.noobject_scale * torch.ones(batch_size, self.num_anchors, height * width,
                                                           requires_grad=False)
        classes_mask = torch.zeres(batch_size, self.num_anchors, height * width, requires_grad=False).byte()
        t_coord = torch.zeros(batch_size, self.num_anchors, 4, height * width, requires_grad=False)
        t_conf = torch.zeros(batch_size, self.num_anchors, height * width, requires_grad=False)
        t_classes = torch.zeros(batch_size, self.num_anchors, height * width, requires_grad=False)

        # Adding two zeros for anchors
        extended_anchors = torch.cat([torch.zeros_like(self.anchors), self.anchors], 1)

        for instance in range(batch_size):
            # Checking if there is info for instance
            if len(target[instance]) == 0:
                continue

            current_predicted_boxes = predicted_boxes[instance * (self.num_anchors * height * width):
                                                      (instance + 1) * (self.num_anchors * height * width)]

            gt_boxes = torch.zeros(len(target[instance]), 4)

            # Getting boxes
            for i, annotation in enumerate(target[instance]):
                # x center
                gt_boxes[i, 0] = (annotation["bbox"][0] + annotation["bbox"][2] / 2) / self.cell_size
                # y center
                gt_boxes[i, 1] = (annotation["bbox"][1] + annotation["bbox"][3] / 2) / self.cell_size
                # height and width
                gt_boxes[i, 2] = (annotation["bbox"][2]) / self.cell_size
                gt_boxes[i, 3] = (annotation["bbox"][3]) / self.cell_size

            # Confidence mask elements set to true if predictions are greater than threshold (iou >thresh)
            iou_gt_predicted = boxes_iou(gt_boxes, current_predicted_boxes)
            print("Ground Truth boxes shape ", gt_boxes.shape)
            print("Predicted boxes shape ", predicted_boxes.shape)
            temp_mask = (iou_gt_predicted > self.threshold).sum(0) >= 1
            print("Temp mask shape ", temp_mask.shape, temp_mask)
            # TODO: Why is there not a 1?
            confidence_mask[instance][temp_mask.view_as(confidence_mask[instance])] = 0

            # Searching for the best anchor for each groung truth box
            gt_hw_boxes = gt_boxes.copy()
            gt_hw_boxes[:, :2] = 0
            iou_hwgt_anchors = boxes_iou(gt_hw_boxes, extended_anchors)
            _, best_anchors = iou_hwgt_anchors.max(1)

            # Setting masks and target value for each ground truth
            for i, annotation in enumerate(target[instance]):
                gi = min(width - 1, max(0, gt_boxes[i, 0]))
                gj = min(height - 1, max(0, gt_boxes[i, 1]))
                best_anchor = best_anchors[i]

                iou = iou_gt_predicted[i][best_anchor * height * width + gj * height + gi]

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
                t_classes[instance][best_anchor][gj * width + gi] = int(annotation["category_id"])

        return coordinates_mask, confidence_mask, classes_mask, t_coord, t_conf, t_classes


def boxes_iou(boxes1, boxes2):
    b1x1, b1y1 = (boxes1[:, :2] - (boxes1[:, 2:4] / 2)).split(1, 1)
    b1x2, b1y2 = (boxes1[:, :2] + (boxes1[:, 2:4] / 2)).split(1, 1)

    b2x1, b2y1 = (boxes2[:, :2] - (boxes2[:, 2:4] / 2)).split(1, 1)
    b2x2, b2y2 = (boxes2[:, :2] + (boxes2[:, 2:4] / 2)).split(1, 1)

    dx = (b1x2.min(b2x2.t()) - b1x1.max(b2x1.t())).clamp(min=0)
    print("yolo loss, dx shape", dx.shape)
    dy = (b1y2.min(b2y2.t()) - b1y1.max(b2y1.t())).clamp(min=0)
    intersections = dx * dy

    area1 = (b1x2 - b1x1) * (b1y2 - b1y1)
    area2 = (b2x2 - b2x1) * (b2y2 - b2y1)
    unions = area1 + area2.t() - intersections

    return intersections / unions


if __name__ == '__main__':
    print(boxes_iou(torch.tensor([[8.0, 8.0, 6.0, 6.0], [10.0, 10.0, 4.0, 4.0], [8.0, 8.0, 6.0, 6.0]]),
                    torch.tensor([[10.0, 10.0, 4.0, 4.0], [8.0, 8.0, 6.0, 6.0]])).sum(0))

    # build target checking
    # myloss= yoloLoss(80,anchors=[(1.3221, 1.73145),
    #                                          (3.19275, 4.00944),
    #                                          (5.05587, 8.09892),
    #                                          (9.47112, 4.84053),
    #                                          (11.2364, 10.0071)])
    #
    # myloss.build_target(torch.tensor([[8.0, 8.0, 6.0, 6.0]]), torch.tensor([[10.0, 10.0, 4.0, 4.0]]),1,1)
