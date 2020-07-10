import cv2
import torch


def draw_predictions(image, output, image_size=416):
    predictions = post_processing(output, gt_classes=None,
                                  image_size=image_size,
                                  confidence_threshold=0.2,
                                  nms_threshold=0.6)
    for prediction in predictions:
        xmin = int(max(prediction[0], 0))
        ymin = int(max(prediction[1], 0))

        xmax = int(min(prediction[0] + prediction[2], image_size))
        ymax = int(min(prediction[1] + prediction[3], image_size))

        cv2.rectangle(image, (xmin, ymin), (xmax, ymax), (200, 200, 0), thickness=4)

        text_size = cv2.getTextSize(prediction[5] + ":%.2f" % prediction[4], cv2.FONT_HERSHEY_PLAIN, fontScale=1, thickness=1)[0]
        cv2.rectangle(image, (xmin, ymin), (xmin + text_size[0] + 3, ymin + text_size[1] + 4), (100, 100, 0), -1)
        cv2.putText(image,
                    prediction[5] + ":%.2f" % prediction[4],
                    (xmin, ymin + text_size[1] + 4),
                    cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255), 1)
        print("Object: {}, bounding box: ({},{}),({},{})".format(prediction[5],xmin,ymin,xmax,ymax))
        return image


def post_processing(outputs,
                    gt_classes,
                    image_size=416,
                    anchors=[(1.3221, 1.73145),
                             (3.19275, 4.00944),
                             (5.05587, 8.09892),
                             (9.47112, 4.84053),
                             (11.2364, 10.0071)],
                    confidence_threshold=0.25,
                    nms_threshold=0.6):
    """
    Creating array of boxes. Where one box stands for one prediction with object coordinates,
    object class, confidence score. Representation of model output.

    :param outputs: tensor from last layer of yolo_v2 model
    :param image_size: height(width) of square image
    :param gt_classes:
    :param anchors:
    :param confidence_threshold: threshold used to classify whether there is an object or not
    :param nms_threshold: threshold for non max suppression for the same object detected
    :return: array of predicted borders for objects detected for future drawing on an image
    """
    num_anchors = len(anchors)
    anchors = torch.Tensor(anchors)

    # Fetching only data
    outputs = outputs.data

    # If there is only one instance then add "batch" dimension
    if outputs.dim() == 3:
        outputs.unsquize_(0)

    batch_size = outputs.size(0)
    height = outputs.size(2)
    width = outputs.size(3)

    # Meshgrid and separate anchors coordinates:
    lin_x = torch.linspace(0, width - 1, width).repeat(height, 1).view(height * width)
    lin_y = torch.linspace(0, height - 1, height).repeat(width, 1).t().contiguous().view(height * width)
    anchor_w = anchors[:, 0].contiguous().view(1, num_anchors, 1)
    anchor_h = anchors[:, 1].contiguous().view(1, num_anchors, 1)

    outputs = outputs.view(batch_size, num_anchors, -1, height * width)
    # Calculating boxes centers, size according to image
    outputs[:, :, 0, :].sigmoid_().add_(lin_x).div_(width)
    outputs[:, :, 1, :].sigmoid_().add_(lin_y).div_(height)
    outputs[:, :, 2, :].exp_().mul_(anchor_w).div_(width)
    outputs[:, :, 3, :].exp_().mul_(anchor_h).div_(height)
    outputs[:, :, 4, :].sigmoid_()

    with torch.no_grad():
        classes_scores = torch.nn.functional.softmax(outputs[:, :, 5:, :], dim=2)

    classes_max, classes_max_idx = torch.max(classes_scores, dim=2)
    classes_max_idx = classes_max_idx.float()

    # Sigmoid(t)=probability*confidence(iou)
    classes_max.mul_(outputs[:, :, 4, :])

    score_thresholded_mask = (classes_max > confidence_threshold).view(-1)

    predicted_boxes = []
    if score_thresholded_mask.sum() == 0:
        for i in range(batch_size):
            predicted_boxes.append(torch.Tensor([]))
    else:
        coordinates = outputs.transpose(2, 3)[..., 0:4]

        coordinates = coordinates[score_thresholded_mask[..., None].expand_as(coordinates)].view(-1, 4)

        scores = classes_max[score_thresholded_mask]
        indexes = classes_max_idx[score_thresholded_mask]
        detections = torch.cat([coordinates, scores[:, None], indexes[:, None]], dim=1)

        cells_per_batch = num_anchors * height * width
        slices = [slice(cells_per_batch * i, cells_per_batch * (i + 1)) for i in range(batch_size)]
        detections_per_batch = torch.IntTensor([score_thresholded_mask[s].int().sum() for s in slices])
        split_indexes = torch.cumsum(detections_per_batch, dim=0)

        # Grouping detections of one image in a batch
        start = 0
        for end in split_indexes:
            predicted_boxes.append(detections[start:end])
            start = end

    # ____Non-max suppression____
    selected_boxes = []
    for boxes in predicted_boxes:
        if boxes.numel() == 0:
            return boxes

        points = boxes[:, 0:2]
        edges = boxes[:, 2:4]
        border_boxes = torch.cat([(points - edges / 2), (points + edges / 2)], dim=1)
        scores = boxes[:, 4]

        # Sorting in descending order to choose most confident prediction
        scores, order = scores.sort(dim=0, descending=True)
        x1, y1, x2, y2 = border_boxes[order].split(1, 1)

        # IoU for any pair of boxes
        dx = (x2.min(x2.t()) - x1.max(x1.t())).clamp(min=0)
        dy = (y2.min(y2.t()) - y1.max(y1.t())).clamp(min=0)

        intersections = dx * dy
        areas = (x2 - x1) * (y2 - y1)

        # Broadcasting help to create 2d array
        unions = (areas + areas.t()) - intersections
        ious = intersections / unions

        # Upper diagonal matrix, excluding diagonal (fill with 0)
        conflicting_boxes_mask = (ious > nms_threshold).triu(diagonal=1)

        keeping_boxes = conflicting_boxes_mask.sum(dim=0)

        len_keeping = len(keeping_boxes) - 1
        for i in range(1, len_keeping):
            if keeping_boxes[i] > 0:
                keeping_boxes -= conflicting_boxes_mask[i]

        keeping_boxes = (keeping_boxes == 0)
        selected_boxes.append(boxes[order][keeping_boxes[:, None].expand_as(boxes)].view(-1, 6).contiguous())

        # Scaling boxes according to image
        final_boxes = []
        for boxes in selected_boxes:
            if boxes.dim() == 0:
                final_boxes.append([])
            else:
                boxes[:, 0:4] *= image_size
                boxes[:, 0] -= boxes[:, 2] / 2
                boxes[:, 1] -= boxes[:, 3] / 2

                final_boxes.append([[box[0].item(),
                                     box[1].item(),
                                     box[2].item(),
                                     box[3].item(),
                                     box[4].item(),
                                     box[5].item()] for box in boxes])

        return final_boxes
