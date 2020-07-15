import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt


def draw_gt_boxes(images, targets):
    gt_images = []
    for i, image in enumerate(images):
        height, width = image.shape[:2]
        for object in targets[i]:
            x1 = int(max(object[0], 0))
            y1 = int(max(object[1], 0))
            x2 = int(min(object[0] + object[2], width))
            y2 = int(min(object[1] + object[3], height))

            cv2.rectangle(image, (x1, y1), (x2, y2), color=(1, 0.43, 0), thickness=2)
            print("___GT___Object : ", object[4].item(), ", coordinates: ", x1, " ", y1, " ", x2, " ", y2)
            # text_size = cv2.getTextSize(str(object[4]) + ": gt_pr=1.0",
            #                             cv2.FONT_HERSHEY_PLAIN,
            #                             fontScale=1,
            #                             thickness=1)[0]
            # cv2.rectangle(image,
            #               (x1, y1),
            #               (x1 + text_size[0] + 3, y1 + text_size[1] + 4),
            #               (1, 0.5, 0.5),
            #               -1)
            # cv2.putText(image,
            #             str(object[4]) + ": gt_pr=1.0",
            #             (x1, y1 + text_size[1] + 4),
            #             cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255), 1)
        gt_images.append(image)
    return gt_images


def draw_predictions(images, outputs, gt_classes_dict=None, color_dict=None, image_size=416):
    predictions = post_processing(outputs,
                                  image_size=image_size,
                                  confidence_threshold=0.20,
                                  nms_threshold=0.6)
    post_images = []
    if len(predictions) != 0:
        for i, image in enumerate(images):
            print("NewImage")
            image = (np.transpose(image.numpy(), (1, 2, 0)))

            # TODO: Bug that better be fixed. Don't know how to fix it.
            np_image = np.zeros(image.shape, np.float32)
            np_image[2:-2, 2:-2, 0:3] = image[2:-2, 2:-2, 0:3]

            if len(predictions[i]) != 0:
                for prediction in predictions[i]:
                    xmin = int(max(prediction[0], 0))
                    ymin = int(max(prediction[1], 0))

                    xmax = int(min(prediction[0] + prediction[2], image_size))
                    ymax = int(min(prediction[1] + prediction[3], image_size))
                    print("Object: {}, bounding box: ({},{}),({},{})".format(prediction[5], xmin, ymin, xmax, ymax))

                    if color_dict is not None:
                        color = color_dict[prediction[5]]
                    else:
                        color = (0.2, 0.7, 0.5)

                    cv2.rectangle(np_image,
                                  (xmin, ymin),
                                  (xmax, ymax),
                                  color, 2)

                    text = (gt_classes_dict[prediction[5]] if gt_classes_dict != None else str(
                        prediction[5])) + ":%.2f" % prediction[4]
                    text_size = cv2.getTextSize(text,
                                                cv2.FONT_HERSHEY_PLAIN,
                                                fontScale=1,
                                                thickness=1)[0]
                    cv2.rectangle(np_image,
                                  (xmin, ymin),
                                  (xmin + text_size[0] + 3, ymin + text_size[1] + 4),
                                  color,
                                  -1)
                    cv2.putText(np_image,
                                text,
                                (xmin, ymin + text_size[1] + 4),
                                cv2.FONT_HERSHEY_PLAIN, 1, (1, 1, 1), 1)

            post_images.append(np_image)

    return post_images


def post_processing(outputs,
                    image_size=416,
                    anchors=[(1.3221, 1.73145),
                             (3.19275, 4.00944),
                             (5.05587, 8.09892),
                             (9.47112, 4.84053),
                             (11.2364, 10.0071)],
                    confidence_threshold=0.2,
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

    score_thresholded_mask = (classes_max > confidence_threshold)

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

        score_thresholded_mask_flattened = score_thresholded_mask.view(-1)
        detections_per_batch = torch.IntTensor([score_thresholded_mask_flattened[s].int().sum() for s in slices])
        split_indexes = torch.cumsum(detections_per_batch, dim=0)

        # Grouping detections of one image in a batch
        start = 0
        for end in split_indexes:
            predicted_boxes.append(detections[start:end])
            start = end

    temp_total = 0
    for boxes in predicted_boxes:
        temp_total += len(boxes)
    print("Amount of predictions before nms: ", temp_total)
    # ____Non-max suppression____
    selected_boxes = []
    for boxes in predicted_boxes:
        if boxes.numel() == 0:
            selected_boxes.append([])
            continue

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

        # len_keeping = len(keeping_boxes) - 1
        # for i in range(1, len_keeping):
        #     if keeping_boxes[i] > 0:
        #         keeping_boxes -= [int(element) for element in conflicting_boxes_mask[i]]

        keeping_boxes = (keeping_boxes == 0)
        selected_boxes.append(boxes[order][keeping_boxes[:, None].expand_as(boxes)].view(-1, 6).contiguous())

    temp_total = 0
    for boxes in selected_boxes:
        temp_total += len(boxes)
    print("Amount of predictions after nms: ", temp_total)

    # Scaling boxes according to image
    final_boxes = []
    for boxes in selected_boxes:
        if len(boxes) == 0:
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
                                 int(box[5].item())] for box in boxes])

    return final_boxes
