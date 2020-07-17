from pycocotools.coco import COCO
import json
from json import encoder
import yolo
import torch
from small.small import data_preparation
from pycocotools.cocoeval import COCOeval
from src.post_processing import post_processing


# ___SMALL___
def load_model(PATH, class_number):
    model = yolo.modelYOLO(num_classes=class_number)
    model.load_state_dict(torch.load(PATH))
    print("Model was loaded from file: ", PATH)
    return model


def evaluate_mAP(gt_json_file, predicted_json_file, annType="bbox"):
    cocoGt = COCO(gt_json_file)
    print(cocoGt.info())
    cocoDt = cocoGt.loadRes(predicted_json_file)
    image_ids, cat_ids = get_img_cat_ids(gt_json_file)
    print("image ids (evaluate_mAP) len: ", len(image_ids))

    cocoEval = COCOeval(cocoGt=cocoGt, cocoDt=cocoDt, iouType=annType)
    cocoEval.params.imgIds = image_ids
    cocoEval.params.catIds = cat_ids

    cocoEval.evaluate()
    cocoEval.accumulate()
    cocoEval.summarize()

    meanAP = cocoEval.stats[0].item()
    return meanAP


def get_img_cat_ids(annName, catNms=["person", "car", "bird", "cat", "dog"]):
    # Extracting only required images
    coco = COCO(annName)
    cat_ids = coco.getCatIds(catNms=catNms)
    # Set
    image_ids = set()
    for cat_id in cat_ids:
        for im_id in coco.getImgIds(catIds=cat_id):
            image_ids.add(im_id)
    image_ids = list(sorted(image_ids))

    return image_ids, cat_ids


def prepare_json_predictions(modelPATH, annFile, jsonName, num_classes, image_size=416, batch_size=4,
                             catNms=["person", "car", "bird", "cat", "dog"]):
    """
    Creates a json file of model predictions by a given name of model weights
    :param modelPATH: Path to saved model weights file
    :param annFile: file of ground truth annotations
    :param jsonName: file name to save predicted boxes
    :param num_classes: for small model equals to 5
    :param image_size: resized image height and width (square image)
    :param batch_size: number of images to load at the same time and to send to model
    :return: fileName of json file
    """
    model = load_model(modelPATH, num_classes)
    # Validation dataset
    dataset = data_preparation.loadCOCO("F:\WORK_Oxagile\INTERN\Datasets\COCO\\",
                                        image_size,
                                        train_bool=False,
                                        shuffle_test=False,
                                        batch_size=batch_size)
    conf = 0.2
    nms = 0.6
    print("Confidence threshold: ", conf)
    print("NMS threshold: ", nms)

    # Will need for json creating
    image_ids, cat_ids = get_img_cat_ids(annFile, catNms=catNms)
    data_json = []

    for b, data in enumerate(dataset):
        if b % 5 == 0:
            print(b)
        images, targets = data[0], data[1]
        with torch.no_grad():
            outputs = model(images)

        predictions = post_processing(outputs=outputs,
                                      image_size=image_size,
                                      confidence_threshold=conf,
                                      nms_threshold=nms)

        if len(predictions) != 0:
            # Here one prediction consists of all predictions per one image
            for im_n, prediction in enumerate(predictions):
                for single_pred in prediction:
                    xmin = int(max(single_pred[0], 0))
                    ymin = int(max(single_pred[1], 0))

                    w = int(min(single_pred[2], image_size))
                    h = int(min(single_pred[3], image_size))
                    single_pred_json = {
                        "image_id": int(image_ids[b + im_n]),
                        "category_id": int(cat_ids[single_pred[5]]),
                        "bbox": [xmin, ymin, w, h],
                        "score": float(single_pred[4])
                    }
                    data_json.append(single_pred_json)

    jsonName += ".json"
    with open(jsonName, "w") as file:
        json.dump(data_json, file)
        print("JSON saved!")

    return jsonName


if __name__ == '__main__':
    path = "F:\WORK_Oxagile\INTERN//"
    modelPATH = path + "ImageSegmentation\small\SMALL_SavedModelWeights6_after15_after20"
    num_classes = 5
    image_size = 32 * 13
    batch_size = 8
    gt_json_file = path + "Datasets\COCO//annotations//annotations_trainval2014//annotations//instances_val2014.json"

    jsonName = "SMALL_predicted_after20"

    # prepare_json_predictions(modelPATH=modelPATH,
    #                          annFile=gt_json_file,
    #                          jsonName=jsonName,
    #                          num_classes=num_classes,
    #                          image_size=image_size,
    #                          batch_size=batch_size)
    mAP = evaluate_mAP(gt_json_file,
                       jsonName + ".json")
