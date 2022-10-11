import torch
import numpy as np
import time
import cv2
from PIL import Image
import torchvision


EXTRACTOR_CONFIGS = {"swin_base_patch4_window12_384_in22k": {"input_size": (384, 384), "output_size": 1024, "mean_std":((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))},
                     "tf_efficientnetv2_s_in21ft1k": {"input_size": (384, 384), "output_size": 1280, "mean_std":((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))},
                     "convnext_tiny_384_in22ft1k": {"input_size": (384, 384), "output_size": 768}
                     }
CONFIDENCE_THRESHOLD = 0
MODELS_FOLDER = "models"

feature_extractor_transforms = torchvision.transforms.Compose([
            torchvision.transforms.Resize((384, 384)),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])

detector_transforms = torchvision.transforms.Compose([
            torchvision.transforms.Resize((512, 640)),
            torchvision.transforms.ToTensor()
        ])

k_best_classes = 4

def get_classes(filepath):
    with open(filepath, 'r') as c_file:
        content = c_file.readlines()
        list_classes = [classe.rstrip("\n") for classe in content]
    return list_classes


def load_detector_torchscript(host_device):
    model = torch.jit.load(f"{MODELS_FOLDER}/yolo.torchscript", map_location='cuda:0')
    # return model.eval().half()
    return model.eval().to(host_device).half()


def load_classifier_torchscript(marque, host_device, light=False):
    model_path = f'{MODELS_FOLDER}/classifier_{marque}.torchscript' if not light else f'{MODELS_FOLDER}/classifier_{marque}_light.torchscript'
    model = torch.jit.load(model_path)
    return model.eval().to(host_device).half()


def predict_torchscript(images, detector, classifier, list_classes, device):
    predictions = {}
    images = [Image.fromarray(image) for image in images if isinstance(image, np.ndarray)]
    yolo_tensors = torch.cat([detector_transforms(image).unsqueeze(0) for image in images]).to(device).half()
    results = detector(yolo_tensors)[0].detach().cpu()
    nms_results = non_max_suppression(results, conf_thres=0.0001)
    if nms_results[0].numel() > 0 and nms_results[1].numel() > 0:
        positions = nms_results[0][0][:4].type(torch.int).tolist(), nms_results[1][0][:4].type(torch.int).tolist()
        cropped_tensors = [feature_extractor_transforms(image.crop(position)).unsqueeze(0).to(device) for image, position in zip(images, positions)]
        input_tensors = torch.cat(cropped_tensors).unsqueeze(0)
        classification_result = classifier(input_tensors.half())[0]
        classification_k_confidences = classification_result.topk(k_best_classes)[0].squeeze().data.cpu()\
                .numpy().tolist()
        classification_k_classes = classification_result.topk(k_best_classes)[1].squeeze().data.cpu()\
                .numpy().tolist()
        if classification_k_confidences[0] > CONFIDENCE_THRESHOLD:
                predictions = {"classes": classification_k_classes, "confidences": classification_k_confidences,
                               "names": [list_classes[index] for index in classification_k_classes]}
    return predictions


def convert_bytes_to_np(data):
    """
    Convert 
    """
    temp = np.frombuffer(data, dtype=np.uint8)
    concat_images = cv2.imdecode(temp, cv2.IMREAD_COLOR)
    return concat_images[:concat_images.shape[0]//2], concat_images[concat_images.shape[0]//2:]


def non_max_suppression(prediction, conf_thres=0.001, iou_thres=0.6, agnostic=False):
    """
    Performs Non-Maximum Suppression (NMS) on inference results
    Returns:
         detections with shape: nx6 (x1, y1, x2, y2, conf, cls)
    """
    # Number of classes.
    nc = 1
    # nc = prediction[0].shape[1] - 5

    # Candidates.
    xc = prediction[..., 4] > conf_thres

    # Settings:
    # Minimum and maximum box width and height in pixels.
    min_wh, max_wh = 2, 640

    # Maximum number of detections per image.
    max_det = 5

    # Timeout.
    time_limit = 10.0

    t = time.time()
    output = [torch.zeros(0, 6)] * prediction.shape[0]
    for xi, x in enumerate(prediction):  # image index, image inference

        # Apply constraints:
        # Confidence.
        x = x[xc[xi]]

        # If none remain process next image.
        if not x.shape[0]:
            continue

        # Compute conf.
        x[:, 5:] *= x[:, 4:5]  # conf = obj_conf * cls_conf

        # Box (center x, center y, width, height) to (x1, y1, x2, y2).
        box = xywh2xyxy(x[:, :4])

        # Best class only.
        conf, j = x[:, 5:].max(1, keepdim=True)
        x = torch.cat((box, conf, j.float()), 1)[conf.view(-1) > conf_thres]

        # If none remain process next image.
        # Number of boxes.
        n = x.shape[0]
        if not n:
            continue

        # Batched NMS:
        # Classes.
        c = x[:, 5:6] * (0 if agnostic else max_wh)

        # Boxes (offset by class), scores.
        boxes, scores = x[:, :4] + c, x[:, 4]

        # NMS.
        i = torchvision.ops.nms(boxes, scores, iou_thres)

        # Limit detections.
        if i.shape[0] > max_det:  # limit detections
            i = i[:max_det]

        output[xi] = x[i]
        if (time.time() - t) > time_limit:
            break  # time limit exceeded

    return output


def xywh2xyxy(x):
    # Convert nx4 boxes from [x, y, w, h] to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
    y = torch.zeros_like(x) if isinstance(x, torch.Tensor) else np.zeros_like(x)
    y[:, 0] = x[:, 0] - x[:, 2] / 2  # top left x
    y[:, 1] = x[:, 1] - x[:, 3] / 2  # top left y
    y[:, 2] = x[:, 0] + x[:, 2] / 2  # bottom right x
    y[:, 3] = x[:, 1] + x[:, 3] / 2  # bottom right y
    return y