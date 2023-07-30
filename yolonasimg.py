import colorsys

import cv2
import torch
import numpy as np
from super_gradients.training import models
from super_gradients.common.object_names import Models
from super_gradients.training.models.detection_models.customizable_detector import CustomizableDetector
from super_gradients.training.pipelines.pipelines import DetectionPipeline
def get_prediction(image_in, pipeline):
    preprocessed_image, processing_metadata = pipeline.image_processor.preprocess_image(image=image_in.copy())

    with torch.no_grad():
        torch_input = torch.Tensor(preprocessed_image).unsqueeze(0).to(device)
        model_output = model(torch_input)
        prediction = pipeline._decode_model_output(model_output, model_input=torch_input)
    return pipeline.image_processor.postprocess_predictions(predictions=prediction[0], metadata=processing_metadata)

def get_color(number):

    hue = number * 30 % 180
    saturation = number * 103 % 256
    value = number * 50 % 256
    color = colorsys.hsv_to_rgb((hue / 179, saturation / 255, value / 255))

    return [int(c * 255) for c in color]

device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
model = models.get("yolo_nas_s", pretrained_weights="coco")
model.eval()
pipeline = DetectionPipeline(
    model=model,
    image_processor=model._image_processor,
    post_prediction_callback=model.get_post_prediction_callback(iou=0.55, conf=0.50),
    class_names=model._class_names,
)
image_path = "../images/mfc33.jpeg"
dataset = cv2.imread(image_path)
pred = get_prediction(dataset, pipeline)
print(pred)