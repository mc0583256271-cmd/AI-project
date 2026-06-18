from transformers import AutoModelForImageClassification, AutoImageProcessor, pipeline
from PIL import Image

class ImageClassifier:
    def __init__(self):
        local_model_path = "model_files"
        model = AutoModelForImageClassification.from_pretrained(local_model_path)
        processor = AutoImageProcessor.from_pretrained(local_model_path)
        self.classifier = pipeline("image-classification", model=model, image_processor=processor)

    def predict(self, image):
        return self.classifier(image)
