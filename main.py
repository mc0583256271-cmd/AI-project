from transformers import AutoModelForImageClassification, AutoImageProcessor, pipeline
from PIL import Image
import os

# מצביע לתיקייה שבה שמרתן את הקבצים שהורדתן
local_model_path = "model_files"

print("--- טוענת מודל מהתיקייה המקומית... ---")

# טעינת המודל והמעבד מהקבצים שעל הדיסק
model = AutoModelForImageClassification.from_pretrained(local_model_path)
processor = AutoImageProcessor.from_pretrained(local_model_path)

# יצירת ה-pipeline המקומי
classifier = pipeline("image-classification", model=model, image_processor=processor)

print("--- המודל נטען בהצלחה! ---")

# לולאה לניתוח התמונות ששמתן בתיקיית images (כמו שעשינו בהתחלה)
image_folder = "images"
for filename in os.listdir(image_folder):
    if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
        img_path = os.path.join(image_folder, filename)
        img = Image.open(img_path)
        
        # הרצת הניתוח
        predictions = classifier(img)
        
        print(f"\n--- תוצאות עבור {filename} ---")
        print(f"המערכת זיהתה: {predictions[0]['label']}")
        print(f"רמת ביטחון: {predictions[0]['score']:.2%}")