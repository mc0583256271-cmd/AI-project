# from transformers import AutoModelForImageClassification, AutoImageProcessor, pipeline
# from PIL import Image
# import os
# import pandas as pd

# # טעינת הדאטה (תחליפי לשם הקובץ האמיתי שלך)
# df = pd.read_csv("nutrition") 

# def get_nutrition_info(food_label):
#     # חיפוש המאכל בעמודת 'label'
#     result = df[df['label'] == food_label]
    
#     if not result.empty:
#         # מחזירים את כל הנתונים של השורה הראשונה שנמצאה
#         return result.iloc[0].to_dict()
#     else:
#         return None

# # דוגמה לאיך להשתמש בזה בתוך הלולאה שלך:
# label_from_model = predictions[0]['label']
# nutrients = get_nutrition_info(label_from_model)

# if nutrients:
#     print(f"--- תוצאות עבור: {label_from_model} ---")
#     print(f"קלוריות: {nutrients['calories']}")
#     print(f"חלבון: {nutrients['protein']} גרם")
#     print(f"פחמימות: {nutrients['carbohydrates']} גרם")
#     print(f"שומנים: {nutrients['fats']} גרם")
# else:
#     print(f"לא מצאנו נתונים עבור {label_from_model} בבסיס הנתונים.")

# # מצביע לתיקייה שבה שמרתן את הקבצים שהורדתן
# local_model_path = "model_files"

# print("--- טוענת מודל מהתיקייה המקומית... ---")

# # טעינת המודל והמעבד מהקבצים שעל הדיסק
# model = AutoModelForImageClassification.from_pretrained(local_model_path)
# processor = AutoImageProcessor.from_pretrained(local_model_path)

# # יצירת ה-pipeline המקומי
# classifier = pipeline("image-classification", model=model, image_processor=processor)

# print("--- המודל נטען בהצלחה! ---")

# # לולאה לניתוח התמונות ששמתן בתיקיית images (כמו שעשינו בהתחלה)
# image_folder = "images"
# for filename in os.listdir(image_folder):
#     if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
#         img_path = os.path.join(image_folder, filename)
#         img = Image.open(img_path)
        
#         # הרצת הניתוח
#         predictions = classifier(img)
        
#         print(f"\n--- תוצאות עבור {filename} ---")
#         print(f"המערכת זיהתה: {predictions[0]['label']}")
#         print(f"רמת ביטחון: {predictions[0]['score']:.2%}")

# from transformers import AutoModelForImageClassification, AutoImageProcessor, pipeline
# from PIL import Image
# import os
# import pandas as pd

# # 1. טעינת ה-CSV (תוודאי שזה שם הקובץ המדויק עם .csv בסוף)
# df = pd.read_csv("nutrition.csv") 

# def get_nutrition_info(food_label):
#     # המרת הפורמט של המודל לפורמט שבטבלה (מקו תחתון לרווח)
#     search_label = food_label.replace('_', ' ')
#     result = df[df['label'] == search_label]
    
#     if not result.empty:
#         return result.iloc[0].to_dict()
#     return None

# # 2. טעינת המודל (חייב לקרות לפני הלולאה)
# local_model_path = "model_files"
# print("--- טוענת מודל מהתיקייה המקומית... ---")
# model = AutoModelForImageClassification.from_pretrained(local_model_path)
# processor = AutoImageProcessor.from_pretrained(local_model_path)
# classifier = pipeline("image-classification", model=model, image_processor=processor)
# print("--- המודל נטען בהצלחה! ---")

# # 3. לולאת הרצת התמונות
# image_folder = "images"
# for filename in os.listdir(image_folder):
#     if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
#         img_path = os.path.join(image_folder, filename)
#         img = Image.open(img_path)
        
#         # הרצת הניתוח
#         predictions = classifier(img)
#         label = predictions[0]['label']
#         score = predictions[0]['score']
        
#         print(f"\n--- תוצאות עבור {filename} ---")
#         print(f"המערכת זיהתה: {label}")
#         print(f"רמת ביטחון: {score:.2%}")
        
#         # 4. שליפת ערכים תזונתיים
#         # nutrients = get_nutrition_info(label)
#         # if nutrients:
#         #     print(f"קלוריות: {nutrients['calories']}")
#         #     print(f"חלבון: {nutrients['protein']} גרם")
#         #     print(f"פחמימות: {nutrients['carbohydrates']} גרם")
#         #     print(f"שומנים: {nutrients['fats']} גרם")
#         # else:
#         #     print(f"לא מצאנו נתונים עבור {label} בבסיס הנתונים.")
#         def get_nutrition_info(food_label):
#     # 1. נקה רווחים מיותרים מכל צד, הפוך לאותיות קטנות
#             target = food_label.strip().lower().replace('_', ' ')
            
#             # 2. נקה גם את עמודת ה-label בטבלה עצמה (למקרה שיש שם רווחים)
#             df['clean_label'] = df['label'].astype(str).str.strip().str.lower().str.replace('_', ' ')
            
#             # 3. עכשיו נחפש התאמה
#             result = df[df['clean_label'] == target]
            
#             if not result.empty:
#                 # מחזיר ממוצע של כל השורות שנמצאו
#                 return result.mean(numeric_only=True).to_dict()
#             else:
#                 # הדפסה שתעזור לנו להבין מה המחשב רואה
#                 print(f"DEBUG: לא מצאתי התאמה ל-'{target}'")
#                 return None


from transformers import AutoModelForImageClassification, AutoImageProcessor, pipeline
from PIL import Image
import os
import pandas as pd

# 1. טעינת הנתונים (CSV) - ודאי ששם הקובץ בתיקייה הוא בדיוק "nutrition.csv"
df = pd.read_csv("nutrition.csv") 

# 2. פונקציית חיפוש חכמה וסלחנית - מוגדרת מחוץ ללולאה
def get_nutrition_info(food_label):
    # הופכים את שם המאכל מהמודל לפורמט נקי (אותיות קטנות, ללא קו תחתון)
    target = food_label.strip().lower().replace('_', ' ')
    
    # חיפוש "חלקי" - האם השם מהמודל מופיע בתוך השם ב-CSV?
    # זה פותר בעיות של רווחים נסתרים או הבדלים קטנים בכתיב
    mask = df['label'].astype(str).str.lower().str.replace('_', ' ').str.contains(target, na=False)
    result = df[mask]
    
    if not result.empty:
        # מחזיר ממוצע של הערכים במידה ויש כמה שורות לאותו מאכל
        return result.mean(numeric_only=True).to_dict()
    else:
        # אם לא נמצא, זה ידפיס לנו למה
        print(f"DEBUG: לא נמצאה התאמה ב-CSV עבור המאכל: '{target}'")
        return None

# 3. טעינת המודל מהתיקייה המקומית
local_model_path = "model_files"
print("--- טוענת מודל... ---")
model = AutoModelForImageClassification.from_pretrained(local_model_path)
processor = AutoImageProcessor.from_pretrained(local_model_path)
classifier = pipeline("image-classification", model=model, image_processor=processor)
print("--- המודל נטען בהצלחה! ---")

# 4. לולאת הרצת התמונות מהתיקייה
image_folder = "images"
for filename in os.listdir(image_folder):
    if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
        img_path = os.path.join(image_folder, filename)
        img = Image.open(img_path)
        
        # הרצת הניתוח
        predictions = classifier(img)
        label = predictions[0]['label']
        score = predictions[0]['score']
        
        print(f"\n--- תוצאות עבור {filename} ---")
        print(f"המערכת זיהתה: {label}")
        print(f"רמת ביטחון: {score:.2%}")
        
        # שליפת הערכים התזונתיים
        nutrients = get_nutrition_info(label)
        
        if nutrients:
            print(f"קלוריות: {nutrients.get('calories', 'N/A')}")
            print(f"חלבון: {nutrients.get('protein', 'N/A')} גרם")
            print(f"פחמימות: {nutrients.get('carbohydrates', 'N/A')} גרם")
            print(f"שומנים: {nutrients.get('fats', 'N/A')} גרם")
        else:
            print(f"לא מצאנו נתונים עבור {label} בבסיס הנתונים.")