import pandas as pd

class NutritionRepository:
    def __init__(self):
        self.df = pd.read_csv("nutrition.csv")

    def get_rows_by_label(self, food_label):
        target = food_label.strip().lower().replace('_', ' ')
        mask = self.df['label'].astype(str).str.lower().str.replace('_', ' ').str.contains(target, na=False)
        result = self.df[mask]
        return result if not result.empty else None
