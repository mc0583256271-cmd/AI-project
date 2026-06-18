class NutritionService:
    def __init__(self, repository):
        self.repository = repository

    def get_available_weights(self, label):
        rows = self.repository.get_rows_by_label(label)
        if rows is None:
            return None, None
        return rows, rows['weight'].tolist()

    def get_row_by_weight(self, rows, weight):
        return rows[rows['weight'] == weight].iloc[0]
