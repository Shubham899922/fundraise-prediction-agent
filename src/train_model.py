from xgboost import XGBClassifier

def train_model(X, y):
    """
    Train an XGBoost classifier on input features and labels.
    """
    model = XGBClassifier(use_label_encoder=False, eval_metric='logloss')
    model.fit(X, y)
    return model
