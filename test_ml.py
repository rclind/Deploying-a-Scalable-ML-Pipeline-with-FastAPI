import pandas as pd

from ml.data import process_data
from ml.model import compute_model_metrics, inference, train_model


def test_compute_model_metrics():
    """
    Test that compute_model_metrics returns precision, recall,
    and F1 scores between 0 and 1.
    """
    y = [1, 0, 1, 1]
    preds = [1, 0, 1, 0]

    p, r, fbeta = compute_model_metrics(y, preds)

    assert 0 <= p <= 1
    assert 0 <= r <= 1
    assert 0 <= fbeta <= 1


def test_inference():
    """
    Test that inference returns one prediction for each input row.
    """
    X_train = [[0, 1], [1, 0], [1, 1], [0, 0]]
    y_train = [1, 0, 1, 0]

    model = train_model(X_train, y_train)
    preds = inference(model, X_train)

    assert len(preds) == len(y_train)


def test_process_data():
    """
    Test that process_data returns features and labels
    with the same number of rows.
    """
    data = pd.DataFrame(
        {
            "age": [25, 45, 30],
            "workclass": ["Private", "Self-emp", "Private"],
            "education": ["Bachelors", "HS-grad", "Masters"],
            "marital-status": ["Never-married", "Married", "Divorced"],
            "occupation": ["Tech-support", "Exec-managerial", "Sales"],
            "relationship": ["Not-in-family", "Husband", "Unmarried"],
            "race": ["White", "Black", "Asian-Pac-Islander"],
            "sex": ["Male", "Female", "Female"],
            "native-country": ["United-States", "United-States", "India"],
            "salary": [">50K", "<=50K", ">50K"],
        }
    )

    cat_features = [
        "workclass",
        "education",
        "marital-status",
        "occupation",
        "relationship",
        "race",
        "sex",
        "native-country",
    ]

    X, y, encoder, lb = process_data(
        data,
        categorical_features=cat_features,
        label="salary",
        training=True,
    )

    assert X.shape[0] == len(y)
    assert encoder is not None
    assert lb is not None
