from env import key
from roboflow import Roboflow

sample_result = {
    "time": 0.2649739610001234,
    "image": {
        "width": 960,
        "height": 1280
    },
    "predictions": [
        {
            "class": "Normal",
            "confidence": 0.3447
        },
        {
            "class": "Dry",
            "confidence": 0.2488
        },
        {
            "class": "Combination",
            "confidence": 0.2056
        },
        {
            "class": "Oily",
            "confidence": 0.2009
        }
    ],
    "top": "Normal",
    "confidence": 0.3447
}

def classify_type_from_image(image_path):
    rf = Roboflow(api_key=key)
    project = rf.workspace().project("skin-classification-crpgh")
    model = project.version(1).model

    result = model.predict(image_path).json()

    return result