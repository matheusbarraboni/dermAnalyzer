from env import key
from roboflow import Roboflow

def classify_type_from_image(image_path):
    rf = Roboflow(api_key=key)
    project = rf.workspace().project("skin-classification-crpgh")
    model = project.version(1).model

    result = model.predict(image_path).json()

    dict_types = {
        "Dry": "seca",
        "Normal": "normal",
        "Oily": "oleosa",
        "Combination": "mista (aspectos de seca e oleosa)"
    }

    return dict_types[result['predictions'][0]['top']]