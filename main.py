from flask import Flask, json, request
from classify_skin_color import classify_color_from_image
from classify_skin_type import classify_type_from_image

api = Flask(__name__)

@api.route('/classify/color', methods=['POST'])
def classify_color():
  return json.dumps({'color': classify_color_from_image(request.json['imagePath'])})

@api.route('/classify/type', methods=['POST'])
def classify_type():
  return json.dumps({'type': classify_type_from_image(request.json['imagePath'])})

if __name__ == '__main__':
    api.run()