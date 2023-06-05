from skimage import io
import numpy as np
import cv2

def classify_color_from_image(image_path):
  result = ""
  color_palette = {
      # [213,168,139]
      "1": 520,
      # [183, 108, 59]
      "2": 350,
      # [133, 75, 55]
      "3": 263,
      # [60,32,29]
      "4": 121
  }

  img = io.imread(image_path)
  img = cv2.GaussianBlur(img, (11, 11), 0)

  pixels = np.float32(img.reshape(-1, 3))

  n_colors = 3
  criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 200, .1)
  flags = cv2.KMEANS_RANDOM_CENTERS

  _, labels, palette = cv2.kmeans(pixels, n_colors, None, criteria, 10, flags)
  _, counts = np.unique(labels, return_counts=True)

  dominant = palette[np.argmax(counts)]

  for cor, predefinicao in color_palette.items():
    if (sum(dominant) > predefinicao):
      return cor
  return result