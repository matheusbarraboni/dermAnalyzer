import cv2
import numpy as np

imagem = cv2.imread('pele_oleosa.jpg')

gray = cv2.cvtColor(imagem, cv2.COLOR_BGR2GRAY)
suave = cv2.GaussianBlur(gray, (7, 7), 0)

# Aplica a lógica de Binarização
#(T, bin) = cv2.threshold(suave, 160, 255, cv2.THRESH_BINARY)

# Aplica a lógica de Binarização
#(T, binI) = cv2.threshold(suave, 160, 255, cv2.THRESH_BINARY_INV)

cv2.imwrite('imagem/imagemProcessada.jpg',suave)