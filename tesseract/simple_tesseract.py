import cv2
from pytesseract import pytesseract
from pytesseract import Output
import re
import glob

pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

img_paths = glob.glob('../data/zip/*.png')
for path in img_paths:
    img = cv2.imread(path)
    img_data = pytesseract.image_to_data(img, output_type=Output.DICT)
    words = img_data['text']
    print(path)
    print(words)
