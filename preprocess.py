#!/usr/bin/python

import cv2
import numpy as np

def load_filenames_labels(arg_file):
  filenames = []
  labels = []

  text_file = open(arg_file, "r")

  for line in text_file.readlines():
    filenames.append(line.split()[0])
    labels.append(line.split()[1])

  return filenames, labels

def load_image(arg_file):
  img = cv2.imread(arg_file, cv2.IMREAD_COLOR)
  img_contours = cv2.imread(arg_file, cv2.IMREAD_COLOR)
  gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
  
  gray = preprocess_image(gray)
  
  _, contours, _ = cv2.findContours(gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

  contours_coords = []

  for contour in contours: 
    x, y, w, h = cv2.boundingRect(contour)
    cv2.rectangle(img_contours, (x, y), (x+w, y+h), (0, 255, 0), 2)

    if w > 10 and h > 10: #broscience to avoid small countours like points
      contours_coords.append(cv2.boundingRect(contour))

  contours_coords = np.array(contours_coords)
  contours_coords = contours_coords[contours_coords[:,0].argsort()]

  x1, y1, _, _ = contours_coords[0]
  x2, y2, w2, h2 = contours_coords[-1]

  img2 = img[y1:y2+h2, x1:x2+w2]
  
  rsz_h, rsz_w, _ = img.shape
  img2 = cv2.resize(img2, (rsz_w, rsz_h))

  display = np.concatenate((img_contours, img2), axis=1)

  cv2.imshow("img", display)
  cv2.waitKey(1000)

  return img

def preprocess_image(arg_img):
  _, thresh = cv2.threshold(arg_img, 127, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
  thresh = cv2.bitwise_not(thresh)

  return thresh
  

if __name__ == "__main__":
  path = "ORAND-CAR-2014/CAR-A/"
  gt_file = "a_train_gt.txt"
  path_train_images = path + "a_train_images/"

  filenames, labels = load_filenames_labels(path + gt_file)  
  x_train = []

  for filename in filenames:
    x_train.append(load_image(path_train_images + filename))

  x_train, labels = np.array(x_train), np.array(labels)

  print(x_train.shape, labels.shape)
