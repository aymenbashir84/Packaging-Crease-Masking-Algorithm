from tkinter import Y
import cv2 
import numpy as np
import matplotlib.pyplot as plt
import time
import glob
def circle_detection(img, min_r, max_r, par1 = 170, par2 = 27, par0 = 55):
  #img = cv2.medianBlur(img,3)
  img1 = abs(255- img)
  cimg = cv2.cvtColor(img,cv2.COLOR_GRAY2BGR)
  cimg = np.asarray(cimg).astype("uint8")
  circles = cv2.HoughCircles(img,cv2.HOUGH_GRADIENT,1,par0,
                            param1=par1,param2=par2,minRadius=min_r,maxRadius=max_r)
  try:
    if len(circles[0,:]) > 0:
       circles = np.uint16(np.around(circles))
       #for i in circles[0,:]:
      # draw the outer circle
           #cv2.circle(cimg,(i[0],i[1]),i[2],(0,255,0),4)
      # draw the center of the circle
        #cv2.circle(img,(i[0],i[1]),2,(0,0,255),3)
    return circles[0,:], cimg
  except:
    return [], cimg


def template():
  tem = cv2.imread("D:\Projects\TTPak\Code\\template (1).png")
  ret, thresh = cv2.threshold(tem, 10, 255, cv2.THRESH_BINARY)
  thresh = thresh[:,:,2]
  height = int(float((668/821))*thresh.shape[0])
  width =  int(float((820/946))*thresh.shape[1])
  resized = cv2.resize(thresh, (width, height), interpolation = cv2.INTER_AREA)
  thresh = resized
  template = resized
  over_all_template = np.zeros([14000, 1100])
  k = 0
  for i in range(20):
      over_all_template[i*template.shape[0]:i*template.shape[0] + template.shape[0], 0: template.shape[1]] = template

  over_all_template = over_all_template[1700:, :]
  te_circle, img = circle_detection(np.asarray(over_all_template).astype("uint8"), 0, 40, 40, 20, 55)
  te_circle = te_circle[te_circle[:, 1].argsort()]
  te_circle_x, te_circle_y = te_circle[int(len(te_circle)/2),0], te_circle[int(len(te_circle)/2),1]
  print(te_circle_x, te_circle_y)
  kernel = np.ones((9, 9), np.uint8)
  img_dilation= cv2.dilate(over_all_template, kernel, iterations=4).astype("uint8")
  plt.imshow(img_dilation)
  cv2.imwrite("D:\Projects\TTPak\Code\img_dilation.png", img_dilation)
  return img_dilation

img_dilation = template()