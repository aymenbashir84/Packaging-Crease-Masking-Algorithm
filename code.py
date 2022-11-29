from tkinter import Y
import cv2 
import numpy as np
import matplotlib.pyplot as plt
import time
import glob

# Set up the detector with default parameters.
def contour_detector(img_b, threshold_l, threshold_u, size_l, size_u, kernel):
  #img = processing_pin_points(img_bi)
  #img_b = cv2.GaussianBlur(img_b,(kernel, kernel), 0)
  img_bi= cv2.cvtColor(img_b, cv2.COLOR_BGR2GRAY)

  thresh = cv2.Canny(img_b,threshold_l, threshold_u)
  #ret, thresh = cv2.threshold(img_bi, threshold_l, 255, cv2.THRESH_BINARY)
  contours, hierarchy = cv2.findContours(image=thresh, mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_NONE)
  params = cv2.SimpleBlobDetector_Params()
  #image_copy = img_b.copy()
  image_copy = np.zeros(img_b.shape)
  c = []
  centers = []
  for contour in contours:
    if cv2.contourArea(contour) > size_l and cv2.contourArea(contour) < size_u:
        c.append(contour)
        #cv2.drawContours(image_copy, contour, -1, (0, 255, 0), 2)
        M = cv2.moments(contour)
        if M['m00'] != 0:
          centers.append([int(M['m10']/M['m00']), int(M['m01']/M['m00'])])
  #cv2.drawContours(image=image_copy, contours=contours, contourIdx=-1, color=(0, 255, 0), thickness=2, lineType=cv2.LINE_AA)
  return image_copy, centers, thresh


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


def generate_template_2(over_template, circles,x, y, dis):
    #dis = []
    indic = []
    circles1 = np.asarray(circles.copy())
    circles1[:, 0] = 100*((circles1[:, 0]/100).astype("intc"))
    circles1 = circles1[np.lexsort((circles1[:, 1], circles1[:, 0]))]
    if len(circles1) > 1:
      for i in range(len(circles1)-1):
        if (abs(int(circles1[i,0]) - int(circles1[i+1, 0])) < 10 and abs(int(circles1[i,1]) - int(circles1[i+1, 1])) < 1000) :
          indic = [i, i+1]
          dis.append(abs(int(circles1[i+1, 1])- int(circles1[i,1])))
          #break
    if len(indic) > 1:
      print(dis)
      arr = dis.copy()
      dis_m = np.mean(np.asarray(dis))
      print(dis)
      print(type(dis))
      width = over_template.shape[1]
      height = int((dis_m/668)*over_template.shape[0]) 
      dim = (width, height)
      #print(dim)
      resized = cv2.resize(over_template, dim, interpolation = cv2.INTER_AREA)
      #te_circle, img = circle_detection(np.asarray(resized[:,:,1]).astype("uint8"), 0, 180, 200, 30, 55)
      te_circle_x, te_circle_y = x, y*float(dis_m/668)

      return resized, te_circle_x, te_circle_y, arr
    else:
      #print("No Dilation Occurred")
      return over_template, x, y, arr

def mask_generate(circles, template, te_circle_x, te_circle_y, img2, blurred_image):
      mask = np.zeros((img2.shape[0], img2.shape[1]))
      masked = 0
      circles1 = np.asarray(circles.copy())
      circles1 = circles1[np.lexsort((circles1[:, 1], circles1[:, 0]))]
      coordinates = [[circles1[0,1], circles1[0, 0]]]
      for cor in range(len(circles1)):
        if len(coordinates) ==4:
              break
        dif = abs(int(coordinates[len(coordinates)-1][1]) - int(circles1[cor, 0]))
        if dif < 900 and dif > 10:
          coordinates.append([circles1[cor,1], circles1[cor, 0]])
      y = np.asarray([item[0] for item in coordinates])
      co_y, co_x = int(te_circle_y)-np.asarray([item[0] for item in coordinates]), abs(int(te_circle_x) - np.asarray([item[1] for item in coordinates]).astype("int"))
      for i in range(len(co_y)):
        if i > 2: 
          mask[0:img2.shape[0], co_x[i]:] = template[co_y[i]: co_y[i] +img2.shape[0], 0:mask[0: 4096, co_x[i]:].shape[1]]
        else:
          mask[0: img2.shape[0], co_x[i]: co_x[i] + template.shape[1]] = template[co_y[i]: co_y[i] +img2.shape[1]]

      masked = np.where(mask.astype("uint16")>0, 0, img2[:,:,0].astype("uint8"))
      output = np.where(mask.astype("uint16")>0,blurred_img[:,:,1].astype("uint8"), img2[:,:,0].astype("uint8") )

      return masked, output

def hole_blurring_2(template, te_circle_x, te_circle_y, img2, blurred_img,  dis, visualize = False):  
    t_s0 = time.time()
    circles, image_circles = circle_detection(np.asarray(img2[:,:,0]).astype("uint8"), 0, 90, 215, 30, 15)
    circles = circles[circles[:, 0].argsort()]
    t_e0 = time.time()
    t_c = t_e0 - t_s0
    print("1. Circle Detection:", t_e0 - t_s0)
    if len(circles) > 1:
       t_s1 = time.time()
       template, te_circle_x, te_circle_y, dist_h = generate_template_2(template, circles.copy(), te_circle_x, te_circle_y, dis)
       t_e1 = time.time()
       t_t = t_e1-t_s1

       t_s2 = time.time()
       masked, output = mask_generate(circles, template, te_circle_x, te_circle_y, img2, blurred_img)
       t_e2 = time.time()
       t_m = t_e2 - t_s2
       if visualize == True:
          plt.figure(figsize = (15, 15))
          plt.imshow(masked[0:3000,:], cmap = 'gray', vmin = 0, vmax = 255)
       t_e3 = time.time()
       #print("5. Getting Output:", t_e3- t_s3)
       return masked, output, t_c, t_t, t_m, dist_h
    else:
       return img2, 0, 0, 0, 0, dist_h

def detect_l(img):
   lis = []
   img = cv2.GaussianBlur(img,(3,3), 1)
   imgi = np.abs(255-img)
   dst = cv2.Canny(imgi, 40, 70, L2gradient = False,  apertureSize = 3)
   lines = cv2.HoughLinesP(dst,1,np.pi/180,300,minLineLength=50,maxLineGap=80)
   try:
     if len(lines) > 0:
        for line in lines:
            x1,y1,x2,y2 = line[0]
            cv2.line(img,(x1,y1),(x2,y2),(0,255,0),2)
        #plt.figure(figsize = (5, 5))
        #plt.imshow(img, cmap = 'gray' )
        print("Streak Detected")
        return img
   except:
     print("No streak detected!")
     return []
  
time_sum=0
samples=7
count = 0
nd = 0
total_time = []
total_blurring = []
circle_detection_time = []
template_rescaling = []
generating_mask = []
template_pasting = []
line_detection_time = []
total_time_check = []

over_all = cv2.imread("D:\Projects\TTPak\Code\\img_dilation.png")
over_all = over_all[:,:,1]
te_circle_x, te_circle_y = (400, 5686)

#blurred_img = cv2.imread("D:\Projects\TTPak\Code\\blurred_img.png")
dis = []
for i in range(1):
    for img_name in glob.glob("D:\Projects\TTPak\Code\Results\*.png"):
      print(img_name)
      img2= cv2.imread(img_name)
      img2 = cv2.flip(img2, 0)
      blurred_img = cv2.GaussianBlur(img2, (125, 125), 0).astype("uint8")
      t_s1 = time.time()
      msk, out, t_c, t_t, t_m, distance = hole_blurring_2(over_all, te_circle_x, te_circle_y, img2, blurred_img, dis, False)
      t_e1 = time.time()
      
      print("Blurring of Hole:", t_e1 - t_s1)
      t_s2 = time.time()
      o = detect_l(out[:,650:])
      t_e2 = time.time()
      print("Line Detection:", t_e2 - t_s2)
      te_f = time.time()
      total_blurring.append(t_e1 - t_s1)
      circle_detection_time.append(t_c)
      template_rescaling.append(t_t)
      generating_mask.append(t_m)
      line_detection_time.append(t_e2-t_s2)
      time_sum = t_c + t_t + t_m + t_e2-t_s2
      total_time.append(time_sum)
      total_time_check.append(te_f - t_s1)
      
      outfile = "D:\Projects\TTPak\Code\Results\Results\\results%s.jpg" %(time_sum)
      outfile1 = "D:\Projects\TTPak\Code\Results\Results\\lines%s.jpg" %(time_sum)
      #o = msk if o == [] else o
      try:
         cv2.imwrite(outfile1, o)
         count = count+1
      except: 
        print("No Defect Detected")
        nd = nd+1
      #cv2.imwrite(outfile, msk)
      #outfile = 'results1%s.png' %(time_sum)
      #cv2.imwrite(outfile, a)
      #average_time_sum=time_sum/samples
      #print(f"Average Time:{average_time_sum}")
      print(te_f - t_s1)

print("Average Total Time :", np.sum(np.asarray(total_time_check)/len(total_time_check)))
print("Distance between Holes :", np.sum(np.asarray(distance)/len(distance)))
print("Average Blurring Time :", np.sum(np.asarray(total_blurring)/len(total_time)))
print("Average Circle Detection Time :", np.sum(np.asarray(circle_detection_time)/len(total_time)))
print("Average Rescaling Time :", np.sum(np.asarray(template_rescaling)/len(total_time)))
print("Average Mask Generation Time :", np.sum(np.asarray(generating_mask)/len(total_time)))
print("Defects:", count)
print("No Defects:", nd)
np.save("total_time.csv", total_time)
np.save("distance.csv", distance)
np.save("Blurring_time.csv", total_blurring)
np.save("Circle_Detection_time.csv", circle_detection_time)
np.save("Template Rescaling.csv", template_rescaling)
np.save("Mask_Generation_time.csv", generating_mask)
np.save("Contour_detection_time.csv", line_detection_time)