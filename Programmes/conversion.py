import cv2
import numpy as np
import os

folder_path = '/home/hugo/Projet3a/Annotations/Test/Image_Detection/base'
folder_new_path = '/home/hugo/Projet3a/Annotations/Test/Image_Detection/new_image_conversion'

i=1
list=os.listdir(folder_path)
sort_list=sorted(list,key=lambda f: int(os.path.splitext(f)[0]))

for file in sort_list:
	os.chdir(folder_path)
	lavandin=cv2.imread(file)
	hsv_lavandin=cv2.cvtColor(lavandin,cv2.COLOR_BGR2HSV)

	lower_green=np.array([25,0,0])
	upper_green=np.array([86,255,255])

	mask=cv2.inRange(hsv_lavandin,lower_green,upper_green)
	green=cv2.bitwise_and(lavandin,lavandin, mask=mask) #bitwise operation on the pixels
	kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
	erosion = cv2.erode(green, kernel, iterations = 1)
	os.chdir(folder_new_path)
	cv2.imwrite(str("{:05d}".format(i))+".jpg",erosion)
	i+=1