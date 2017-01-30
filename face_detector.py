# -*- coding: utf-8 -*-
import sys
import dlib
import imutils
import os
from skimage import io
import argparse
from imutils import paths
from imutils.object_detection import non_max_suppression
import cv2
import time

t=time.time() # Measure time taken to run

# initialize dlib face detector
detector = dlib.get_frontal_face_detector()
len_0_cases = []
len_2_cases = []
for subjectDir in os.listdir("Images/"):
# subjectDir = "059-fn059"
    for sessionDir in os.listdir("Images/"+subjectDir+"/"):
        os.chdir("Images/"+subjectDir+"/"+sessionDir+"/")
        for fileNames in os.listdir("."):
            if fileNames.endswith(".png"):
                img = io.imread(fileNames)
                img = imutils.resize(img, width=min(400, img.shape[1]))
                dets = detector(img, 1) # The 1 in the second argument indicates that we should upsample the image
                                        # 1 time.  This will make everything bigger and allow us to detect more
                                        # faces.
                if (len(dets)<1):
                    # print("{} faces detected in a single image".format(len(dets)))
                    len_0_cases.append(fileNames)
                    # print ("<1 "+fileNames)
                elif (len(dets)>1):
                    # print("{} faces detected in a single image".format(len(dets)))
                    len_2_cases.append(fileNames)
                    # print (">1 "+fileNames)
                else:
                    for i,d in enumerate(dets): 
                        imgCropped = img[d.top():d.bottom(),d.left():d.right()]
			imgCropped = imutils.resize(imgCropped, width=180, height = 200)
                        # tempVer = d.bottom()-d.top()
                        # tempHor = d.right()-d.left()
                        # if tempVer>maxDimVer:
                        #     maxDimVer=tempVer
                        #     maxDimVerName=fileNames
                        # if tempHor>maxDimHor:
                        #     maxDimHor=tempHor
                        #     maxDimHorName=fileNames
                        file_name, file_extension = os.path.splitext(fileNames)
                        io.imsave(file_name+"_cropped.png",imgCropped)
        os.chdir("../../..")
    print subjectDir

# print("Maximum Horizontal Dimension of face is {}".format(maxDimHor))
# print("Maximum Vertical Dimension of face is {}".format(maxDimVer))                
print("Elapsed time is {}".format(time.time()-t))
len_0_file = open('len_0.txt', 'w')
for item in len_0_cases:
    len_0_file.write("%s\n" % item)
len_0_file.close()
len_2_file = open('len_2.txt','w')
for item in len_2_cases:
    len_2_file.write("%s\n" % item)
len_2_file.close()
# 206 vert 186 hor
