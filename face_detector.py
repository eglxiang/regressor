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
failure_cases=[]
for subjectDir in os.listdir("Images/"):
# subjectDir = "059-fn059"
    print subjectDir
    for sessionDir in os.listdir("Images/"+subjectDir+"/"):
        os.chdir("Images/"+subjectDir+"/"+sessionDir+"/")
        if not os.path.exists("Cropped/"):
            os.makedirs("Cropped/")
        for fileNames in os.listdir("."):
            if fileNames.endswith(".png"):
                img = io.imread(fileNames)
                # img = imutils.resize(img, width=min(400, img.shape[1]))
                dets = detector(img, 1) # The 1 in the second argument indicates that we should upsample the image
                                        # 1 time.  This will make everything bigger and allow us to detect more
                                        # faces.
                if (len(dets)<1):
                    # print("{} faces detected in a single image".format(len(dets)))
                    len_0_cases.append(fileNames)
                    # print ("<1 "+fileNames)
                elif (len(dets)>1):
                    # print("{} faces detected in a single image".format(len(dets)))
                    for i,d in enumerate(dets): 
                        try:
                            if (d.right()-d.left())<0.33*img.shape[1]: # if the witdth is too small to be a face
                                continue
                            else:                            
                                imgCropped = img[d.top():d.bottom(),d.left():d.right()]
                                imgCropped = cv2.resize(imgCropped, (180,200), interpolation = cv2.INTER_CUBIC)
                                file_name, file_extension = os.path.splitext(fileNames)
                                io.imsave("Cropped/"+file_name+"_cropped.png",imgCropped)
                        except:
                            failure_cases.append(fileNames)
                else:
                    for i,d in enumerate(dets):
                        try:
                            imgCropped = img[d.top():d.bottom(),d.left():d.right()]
                            imgCropped = cv2.resize(imgCropped, (180,200), interpolation = cv2.INTER_CUBIC)
                            file_name, file_extension = os.path.splitext(fileNames)
                            io.imsave("Cropped/"+file_name+"_cropped.png",imgCropped)
                        except:
                            failure_cases.append(fileNames)
                
        os.chdir("../../..")
#    print("Done with {}.".format(subjectDir))    

print("Elapsed time is {}".format(time.time()-t))
len_0_file = open('len_0.txt', 'a+')
for item in len_0_cases:
    len_0_file.write("%s\n" % item)
len_0_file.close()
failure_file = open('failures.txt', 'a+')
for item in failure_cases:
    failure_file.write("%s\n" % item)
failure_file.close()
# 206 vert 186 hor
