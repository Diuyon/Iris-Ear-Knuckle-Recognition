from os import listdir
from os.path import isfile, join
import numpy
import cv2

left_ear_cascade=cv2.CascadeClassifier('/home/ubuntu/opencv/data/haarcascades/haarcascade_mcs_leftear.xml')
right_ear_cascade=cv2.CascadeClassifier('/home/ubuntu/opencv/data/haarcascades/haarcascade_mcs_rightear.xml')
mypath='/home/ubuntu/Downloads/ear_database/fface'
onlyfiles = [ f for f in listdir(mypath) if isfile(join(mypath,f)) ]
images = numpy.empty(len(onlyfiles), dtype=object)
gray = numpy.empty(len(onlyfiles), dtype=object)
yes=1
no=1
for n in range(0, len(onlyfiles)):
    images[n] = cv2.imread( join(mypath,onlyfiles[n]),1)
    gray[n]=cv2.imread( join(mypath,onlyfiles[n]),0)
    left_ear = left_ear_cascade.detectMultiScale(gray[n], 1.015, 5)
    right_ear = right_ear_cascade.detectMultiScale(gray[n], 1.015,5)
    if len(left_ear)!=0:
        for (x,y,w,h) in left_ear:
            cv2.rectangle(images[n],(x,y),(x+w,y+h),(255,0,0),2)
            roi_gray = gray[n][y:y+h, x:x+w]
            roi_color = images[n][y:y+h, x:x+w]
            roi=roi_color
            cv2.imwrite('/home/ubuntu/Downloads/ear_database/converted/ear_roi'+str(yes)+'.jpg',roi)
            yes+=1
            #cv2.imshow('img',images[n])
            cv2.waitKey(0)
    elif len(right_ear)!=0:
        for (x,y,w,h) in right_ear:
            cv2.rectangle(images[n],(x,y),(x+w,y+h),(255,0,0),2)
            roi_gray = gray[n][y:y+h, x:x+w]
            roi_color = images[n][y:y+h, x:x+w]
            roi=roi_color
            cv2.imwrite('/home/ubuntu/Downloads/ear_database/converted/ear_roi'+str(yes)+'.jpg',roi)
            yes+=1
            #cv2.imshow('img',images[n])
            cv2.waitKey(0)
    else:
        cv2.imwrite('home/ubuntu/Downloads/ear_database/not_converted/Image'+str(no)+'.jpg',images[n])
        no+=1
    cv2.destroyAllWindows()
