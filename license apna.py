# -*- coding: utf-8 -*-
"""
Created on Thu Jan  2 16:01:26 2020

@author: rishabh
"""


import cv2
import imutils
import pytesseract
pytesseract.pytesseract.tesseract_cmd=r"C:\Users\rishabh\AppData\Local\Tesseract-OCR\tesseract"

image=cv2.imread("F:\Cropped Images-Text\8.png") 

image= imutils.resize(image,width=500)
cv2.imshow("ORIGINAL",image)
cv2.waitKey(0)


gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
cv2.imshow("GRAY SCALE ",gray)
cv2.waitKey(0)

gray=cv2.bilateralFilter(gray,11,17,17)
cv2.imshow("BILATERAL",gray)
cv2.waitKey(0)

edged=cv2.Canny(gray,170,200)
cv2.imshow("Canny",edged)
cv2.waitKey(0) 
 

cnts, new =cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

img1=image.copy()
cv2.drawContours(img1,cnts,-1,(0,255,0),3)
cv2.imshow("ALL Contours",img1)
cv2.waitKey(0)

cnts=sorted(cnts,key=cv2.contourArea,reverse=True)[:30]
NumberPlateCnt = None

img2=image.copy()
cv2.drawContours(img2,cnts,-1,(0,255,0),3)
cv2.imshow("TOP 30 contours",img2)
cv2.waitKey(0)


count=0
idx=7

for c in cnts:
    peri=cv2.arcLength(c,True)
    approx=cv2.approxPolyDP(c,0.02*peri,True)
    
    if len(approx)==9:
        NumberPlateCnt=approx
        
        
        x,y,w,h=cv2.boundingRect(c)
        new_img=image[y:y+h,x:x+w]
        cv2.imwrite('F:\Cropped Images-Text'+str(idx)+'.png',new_img)
        idx+=1
        
        #break

cv2.drawContours(image,[NumberPlateCnt],-1,(0,255,0),3)
cv2.imshow("Final IMage",image)
cv2.waitKey(0)

Cropped_img_loc='F:\Cropped Images-Text\7.png'
cv2.imshow("cropped Image",cv2.imread(Cropped_img_loc))


text=pytesseract.image_to_string(Cropped_img_loc,lang='eng') 
print("Number is:",text)
cv2.waitKey(0)          