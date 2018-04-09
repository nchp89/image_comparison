import cv2
import numpy as np
from skimage.measure import compare_ssim as ssim
import urllib

white = [255,255,255]
#Read the image, convert it into grayscale, and make in binary image for threshold value of 1.
req = urllib.urlopen('https://static.graybar.com/content-resource/image/leviton-5500-20n-22000836/336768-portrait_ratio1x1-300-300-cac8008fafbc0f0b94f886ccc2f5fc8f-kt.jpg')
arr = np.asarray(bytearray(req.read()), dtype=np.uint8)
img = cv2.imdecode(arr, cv2.IMREAD_COLOR)

req = urllib.urlopen('https://static.graybar.com/supplierimages/LEVITON_MANUFACTURING_COMPANY_INC/LEVMFCE01253_F9_PE_003.png')
arr = np.asarray(bytearray(req.read()), dtype=np.uint8)
img1 = cv2.imdecode(arr, cv2.IMREAD_COLOR)
	
img_inv = cv2.bitwise_not(img)
img1_inv = cv2.bitwise_not(img1)

gray = cv2.cvtColor(img_inv,cv2.COLOR_BGR2GRAY)
_,thresh = cv2.threshold(gray,20,255,cv2.THRESH_BINARY)
gray1 = cv2.cvtColor(img1_inv,cv2.COLOR_BGR2GRAY)
_,thresh1 = cv2.threshold(gray1,20,255,cv2.THRESH_BINARY)


#Now find contours in it. There will be only one object, so find bounding rectangle for it.
contours = cv2.findContours(thresh,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
cnt = contours[0]
x,y,w,h = cv2.boundingRect(cnt)

contours1 = cv2.findContours(thresh1,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
cnt1 = contours1[0]
x1,y1,w1,h1 = cv2.boundingRect(cnt1)


#Now crop the image, and save it into another file.
crop = img[y:y+h,x:x+w]
crop1 = img1[y1:y1+h1,x1:x1+w1]

crop = cv2.resize(crop, (64, 64), interpolation=cv2.INTER_AREA)
crop1 = cv2.resize(crop1, (64, 64), interpolation=cv2.INTER_AREA)

print(ssim(crop,crop1,multichannel=True))

cv2.imwrite('after.jpg',crop)
cv2.imwrite('after1.jpg',crop1)
