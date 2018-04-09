'''
Created on 9/18/2017
Author: Nick Hunkins (a-hunkinc)

Script to compare two images and return the GBID for those images that
are poor matches. This used the structural similarity measurement method in the skimage lib
'''

import os
import urllib
import numpy as np
import math
import re
import cv2
from skimage.measure import compare_ssim as ssim
from collections import OrderedDict



def get_image_directory():
	#get directory that contains image files. Files should be .htm
	dir = raw_input("Enter directory of image files. Press enter to use the default directory 'images' :")
	if dir == '':
		dir = 'images'
	return dir

	
	
def parse_line(line):
	#parse out info from html format
	pieces = line.split('<img src=')
	gbid = pieces[0][4:12]
	urls = re.findall('<img src="(.*?)">', line)
	formatted_line = (gbid, urls[0], urls[1])
	return formatted_line
	
	
	
def load_image_list(dir):
	#load image files from given directory into a list of tuples
	image_list = []
	
	#read contents of directory into dir_contents
	dir_contents = os.listdir(dir)
	
	#go thru directory and read file info into appropriate list
	for file in dir_contents:
		f = open(dir + '/' + file)
		for line in f:
			#check if line contains html info
			if line.startswith('<br>'):
				image_list.append(parse_line(line))
	
	#return image_list as type list of triple tuples
	return image_list

	
	
def compare_image_sizes(img1, img2):
	#make images the same size so we can use ssim method
	dim1 = img1.shape
	dim2 = img2.shape
	#calculate diagonal of image and take the smaller image as the new size
	size1 = math.sqrt(dim1[0]**2 + dim1[1]**2)
	size2 = math.sqrt(dim2[0]**2 + dim2[1]**2)
	if size1 <= size2:
		return (img1, img2)
	else:
		return (img2, img1)


def shape_image(s_img, l_img):
	s_dim = s_img.shape
	l_dim = l_img.shape
	print(s_dim)
	print(l_dim)
			
	hdiff = l_dim[0] - s_dim[0]
	wdiff = l_dim[1] - s_dim[1]

	htop = 0
	hbot = 0
	wr = 0
	wl = 0
	
	# get height crop distances. account for odd and even cases
	if hdiff%2 != 0:
		htop = ((hdiff+1) / 2) -1
		hbot = ((hdiff+1) / 2) 
	else:
		htop = hdiff / 2
		hbot = hdiff / 2
	# get width crop distances
	if wdiff%2 != 0:
		wr = ((wdiff+1) /2) -1
		wl = ((wdiff+1) /2)
	else:
		wr = wdiff / 2
		wl = wdiff / 2
			
	sh_img = l_img[hbot : l_img.shape[0] - htop, wl : l_img.shape[1] - wr]
		
	# print(hdiff)
	# print(wdiff)
	# print('****')
	# print(htop)
	# print(hbot)
	# print(wr)
	# print(wl)
	# print(sh_img.shape)
	# print('')
	return(sh_img)
		

def crop_image(s_img, l_img):
	#this function trims off white space and returns rectangular images
	#that are bounded around the objecst in the images
	#invert img colors
	img_inv = cv2.bitwise_not(s_img)
	img1_inv = cv2.bitwise_not(l_img)

	gray = cv2.cvtColor(img_inv,cv2.COLOR_BGR2GRAY)
	_,thresh = cv2.threshold(gray,20,255,cv2.THRESH_BINARY)
	gray1 = cv2.cvtColor(img1_inv,cv2.COLOR_BGR2GRAY)
	_,thresh1 = cv2.threshold(gray1,20,255,cv2.THRESH_BINARY)


	#Now find contours in it. There will be only one object, so find bounding rectangle for it.
	contours = cv2.findContours(thresh,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
	cnt = contours[0]
	x,y,w,h = cv2.boundingRect(cnt)

	contours1 = cv2.findContours(thresh1,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
	cnt1 = contours1[0]
	x1,y1,w1,h1 = cv2.boundingRect(cnt1)


	#Now crop the image, and save it into another file.
	crop = s_img[y:y+h,x:x+w]
	crop1 = l_img[y1:y1+h1,x1:x1+w1]

	crop = cv2.resize(crop, (64, 64), interpolation=cv2.INTER_AREA)
	crop1 = cv2.resize(crop1, (64, 64), interpolation=cv2.INTER_AREA)
	
	return (crop,crop1)

	
	
def make_windows(s_img, l_img):
	#get dimensions
	s_dim = s_img.shape
	l_dim = l_img.shape
	
	#find center of imgs
	s_center = (int(s_dim[0]/2), int(s_dim[1]/2))
	l_center = (int(l_dim[0]/2), int(l_dim[1]/2))

	#get window around center of image based on img size
	if s_dim[0] >= 100 and s_dim[1] >= 100:
		s_window = s_img[s_center[0]-50:s_center[0]+50, s_center[1]-50:s_center[1]+50]
		l_window = l_img[l_center[0]-50:l_center[0]+50, l_center[1]-50:l_center[1]+50]
	elif s_dim[0] >= 50 and s_dim[1] >= 50:
		s_window = s_img[s_center[0]-25:s_center[0]+25, s_center[1]-25:s_center[1]+25]
		l_window = l_img[l_center[0]-25:l_center[0]+25, l_center[1]-25:l_center[1]+25]
	else:
		x = min(s_dim[0], s_dim[1])/2 -1
		s_window = s_img[s_center[0]-x : s_center[0]+x , s_center[1]-x : s_center[1]+x]
		l_window = l_img[l_center[0]-x : l_center[0]+x , l_center[1]-x : l_center[1]+x]
	
	return(s_window, l_window)

	
	
def calc_score(img1_url, img2_url):
	#use the structural similarity measurement (ssim) method to calculate difference score
	img1 = get_image_from_url(img1_url)
	img2 = get_image_from_url(img2_url)
	
	#resize images so we can do ssim test
	result = compare_image_sizes(img1,img2)
	s_img = result[0]
	l_img = result[1]
	
	#resize the larger image via cropping
	windows = make_windows(s_img, l_img)
	cropped_images = crop_image(s_img, l_img)
	sh_img = shape_image(s_img, l_img)
	
	#resize images 
	sh_img_rz = cv2.resize(sh_img, (64, 64), interpolation=cv2.INTER_AREA)
	s_img_rz = cv2.resize(s_img, (64, 64), interpolation=cv2.INTER_AREA)
	
	#now do ssim with resized images
	ssim_score1 = ssim(windows[0], windows[1], multichannel=True)
	ssim_score2 = ssim(cropped_images[0], cropped_images[1], multichannel=True)
	ssim_score3 = ssim(sh_img_rz, s_img_rz, multichannel=True)
	
	return (ssim_score1, ssim_score2, ssim_score3, (img1_url, img2_url))



def get_image_from_url(img):
	#get images from url
	req = urllib.urlopen(img)
	arr = np.asarray(bytearray(req.read()), dtype=np.uint8)
	img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
	
	return img

	
	
def compare_images(image_list):
	#calculate structural similarity measurement and add to dict w/ gbid as key
	ssim_dict = OrderedDict()
	ef = open("errors.txt", 'w')
	for i in image_list:
		try:
			score = calc_score(i[1], i[2])
			ssim_dict[i[0]] = score
		except: #if there's an error, put it in the dict and log it in errors.txt
			print('Error: ' + i[1] + ' ::: ' + i[2])
			ef.write(i[0] + '\nImg1 URL: ' + i[1] + '\nImg2 URL: ' + i[2] +'\n')
	ef.close()
	return ssim_dict



if __name__ == '__main__':
	#main got a bit sloppy from all the HTML features which were an afterthought
	
	dir = get_image_directory()
	image_list = load_image_list(dir)
	ssim_dict = compare_images(image_list)
	
	#open files for writing
	html_header = '''
	<!DOCTYPE html>
	<html>
	<body>
	'''
	w1 = open('same.htm', 'w')
	f1 = open('same.txt', 'w')
	w2 = open('review.htm', 'w')
	f2 = open('review.txt', 'w')
	w3 = open('not_same.htm', 'w')
	f3 = open('not_same.txt', 'w')
	
	#write headers to html docs
	w1.write(html_header)
	w2.write(html_header)
	w3.write(html_header)
	
	for key in ssim_dict:
		s1 = round(ssim_dict[key][0],2)
		s2 = round(ssim_dict[key][1],2)
		s3 = round(ssim_dict[key][2],2)
		#if the scores meet a criteria then add them to same.txt and same.htm
		if s1 >= 0.99 or s2 >= 0.91 or s3 >= 0.99 :
			w1.write('<br><button id="' + str(key) + '" onclick="appendGBID(id)">' + str(key) + '</button><img src="' + \
			(ssim_dict[key])[3][0] + '"><img src="' + (ssim_dict[key])[3][1]+'">' + str((s1,s2,s3)) + '<br>')
			f1.write(key + '\n')
		elif s2 >= 0.75 or s3 >= 0.85:
			w2.write('<br><button id="' + str(key) + '" onclick="appendGBID(id)">' + str(key) + '</button><img src="' + \
			(ssim_dict[key])[3][0] + '"><img src="' + (ssim_dict[key])[3][1]+'">' + str((s1,s2,s3)) + '<br>')
			f2.write(key + '\n')
		else:
			w3.write('<br><button id="' + str(key) + '" onclick="appendGBID(id)">' + str(key) + '</button><img src="' + \
			(ssim_dict[key])[3][0] + '"><img src="' + (ssim_dict[key])[3][1]+'">' + str((s1,s2,s3)) + '<br>')
			f3.write(key + '\n')
	

	html_footer = '''
	<script>
	function appendGBID(id) {
		var node = document.createElement("LI");
		var textnode = document.createTextNode(id);
		node.appendChild(textnode);
		document.getElementById("gbidList").appendChild(node);
	}
	</script>
	
	<h3><u>GBID LOG</u></h3>
	<ul id="gbidList">
	</ul>
	
	</body>
	</html>
	'''
	
	#write footers to html docs
	w1.write(html_footer)
	w2.write(html_footer)
	w3.write(html_footer)
	
	#close file streams
	w1.close()		
	f1.close()
	w2.close()
	f2.close()
	w3.close()
	f3.close()
	
	
	
	
