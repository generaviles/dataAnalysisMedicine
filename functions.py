"""
Customized functions for MsC thesis project.
In the format of Ricardo's workflow

May 2018
"""

import numpy as np
import math
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
#import cv2


#=======Image Complement===================
def imcomplement(image):
	inversa = np.subtract(np.max(image),image)
	return inversa

#======== Image Histogram=================

def hist16(image):
	M,N = np.shape(image)
	vMin = np.min(image)
	vMax = np.max(image)
	#~ print vMin,vMax
	x = np.arange(vMin,vMax+1)
	#~ print x
	histograma = np.zeros(len(x), dtype=int)
	for l in range(len(histograma)):
		#~ a = np.argwhere(x==l)
		#~ print a
		histograma[l]=sum(sum(image == x[l]))
	return histograma,x

#===================== Cumulative Histogram ==============

def histAc(histograma):
	hac = np.zeros(len(histograma), dtype=int)
	suma = 0
	for i in range(len(histograma)):
		suma = suma + histograma[i]
		hac[i] = suma
	return hac

#==============Histogram Equalization==================

def histEq(imagen,hac,x):
	n,m = np.shape(imagen)
	P = n * m
	L = np.max(imagen)
	s = (n,m)
	imgEqDec = np.zeros((s), dtype=float)
	for i in range(n):
		for j in range(m):
			indx = np.argwhere(x==imagen[i][j])
			b = hac[indx] * (L / float(P))
			imgEqDec[i][j] = b
	return imgEqDec

#==================RGB to GrayScale with MatPlotLib===============
#Using the following formula Y'= 0.229 R + 0.587 G + 0.114 B

#def rgb2gray(rgb):
#	return np.dot(rgb[...,3],[0.299, 0.587, 0.114])

def rgb2gray(imagen):
	RED = imagen[:,:,0]
	GREEN = imagen[:,:,1]
	BLUE = imagen[:,:,2]
	gray =  np.uint8(0.29 * RED) + np.uint8(0.59 * GREEN) + np.uint8(0.11 * BLUE)
	return gray
