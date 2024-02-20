#!/usr/bin/env python3

import cv2
import numpy as np
import argparse
import time
import io

from ultralytics import YOLO
from ultralytics.solutions import object_counter

#We need to know if we are running on the Pi, because openCV behaves a little oddly on all the builds!
#https://raspberrypi.stackexchange.com/questions/5100/detect-that-a-python-program-is-running-on-the-pi
def is_raspberrypi():
    try:
        with io.open('/sys/firmware/devicetree/base/model', 'r') as m:
            if 'raspberry pi' in m.read().lower(): return True
    except Exception: pass
    return False

isPi = is_raspberrypi()

parser = argparse.ArgumentParser()
parser.add_argument("--device", type=int, default=0, help="Video Device number e.g. 0, use v4l2-ctl --list-devices")
args = parser.parse_args()
	
if args.device:
	dev = args.device
else:
	dev = 0

# model define
model = YOLO("yolov8n.pt")
	
#init video
cap = cv2.VideoCapture('/dev/video'+str(dev), cv2.CAP_V4L)
#cap = cv2.VideoCapture(0)
#pull in the video but do NOT automatically convert to RGB, else it breaks the temperature data!
#https://stackoverflow.com/questions/63108721/opencv-setting-videocap-property-to-cap-prop-convert-rgb-generates-weird-boolean
if isPi == True:
	cap.set(cv2.CAP_PROP_CONVERT_RGB, 0.0)
else:
	cap.set(cv2.CAP_PROP_CONVERT_RGB, False)

#256x192 General settings
width = 256 #Sensor width
height = 192 #sensor height
scale = 3 #scale multiplier
newWidth = width*scale 
newHeight = height*scale
alpha = 1.0 # Contrast control (1.0-3.0)
font=cv2.FONT_HERSHEY_SIMPLEX
dispFullscreen = False
cv2.namedWindow('Thermal',cv2.WINDOW_GUI_NORMAL)
cv2.resizeWindow('Thermal', newWidth,newHeight)
rad = 0 #blur radius
threshold = 2
recording = False
elapsed = "00:00:00"
snaptime = "None"

def rec():
	now = time.strftime("%Y%m%d--%H%M%S")
	#do NOT use mp4 here, it is flakey!
	videoOut = cv2.VideoWriter(now+'output.avi', cv2.VideoWriter_fourcc(*'XVID'),25, (newWidth,newHeight))
	return(videoOut)

def snapshot(heatmap):
	#I would put colons in here, but it Win throws a fit if you try and open them!
	now = time.strftime("%Y%m%d-%H%M%S") 
	snaptime = time.strftime("%H:%M:%S")
	cv2.imwrite("TC001"+now+".png", heatmap)
	return snaptime
 

while(cap.isOpened()):
	# Capture frame-by-frame
	ret, frame = cap.read()
	if ret == True:
		imdata,thdata = np.array_split(frame, 2)
		#now parse the data from the bottom frame and convert to temp!
		#https://www.eevblog.com/forum/thermal-imaging/infiray-and-their-p2-pro-discussion/200/
		#Huge props to LeoDJ for figuring out how the data is stored and how to compute temp from it.
		#grab data from the center pixel...
		hi = thdata[96][128][0]
		lo = thdata[96][128][1]
		#print(hi,lo)
		lo = lo*256
		rawtemp = hi+lo
		#print(rawtemp)
		temp = (rawtemp/64)-273.15
		temp = round(temp,2)
		#print(temp)
		#break

		#find the max temperature in the frame
		lomax = thdata[...,1].max()
		posmax = thdata[...,1].argmax()
		#since argmax returns a linear index, convert back to row and col
		mcol,mrow = divmod(posmax,width)
		himax = thdata[mcol][mrow][0]
		lomax=lomax*256
		maxtemp = himax+lomax
		maxtemp = (maxtemp/64)-273.15
		maxtemp = round(maxtemp,2)

		
		#find the lowest temperature in the frame
		lomin = thdata[...,1].min()
		posmin = thdata[...,1].argmin()
		#since argmax returns a linear index, convert back to row and col
		lcol,lrow = divmod(posmin,width)
		himin = thdata[lcol][lrow][0]
		lomin=lomin*256
		mintemp = himin+lomin
		mintemp = (mintemp/64)-273.15
		mintemp = round(mintemp,2)

		#find the average temperature in the frame
		loavg = thdata[...,1].mean()
		hiavg = thdata[...,0].mean()
		loavg=loavg*256
		avgtemp = loavg+hiavg
		avgtemp = (avgtemp/64)-273.15
		avgtemp = round(avgtemp,2)

		# Convert the real image to RGB
		bgr = cv2.cvtColor(imdata,  cv2.COLOR_YUV2BGR_YUYV)
		# channel
		if bgr.shape[2] == 2:
			bgr = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
		#Contrast
		bgr = cv2.convertScaleAbs(bgr, alpha=alpha)#Contrast
		#bicubic interpolate, upscale and blur
		bgr = cv2.resize(bgr,(newWidth,newHeight),interpolation=cv2.INTER_CUBIC)#Scale up!
		if rad>0:
			bgr = cv2.blur(bgr,(rad,rad))
			
		heatmap = cv2.applyColorMap(bgr, cv2.COLORMAP_JET)

		# yolo detection
		tracks = model.track(heatmap, persist=True, show=False)


		#display image
		cv2.imshow('Thermal',heatmap)

		if recording == True:
			elapsed = (time.time() - start)
			elapsed = time.strftime("%H:%M:%S", time.gmtime(elapsed)) 
			#print(elapsed)
			videoOut.write(heatmap)
		
		keyPress = cv2.waitKey(1)
		if keyPress == ord('a'): #Increase blur radius
			rad += 1
		if keyPress == ord('z'): #Decrease blur radius
			rad -= 1
			if rad <= 0:
				rad = 0

		if keyPress == ord('s'): #Increase threshold
			threshold += 1
		if keyPress == ord('x'): #Decrease threashold
			threshold -= 1
			if threshold <= 0:
				threshold = 0

		if keyPress == ord('d'): #Increase scale
			scale += 1
			if scale >=5:
				scale = 5
			newWidth = width*scale
			newHeight = height*scale
			if dispFullscreen == False and isPi == False:
				cv2.resizeWindow('Thermal', newWidth,newHeight)
		if keyPress == ord('c'): #Decrease scale
			scale -= 1
			if scale <= 1:
				scale = 1
			newWidth = width*scale
			newHeight = height*scale
			if dispFullscreen == False and isPi == False:
				cv2.resizeWindow('Thermal', newWidth,newHeight)

		if keyPress == ord('q'): #enable fullscreen
			dispFullscreen = True
			cv2.namedWindow('Thermal',cv2.WND_PROP_FULLSCREEN)
			cv2.setWindowProperty('Thermal',cv2.WND_PROP_FULLSCREEN,cv2.WINDOW_FULLSCREEN)
		if keyPress == ord('w'): #disable fullscreen
			dispFullscreen = False
			cv2.namedWindow('Thermal',cv2.WINDOW_GUI_NORMAL)
			cv2.setWindowProperty('Thermal',cv2.WND_PROP_AUTOSIZE,cv2.WINDOW_GUI_NORMAL)
			cv2.resizeWindow('Thermal', newWidth,newHeight)

		if keyPress == ord('f'): #contrast+
			alpha += 0.1
			alpha = round(alpha,1)#fix round error
			if alpha >= 3.0:
				alpha=3.0
		if keyPress == ord('v'): #contrast-
			alpha -= 0.1
			alpha = round(alpha,1)#fix round error
			if alpha<=0:
				alpha = 0.0

		if keyPress == ord('m'): #m to cycle through color maps
			colormap += 1
			if colormap == 11:
				colormap = 0

		if keyPress == ord('r') and recording == False: #r to start reording
			videoOut = rec()
			recording = True
			start = time.time()
		if keyPress == ord('t'): #f to finish reording
			recording = False
			elapsed = "00:00:00"

		if keyPress == ord('p'): #f to finish reording
			snaptime = snapshot(heatmap)

		if keyPress == ord('q'):
			break
			capture.release()
			cv2.destroyAllWindows()
		
