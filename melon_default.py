#!/usr/bin/env python3
'''
Les Wright 21 June 2023
https://youtube.com/leslaboratory
A Python program to read, parse, and display thermal data from the Topdon TC001 Thermal camera!
'''
print('Les Wright 21 June 2023')
print('https://youtube.com/leslaboratory')
print('A Python program to read, parse, and display thermal data from the Topdon TC001 Thermal camera!')
print('')
print('Tested on Debian all features are working correctly')
print('This will work on the Pi; however, a number of workarounds are implemented!')
print('Seemingly there are bugs in the compiled version of cv2 that ships with the Pi!')
print('')

import cv2
import numpy as np
import argparse
import io

def is_raspberrypi():
    try:
        with io.open('/sys/firmware/devicetree/base/model', 'r') as m:
            if 'raspberry pi' in m.read().lower(): return True
    except Exception:
        pass
    return False

isPi = is_raspberrypi()

parser = argparse.ArgumentParser()
parser.add_argument("--device", type=int, default=0, help="Video Device number e.g. 0, use v4l2-ctl --list-devices")
args = parser.parse_args()
	
if args.device:
    dev = args.device
else:
    dev = 0
	
#init video
cap = cv2.VideoCapture('/dev/video'+str(dev), cv2.CAP_V4L)

if isPi == True:
    cap.set(cv2.CAP_PROP_CONVERT_RGB, 0.0)
else:
    cap.set(cv2.CAP_PROP_CONVERT_RGB, False)

# 256x192 General settings
width = 256  # Sensor width
height = 192  # Sensor height
scale = 3     # Scale multiplier
newWidth = width * scale 
newHeight = height * scale
alpha = 1.0   # Contrast control (1.0-3.0)
colormap = 0
font = cv2.FONT_HERSHEY_SIMPLEX
dispFullscreen = False
cv2.namedWindow('Thermal', cv2.WINDOW_GUI_NORMAL)
cv2.resizeWindow('Thermal', newWidth, newHeight)
rad = 0  # blur radius

while(cap.isOpened()):
    # Capture frame-by-frame
    ret, frame = cap.read()
    if ret == True:
        imdata, thdata = np.array_split(frame, 2)

        hi = thdata[96][128][0]
        lo = thdata[96][128][1]
        lo = lo * 256
        rawtemp = hi + lo
        temp = (rawtemp / 64) - 273.15
        temp = round(temp, 2)

        # Convert the real image to RGB
        bgr = cv2.cvtColor(imdata, cv2.COLOR_YUV2BGR_YUYV)
        bgr = cv2.convertScaleAbs(bgr, alpha=alpha)  # Contrast
        bgr = cv2.resize(bgr, (newWidth, newHeight), interpolation=cv2.INTER_CUBIC)  # Scale up!
        if rad > 0:
            bgr = cv2.blur(bgr, (rad, rad))

        # Apply colormap
        if colormap == 0:
            heatmap = cv2.applyColorMap(bgr, cv2.COLORMAP_JET)
        # (이하 중략)
        # ... (이하 생략)
        # ...

        # Draw crosshairs
        cv2.line(heatmap, (int(newWidth/2), int(newHeight/2)+20),
                 (int(newWidth/2), int(newHeight/2)-20), (255, 255, 255), 2)  # vline
        cv2.line(heatmap, (int(newWidth/2)+20, int(newHeight/2)),
                 (int(newWidth/2)-20, int(newHeight/2)), (255, 255, 255), 2)  # hline

        cv2.line(heatmap, (int(newWidth/2), int(newHeight/2)+20),
                 (int(newWidth/2), int(newHeight/2)-20), (0, 0, 0), 1)  # vline
        cv2.line(heatmap, (int(newWidth/2)+20, int(newHeight/2)),
                 (int(newWidth/2)-20, int(newHeight/2)), (0, 0, 0), 1)  # hline

        # Show temp
        cv2.putText(heatmap, str(temp)+' C', (int(newWidth/2)+10, int(newHeight/2)-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 2, cv2.LINE_AA)
        cv2.putText(heatmap, str(temp)+' C', (int(newWidth/2)+10, int(newHeight/2)-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1, cv2.LINE_AA)

        # Display image
        cv2.imshow('Thermal', heatmap)

        # 종료 키
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# 종료
cap.release()
cv2.destroyAllWindows()
