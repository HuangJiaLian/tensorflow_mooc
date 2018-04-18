import cv2
import numpy as np 
import datetime
import os

drawing = False
start = (-1, -1)
lastPoint = (-1,-1)

# mouse callback
def mouse_event(event, x, y, flags, param):
	global drawing, img, lastPoint
	# Left button down
	if event == cv2.EVENT_LBUTTONDOWN:
		drawing = True
		lastPoint = (x, y)
		start = (x, y)
	elif event == cv2.EVENT_MOUSEMOVE:
		if drawing:
			# -1: fill
			# cv2.circle(img, (x,y),5,200,-1)
			cv2.line(img,lastPoint,(x, y), 200, 5)
	elif event == cv2.EVENT_LBUTTONUP:
		drawing = False
	lastPoint = (x, y)


img = np.zeros((512,512,1), np.uint8)
cv2.namedWindow('Draw a number from 0 to 9:')

# assign callback
cv2.setMouseCallback('Draw a number from 0 to 9:', mouse_event)

# List all possible events
events = [i for i in dir(cv2) if 'EVENT' in i]
print(events)

while True:
	cv2.imshow('Draw a number from 0 to 9:', img)

	if cv2.waitKey(20) == 27 or cv2.waitKey(20) == 113: # break when press ESC/q
		break
	elif cv2.waitKey(20) == 115: # s for 'save'
		dt = str(datetime.datetime.now())
		imgName = 'img_'+dt+'.png'
		cv2.imwrite(imgName, img)
		# print(imgName),
		print(imgName + " saved")
		# print("%s Saved." %(imgName)

