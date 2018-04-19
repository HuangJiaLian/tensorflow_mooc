#coding:utf-8
import tensorflow as tf
import numpy as np
from PIL import Image
import mnist_backward
import mnist_forward
import sys
import cv2
import os

PaintSize = 350
BgColor = (255, 255, 255)
PaintColor = (0, 0, 0)
StrokeWeight = 20


drawing = False
start = (-1, -1)
lastPoint = (-1,-1)

# Initialise background color to white.
img = np.full((PaintSize, PaintSize,3), BgColor, dtype=np.uint8)

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
			cv2.line(img,lastPoint,(x, y), PaintColor, StrokeWeight)
	elif event == cv2.EVENT_LBUTTONUP:
		drawing = False
	lastPoint = (x, y)



# Image Processing Step
def pre_pic(picName):
	img = Image.open(picName)
	reIm = img.resize((28,28), Image.ANTIALIAS) # Reshape the image to the size of 28x28

	im_arr = np.array(reIm.convert('L')) # convert to grayscale 
	cv2.imshow("Little",im_arr)
	threshold = 50

	for i in range(28):
		for j in range(28):
			im_arr[i][j] = 255 - im_arr[i][j]
			if (im_arr[i][j] < threshold):
				im_arr[i][j] = 0
			else:
				im_arr[i][j] = 255

	nm_arr = im_arr.reshape([1,784])
	nm_arr = nm_arr.astype(np.float32)
	img_ready = np.multiply(nm_arr, 1.0/255.0)
	return img_ready


# Use trained model
# how the process worked ?
def restore_model(testPicArr):
	with tf.Graph().as_default() as tg:
		x = tf.placeholder(tf.float32, [None, mnist_forward.INPUT_NODE])
		y = mnist_forward.forward(x, None)
		preValue = tf.argmax(y,1)

		variable_averages = tf.train.ExponentialMovingAverage(mnist_backward.MOVING_AVERAGE_DECAY)
		variables_to_restore = variable_averages.variables_to_restore()
		saver = tf.train.Saver(variables_to_restore)

		with tf.Session() as sess:
			ckpt = tf.train.get_checkpoint_state(mnist_backward.MODEL_SAVE_PATH)
			if ckpt and ckpt.model_checkpoint_path:
				saver.restore(sess, ckpt.model_checkpoint_path)
				preValue = sess.run(preValue, feed_dict={x:testPicArr})
				return preValue
			else:
				print("No checkpoint file found")
				return -1


def application():
	global img 
	cv2.namedWindow('Press \'s\' to Save,\'c\' to clear')
	# assign callback
	cv2.setMouseCallback('Press \'s\' to Save,\'c\' to clear', mouse_event)

	print("Press q or Esc to quit the program:")
	while True:
		cv2.imshow('Press \'s\' to Save,\'c\' to clear', img)
		key = cv2.waitKey(20)

		if key == 27 or key == 113: # break when press ESC/q
			break
		elif key == 115: # s for 'save'
			imgName = './pic/handWrite.png'
			cv2.imwrite(imgName, img)
			print(imgName + " saved")
			testPicArr = pre_pic(imgName)
			preValue = restore_model(testPicArr)
			print ("The prediction number is:", preValue)
		elif key == 99: # c for 'clear'
			img = np.full((PaintSize, PaintSize,3), BgColor, dtype=np.uint8)
		else:
			pass


def main():
	application()

if __name__ == '__main__':
	main()