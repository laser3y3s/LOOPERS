import cv2
import numpy as np
import argparse
import pyrebase

config = {
    "apiKey": "AIzaSyBSr6P73_gJaa3v-TjnTXNbey-bRzSn5bc",
    "authDomain": "carparking-fa9aa.firebaseapp.com",
    "databaseURL": "https://carparking-fa9aa.firebaseio.com",
    "storageBucket": "carparking-fa9aa.appspot.com"
}

firebase = pyrebase.initialize_app(config)
db = firebase.database()

#data = {"count": 55}
#db.child("parking").update(data)

change=0
# top left, top right, bottom left, bottom right
pts = [(0,0),(0,0),(0,0),(0,0)]
pointIndex = 0
firstFrame = None
firstFrame1 =  None

ap = argparse.ArgumentParser()
ap.add_argument("-a", "--min-area", type=int, default=2000, help="minimum area size")
args = vars(ap.parse_args())

cam = cv2.VideoCapture(2)

_,img = cam.read()
_,img = cam.read()

# Aspect ratio for an A4 sheet. 1:1.414
# 500 * 1.414 = 707, that is why I chose this size.
ASPECT_RATIO = (510,570)

pts2 = np.float32([[0,0],[ASPECT_RATIO[1],0],[0,ASPECT_RATIO[0]],[ASPECT_RATIO[1],ASPECT_RATIO[0]]])
# mouse callback function
def draw_circle(event,x,y,flags,param):
	global img
	global pointIndex
	global pts

	if event == cv2.EVENT_LBUTTONDBLCLK:
		cv2.circle(img,(x,y),2,(255,0,0),-1)
		print(x,y)
		pts[pointIndex] = (x,y)
		pointIndex = pointIndex + 1

def count(cnt):
	remain = 6 - cnt
	global change
	if remain!=change:
		data = {"count": remain}
		db.child("parking").update(data)
		change =remain
	return(remain)


def selectFourPoints():
	global img
	global pointIndex

	print ("Please select 4 points, by double clicking on each of them in the order: \n\
	top left, top right, bottom left, bottom right.")

	while(pointIndex != 4):
		cv2.imshow('image',img)
		key = cv2.waitKey(20) & 0xFF
		if key == 27:
			return False

	return True
# Create a black image, a window and bind the function to window
# img = np.zeros((512,512,3), np.uint8)

cv2.namedWindow('image')
cv2.setMouseCallback('image',draw_circle)

while(1):
	if(selectFourPoints()):

		# The four points of the A4 paper in the image
		pts1 = np.float32([\
			[pts[0][0],pts[0][1]],\
			[pts[1][0],pts[1][1]],\
			[pts[2][0],pts[2][1]],\
			[pts[3][0],pts[3][1]] ])

		M = cv2.getPerspectiveTransform(pts1,pts2)

		while(1):

			_,frame = cam.read()

			dst = cv2.warpPerspective(frame,M,(707,500))


			flag1=1
			flag2=1
			flag3=1
			flag4=1
			flag5=1
			flag6=1
			flagbig=1
			counter=0

			if firstFrame1 is None:
				firstFrame1 = dst
				continue
			gray = cv2.cvtColor(dst,cv2.COLOR_BGR2GRAY)
			gray  = cv2.GaussianBlur(gray,(27,27),0)

			if firstFrame is None:
				firstFrame = gray
				continue

			frameDelta = cv2.absdiff(firstFrame,gray)
			cv2.imshow("Test",frameDelta)
			_,thresh = cv2.threshold(frameDelta,30,255,cv2.THRESH_BINARY)
			thresh = cv2.dilate(thresh,None,iterations=1)

			#ROI 1
			roi1 = thresh[43:167,81:154]               #im[y1:y2, x1:x2] , (x1,y1) as the top-left vertex and (x2,y2) as the bottom-right vertex of a rectangle region
			cv2.imshow("ROI 1",roi1)
			(_,cnts,_)=cv2.findContours(roi1.copy(),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)

			for c in cnts:
				if cv2.contourArea(c) < args["min_area"] :
					continue
				flag1=2
			if flag1 is 1:
				cv2.circle(firstFrame1,(81,43),20,(0,255,0),-1)
			elif flag1 is 2:
				counter=counter+1
				cv2.circle(firstFrame1,(81,43),20,(0,0,255),-1)

			#ROI 2
			roi2 = thresh[36:162,252:329]  # im[y1:y2, x1:x2] , (x1,y1) as the top-left vertex and (x2,y2) as the bottom-right vertex of a rectangle region
			cv2.imshow("ROI 2", roi2)
			(_, cnts, _) = cv2.findContours(roi2.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

			for c in cnts:
				if cv2.contourArea(c) < args["min_area"]:
					continue
				flag2 = 2
			if flag2 is 1:
				cv2.circle(firstFrame1, (252,36), 20, (0, 255, 0), -1)
			elif flag2 is 2:
				counter = counter + 1
				cv2.circle(firstFrame1, (252,36), 20, (0, 0, 255), -1)

			# ROI 3
			roi3 = thresh[36:163,424:505]     # im[y1:y2, x1:x2] , (x1,y1) as the top-left vertex and (x2,y2) as the bottom-right vertex of a rectangle region
			cv2.imshow("ROI 3", roi3)
			(_, cnts, _) = cv2.findContours(roi3.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

			for c in cnts:
				if cv2.contourArea(c) < args["min_area"]:
					continue
				flag3 = 2
			if flag3 is 1:
				cv2.circle(firstFrame1, (424,36), 20, (0, 255, 0), -1)
			elif flag3 is 2:
				counter = counter + 1
				cv2.circle(firstFrame1, (424,36), 20, (0, 0, 255), -1)

			cv2.imshow("Status",firstFrame1)

			cv2.setMouseCallback('output', draw_circle)

			key = cv2.waitKey(10) & 0xFF
			if key == 27:
				break

			# ROI 4
			roi4 = thresh[344:468,82:151]     # im[y1:y2, x1:x2] , (x1,y1) as the top-left vertex and (x2,y2) as the bottom-right vertex of a rectangle region
			cv2.imshow("ROI 3", roi4)
			(_, cnts, _) = cv2.findContours(roi4.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

			for c in cnts:
				if cv2.contourArea(c) < args["min_area"]:
					continue
				flag4 = 2
			if flag4 is 1:
				cv2.circle(firstFrame1, (82,344), 20, (0, 255, 0), -1)
			elif flag4 is 2:
				counter = counter + 1
				cv2.circle(firstFrame1, (82,344), 20, (0, 0, 255), -1)

			cv2.imshow("Status",firstFrame1)

			cv2.setMouseCallback('output', draw_circle)

			key = cv2.waitKey(10) & 0xFF
			if key == 27:
				break

			# ROI 5
			roi5 = thresh[344:472,253:326]     # im[y1:y2, x1:x2] , (x1,y1) as the top-left vertex and (x2,y2) as the bottom-right vertex of a rectangle region
			cv2.imshow("ROI 3", roi5)
			(_, cnts, _) = cv2.findContours(roi5.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

			for c in cnts:
				if cv2.contourArea(c) < args["min_area"]:
					continue
				flag5 = 2
			if flag5 is 1:
				cv2.circle(firstFrame1, (253,344), 20, (0, 255, 0), -1)
			elif flag5 is 2:
				counter = counter + 1
				cv2.circle(firstFrame1, (253,344), 20, (0, 0, 255), -1)

			cv2.imshow("Status",firstFrame1)

			cv2.setMouseCallback('output', draw_circle)

			key = cv2.waitKey(10) & 0xFF
			if key == 27:
				break

			# ROI 6
			roi6 = thresh[346:471,423:496]     # im[y1:y2, x1:x2] , (x1,y1) as the top-left vertex and (x2,y2) as the bottom-right vertex of a rectangle region
			cv2.imshow("ROI 3", roi6)
			(_, cnts, _) = cv2.findContours(roi6.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

			for c in cnts:
				if cv2.contourArea(c) < args["min_area"]:
					continue
				flag6 = 2
			if flag6 is 1:
				cv2.circle(firstFrame1, (423,346), 20, (0, 255, 0), -1)
			elif flag6 is 2:
				counter = counter + 1
				cv2.circle(firstFrame1, (423,346), 20, (0, 0, 255), -1)

			cv2.imshow("Status",firstFrame1)

			cv2.setMouseCallback('output', draw_circle)

			key = cv2.waitKey(10) & 0xFF
			if key == 27:
				break

            #BIG ROI
			bigroi = thresh[186:332,5:570]  # im[y1:y2, x1:x2] , (x1,y1) as the top-left vertex and (x2,y2) as the bottom-right vertex of a rectangle region
			cv2.imshow("BIG ROI", bigroi)
			(_, cnts, _) = cv2.findContours(bigroi.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

			for c in cnts:
				if cv2.contourArea(c) < args["min_area"]:
					continue
				counter = counter + 1
				flagbig = 2
			if flagbig is 1:
				cv2.circle(firstFrame1, (5,186), 20, (0, 255, 0), -1)
			elif flagbig is 2:
				cv2.circle(firstFrame1, (5,186), 20, (0, 0, 255), -1)

			cv2.imshow("Status", firstFrame1)
			cnt = count(counter)
			cv2.putText(firstFrame1, "%s" % cnt, (200, 500), cv2.FONT_HERSHEY_SIMPLEX, 2, 255)
			cv2.putText(dst, "%s" % cnt, (200, 500), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 10)

			cv2.imshow("Status", firstFrame1)
			cv2.imshow("Status - 2", dst)

			cv2.setMouseCallback('output', draw_circle)

			key = cv2.waitKey(10) & 0xFF
			if key == 27:
				break
	else:
		print ("Exit")

	break


	# cv2.imshow('image',img)
	# if cv2.waitKey(20) & 0xFF == 27:
	# 	break
cam.release()
cv2.destroyAllWindows()