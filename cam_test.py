import cv2
import numpy as np
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
#cap.set(cv2.CAP_PROP_FPS, 30)
#cap.set(cv2.CAP_PROP_FRAME_WIDTH, 3840)
#cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 2160)
video_fps = int(cap.get(cv2.CAP_PROP_FPS))
video_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
video_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
print(video_fps, video_width, video_height)
display = np.zeros((1, 1))
isDraw = False
while True:
	ret, img = cap.read()
	if not ret:
		break
	if isDraw:
		cv2.imshow("image", img)
	else:
		cv2.imshow("image", display)
	key = cv2.waitKey(1)
	if key & 0xFF == ord('d'): # esc
	    isDraw = not isDraw

	if key & 0xFF == ord('x'):
		break	    


