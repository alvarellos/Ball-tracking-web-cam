import cv2
import numpy as np
 
cap = cv2.VideoCapture(0)
 
#changed color to green
#lower_green = np.array([30,80,80])
#upper_green = np.array([80,255,255])

lower_green = np.array([165,72,0])
#lower_green = np.array([165,0,0])
#upper_green = np.array([180,255,180])
upper_green = np.array([245,255,180])
 
frame_count = 0
points = []
 
ret, frame = cap.read()
Height, Width = frame.shape[:2]
center = int(Height/2), int(Width/2) # you need to do this once -moved outside the loop
 
min_valid_area = 700 #min area that a valid contour is supposed to have
max_line_lenght = 80 #lenght of tracking line, increase to have longer lines
 
while cap.isOpened():
    ret, frame = cap.read()
    
    if ret: #only run this if there is a valid cam frame...
        hsv_img = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv_img, lower_green, upper_green)
        blurred = cv2.GaussianBlur(mask, (3, 3), 0) #added blur to remove fuzzy contour detections
        
        _, contours, _ = cv2.findContours(blurred.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if len(contours) > 0:
            main_contour = max(contours, key=cv2.contourArea)
            main_contour_area = cv2.contourArea(main_contour)
            
            if main_contour_area >= min_valid_area:
                (contour_x, contour_y), contour_radius = cv2.minEnclosingCircle(main_contour)
                cv2.circle(frame, (int(contour_x), int(contour_y)), int(contour_radius),(0, 0, 255), 2)
                cv2.circle(frame, (int(contour_x), int(contour_y)), int(contour_radius / 10),(0, 255, 0), -1)
                #no need to calculate moments when you already calculated min Enclosing Circle...
                
                if len(points) >= max_line_lenght: #if the line gets to long, remove the first element
                    points.pop(0)
                    
                points.append((int(contour_x), int(contour_y)))
                # for i in range(1, len(points)): #no need for try, exept if you put it in the right place
                  #  cv2.line(frame, points[i - 1], points[i], (0, 255, 0), 2)
                    
                frame_count = 0
            else:
                frame_count += 1
        else:
            frame_count += 1
  
        if frame_count == 10:
            points = []
            frame_count = 0
                
        frame = cv2.flip(frame, 1)
        cv2.imshow('Object tracker', frame)
    else:
        print("Cam frame error...")
    
    if cv2.waitKey(1) == 13:
        break
print('closing')
cap.release()
cv2.destroyAllWindows()