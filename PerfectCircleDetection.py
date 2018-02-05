import sys
import numpy as np
import cv2
import time
from skimage.filters import gaussian
from skimage.segmentation import active_contour

frames = list()
comparison_factors = list()
frame_no = 0; step = 2; max_val = 0
capture = cv2.VideoCapture(sys.argv[1])

#capture = cv2.VideoCapture('C:/Users/Jolton/Desktop/Files/Xyken_mini_project/sample.mov')

#Finds active contour given an initial larger contour
#Uses skimage module
def snakes(img,img_gray,gauss_img,circles):
    
    largest_contour = list()
    
    for i in circles[0,:]:
        print("Circle centre x-cord is %d , Circle centre y-cord is %d and the radius is %d \n" %( i[0], i[1], i[2]))
        #draw the center
        cv2.circle(img,(i[0],i[1]),3,(0,0,255),-1,8,0)
        # draw the outer circle
        cv2.circle(img,(i[0],i[1]),(i[2]*1.7).astype("int"),(255,0,0),2,8,0)
        for theta in range(0,360):
            x_on_circle = i[0] + i[2]*1.7*np.cos((np.pi*theta)/180)
            y_on_circle = i[1] + i[2]*1.7*np.sin((np.pi*theta)/180)
            largest_contour.append([x_on_circle, y_on_circle])
    
    largest_contour = np.array(largest_contour) 
    snake = active_contour(gaussian(img_gray, 3),largest_contour, alpha=0.075, beta=10, gamma=0.005)
    return snake

t0 = time.time()
while (capture.isOpened()):
    ret, src = capture.read()
    if(type(src) == type(None)):
        break
    else: 
        frame_no += 1
        print("Frame no: %d" %(frame_no))
        src = cv2.flip(src, -1);
        frame = src.copy()
        frames.append(frame) 
        
        src_gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)    
        src_gauss = cv2.GaussianBlur(src_gray,(9,9),0,0)
        
        #Hough transform for finding the circular boundary which is then made larger to act as initial contour
        circles = cv2.HoughCircles(src_gauss, cv2.HOUGH_GRADIENT,1,np.size(src_gauss,0)/8,param1=200,param2=50,minRadius=0,maxRadius=55)
        if type(circles) == type(None):
            print("No contour detected, the frame is too blurred to detect a shape close to a circle")
            continue
        
        circle_contour = snakes(src,src_gray,src_gauss,circles)
        
        # Approximation of contour for a smoother fit
        eps = 0.0005
        arc = cv2.arcLength(circle_contour.astype("int"),True)
        epsilon = arc * eps
        approx = cv2.approxPolyDP(circle_contour.astype("int"), epsilon, True)
        
        cv2.drawContours(src, [approx], -1, (0,255,0), 2, cv2.LINE_AA)
        cv2.imshow("Contour", src)
        cv2.waitKey(10)
            
        area = cv2.contourArea(approx.astype("int"))
        perimeter = cv2.arcLength(approx.astype("int"),True)
        
        #A perfect circle is defined by the roundness ratio being equal to 1
        Roundness_ratio = 4*np.pi*(area/np.square(perimeter))
        print("Roundness Ratio = %f" %Roundness_ratio)
        
        if Roundness_ratio > max_val:
            max_val = Roundness_ratio
            max_val_frame_no = frame_no-1

capture.release()            

print("The best circle can be found in frame %i" %(max_val_frame_no))

bst_circ_frame = frames[max_val_frame_no]
bst_circ_frame_copy = bst_circ_frame.copy() 
bst_circ_frame_gray = cv2.cvtColor(bst_circ_frame, cv2.COLOR_BGR2GRAY)
bst_circ_frame_gauss = cv2.GaussianBlur(bst_circ_frame_gray,(9,9),0,0)

bst_circ_init_cont = cv2.HoughCircles(bst_circ_frame_gauss,cv2.HOUGH_GRADIENT,1,np.size(bst_circ_frame_gauss,0)/8,param1=200,param2=50,minRadius=0,maxRadius=55)
bst_circ_contour = snakes(bst_circ_frame,bst_circ_frame_gray,bst_circ_frame_gauss,bst_circ_init_cont)

#Using Moments to find the centroid of the contour(center of the circle)
M = cv2.moments(bst_circ_contour.astype("int"))
cx = int(M['m10']/M['m00'])  
cy = int(M['m01']/M['m00'])
approx_pts = cv2.approxPolyDP(bst_circ_contour.astype("int"), epsilon, True)
cv2.drawContours(bst_circ_frame_copy, [approx_pts], -1, (0,0,255), 1, cv2.LINE_AA)
cv2.circle(bst_circ_frame_copy,(cx,cy),3,(0,255,255),-1,8,0)
cv2.imshow("Best Circle Frame", bst_circ_frame_copy)
cv2.waitKey(30)
cv2.imwrite('Best_circle_frame.jpg',bst_circ_frame_copy)

radius = np.round(np.sqrt(np.square(bst_circ_contour[1][1]-cy) + np.square(bst_circ_contour[1][0]-cx)))
print("The radius of the best circle is %d and the roundness ratio is %d" %(radius,Roundness_ratio))

t1 = time.time()
total_time = t1-t0

print("Time Elapse: %d" %total_time)