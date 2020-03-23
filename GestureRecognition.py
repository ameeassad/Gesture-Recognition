import sys
import cv2
import numpy as np
import imutils


def gesture(filename):
    
    #load file
    img = cv2.imread(filename)
    
    #resizing
    height= 300
    img = imutils.resize(img, height)
    width = img.shape[1]
    
    #Create Binary Image
    #blurring
    img = cv2.GaussianBlur(img,(5,5),0)
    
    RED, GREEN, BLUE = (2, 1, 0)
    reds = img[:, :, RED]
    greens = img[:, :, GREEN]
    blues = img[:, :, BLUE]

    mask = ((greens < 35) | (reds >= greens) | (blues >= greens)) * 255

    contour_file = "contoured/"+filename
    cv2.imwrite(contour_file, mask)
    binary_img = cv2.imread(contour_file)
    
    #binary image processing
    binary_img = cv2.cvtColor(binary_img, cv2.COLOR_BGR2GRAY)
    binary_img = cv2.bilateralFilter(binary_img, 11, 17, 17)  #blur
    binary_img = cv2.dilate(binary_img, None) #smooth
    
    #Get Contours
    contours,hierarchy=cv2.findContours(binary_img,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea)
    cnt = contours[len(contours)-1]
    
    #Remove Arm
    #find extreme top point
    top_y = cnt[cnt[:, :, 1].argmin()][0][1]
    if (top_y > height/4):
        top_half = False
        i = 0
        #delete some contour points at lower border
        for row in cnt:
            if (row[0][1] > 9*height/10):
                cnt=np.delete(cnt,i,0)
            else:
                i+=1
    else:
        top_half = True
        i = 0
        #delete 1/3 of the contour points 
        for row in cnt:
            if (row[0][1] > 2*height/3):
                cnt=np.delete(cnt,i,0)
            else:
                i+=1
    
    #Smooth Contours
    approx = cv2.approxPolyDP(cnt,0.01*cv2.arcLength(cnt,True),True)
    cnt = np.copy(approx)

    #Calculate Location
    #compute the centre of contour
    M = cv2.moments(cnt)
    #calculate x,y coordinate of center
    cX = int(M["m10"]/M["m00"])
    cY = int(M["m01"] / M["m00"])
    
    
    #calculate quadrant
    if (top_half):
        if (cX < width/3):
            position = "top-left"
        elif (cX > 2*width/3):
            position = "top-right"
        else:
            position = "unsure position"
    else:
        if (cX < width/3):
            position = "bottom-left"
        elif (cX > 2*width/3):
            position = "bottom-right"
        else:
            position = "unsure position"
    if (cX < 3*width/5 and cX > 2*width/5 and cY < 2*height/3 and cY > height/3):
        position = "centre"
        
    #Calculate Gesture
    hull = cv2.convexHull(cnt,returnPoints = False)
    # convexity defects - taken from opencv documentation
    if len(hull) > 3:
        defects = cv2.convexityDefects(cnt,hull)
        count = 0
        for i in range(defects.shape[0]):
            s,e,f,d = defects[i,0]
            far = tuple(cnt[f][0])
            
            if far[1]<cY:
                count+=1
            cv2.circle(img,far,5,[0,0,255],-1)
    else:
        gesture = "unknown gesture"
        
        
    if (count >= 3 and count < 6):
        gesture = "splay"
    elif (count < 2):
        gesture = "fist"
    else:
        gesture = "unknown gesture"
    
    
    #Labeling
    #draw it on the image
    cv2.drawContours(img, [cnt], -1, (0, 255, 0), 2)
    cv2.circle(img, (cX, cY), 7, (255, 255, 255), -1)
    cv2.putText(img, gesture + " at " + position, (0, 15),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    
    
    newname = "output/"+filename
    cv2.imwrite(newname, img)
    
    return (gesture, position), (cX, cY)
    
    
    #Some Testing  
    #show convex hull
    #hull = cv2.convexHull(cnt)
    #cv2.drawContours(img,[hull],0,(0,0,255),2)

def main():
    
    password = (("fist", "centre"), ("splay", "top-right"))
    
    comb1, pos_1 = gesture("user/20.png")
    comb2, pos_2 = gesture("user/21.png")

    if (pos_2[0] < pos_1[0]):
        rel_pos = "further left"
    elif (pos_2[0] > pos_1[0]):
        rel_pos = "further right"
    else:
        rel_pos = "same vertical point"
        
    if (pos_2[1] < pos_1[1]):
        rel_pos += " and further up."
    elif (pos_2[1] > pos_1[1]):
        rel_pos += " and further down."
    else:
        rel_pos += " and same horizontal point."
    print("The second gesture is", rel_pos)
            
    
    if ((comb1, comb2)==password):
        print("UNLOCKED")
    else:
        print("Incorrect password.")
    
    
if __name__ == "__main__":
    main()