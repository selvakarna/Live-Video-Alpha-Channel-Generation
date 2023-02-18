import cv2
import numpy as np
fac_dect = cv2.CascadeClassifier('haarcascade_frontalface_alt2.xml')
#  * 
#  *
#  *  Created on: 23-Feb-2023
#  *      Author: selva karna
#  */
def rgb2rgba(rgb):
      # type: (np.ndarray) -> np.ndarray
  
      assert rgb.ndim == 3, "rgb must be 3 dimensional"
      assert rgb.shape[2] == 3, "rgb shape must be (H, W, 3)"
      assert rgb.dtype == np.uint8, "rgb dtype must be np.uint8"

      a = np.full(rgb.shape[:2], 255, dtype=np.uint8)
      rgb[:,:,2]=a  ### ALpha Value change here 
      rgba = np.dstack((rgb, a))
        # a = np.full(rgb.shape[:2], 255, dtype=np.uint8)
      # rgba = np.dstack((rgb, rgb[:,:,1]+0.28))
      # rgb[:,:,2]=a
      return rgba
def livecap():
    vid = cv2.VideoCapture(0)
  
    while(True):
        
        # Capture the video frame
        # by frame
        ret, frame = vid.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = fac_dect.detectMultiScale(gray,1.1, 4 )
        for (x,y, w, h) in faces:
          cv2.rectangle(frame, pt1 = (x,y),pt2 = (x+w, y+h), color = (255,0,0),thickness =  3)
        ###
        ###########RGB 2 RGBA Conversion
        rgba=rgb2rgba(frame)
        print("The alpha dimenstion is ", np.shape(rgba))
    
              
   

        cv2.imshow('frame', rgba)
        # return frame
 
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
       
    
    # After the loop release the cap object
    vid.release()
    # Destroy all the windows
    cv2.destroyAllWindows()
    

livecap()
