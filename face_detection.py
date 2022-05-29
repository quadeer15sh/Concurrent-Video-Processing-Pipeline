import cv2  
import time
from threading import Thread
import numpy as np 

class FaceDetection:

    def __init__(self,src=0):

        # class attributes related to the model and the video
        self.src = src
        self.model = cv2.CascadeClassifier('models/haarcascade_frontalface_default.xml')
        self.capture = cv2.VideoCapture(src)
        self._, self.img = self.capture.read()
        self.cTime = 0 
        self.pTime = 0

        # Thread for running the CNN classification of each video frame
        self.t = Thread(target=self.detectFaces)
        self.t.daemon = True
        self.t.start()

    # Thread function to detect faces in each video frame using haar cascades
    # Heavy video processing functionality should be defined here
    def detectFaces(self):

        while True:
            gray = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)  
            self.faces = self.model.detectMultiScale(gray, 1.1, 4)  
            time.sleep(1/60)
        return

    # Running the read/display of the video on the main thread
    def display(self):

        while True:  

            self.img = cv2.flip(self.capture.read()[1],1)
            try:
                for (x, y, w, h) in self.faces:  
                    cv2.rectangle(self.img, (x, y), (x+w, y+h), (255, 0, 0), 2)  
            except:
                pass

            self.cTime = time.time()
            fps = 1 / (self.cTime - self.pTime)
            self.pTime = self.cTime
            cv2.putText(self.img, "FPS: "+str(int(fps)), (10, 70), cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 2)
            cv2.imshow('Video', self.img)  
            k = cv2.waitKey(30) & 0xff  
            if k==27:  
                break

        self.capture.release()

if __name__ == '__main__':
    
    video = FaceDetection(src='videos/man.mp4')
    try:
        video.display()
    except:
        pass