import cv2  
import time
from threading import Thread
import tensorflow as tf 
import numpy as np 

class VideoClassification:

    # Cat/Dog Classifier
    # Due to github file storage restrictions, please download the trained model from the link below:
    # https://drive.google.com/file/d/1mYyAPOjmhpK3wz88lET5mb2fyRBI8993/view?usp=sharing
    def __init__(self,src=0):

        # class attributes related to the model and the video
        # Ignore the CUDA dependency related errors when the tensorflow model is loaded
        self.src = src
        self.model = tf.keras.models.load_model("models/cat_dog.h5")
        self.capture = cv2.VideoCapture(src)
        self._, self.img = self.capture.read()
        self.label = 'None'
        self.cTime = 0 
        self.pTime = 0

        # Thread for running the CNN classification of each video frame
        self.t = Thread(target=self.frameClassify)
        self.t.daemon = True
        self.t.start()

    # private method to predict the class based on the probability retured by the CNN
    def __prediction(self, result):
        if result[0][0] < 0.5:
            return 'Cat'
        else:
            return 'Dog'

    # Thread function to classify each video frame using CNN
    # Heavy video processing functionality should be defined here
    def frameClassify(self):

        while True:
            x = cv2.resize(self.img, (224, 224),
                interpolation = cv2.INTER_NEAREST)
            x = x/255.
            result = self.model.predict(np.expand_dims(x,axis=0))
            self.label = self.__prediction(result)
            time.sleep(1/60)
        return

    # Running the read/display of the video on the main thread
    def display(self):

        while True:  
            
            self.img = cv2.flip(self.capture.read()[1],1)
            self.cTime = time.time()
            fps = 1 / (self.cTime - self.pTime)
            self.pTime = self.cTime
            cv2.putText(self.img, "FPS: "+str(int(fps)), (10, 70), cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 2)
            cv2.putText(self.img, "Class: "+self.label, (10, 110), cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 2)
            cv2.imshow('Video', self.img)  
            k = cv2.waitKey(30) & 0xff  
            if k==27:  
                break

        self.capture.release()

if __name__ == '__main__':
    
    video = VideoClassification(src='videos/cat_dog.mp4')
    try:
        video.display()
    except:
        pass