from datetime import datetime
import sys
import logging
import os
import cv2
from utils import write_image, key_action, init_cam
from tensorflow.keras.models import load_model
import numpy as np

if __name__ == "__main__":

    # folder to write images to
    out_folder = sys.argv[1]

    # maybe you need this
    os.environ['KMP_DUPLICATE_LIB_OK']='True'

    logging.getLogger().setLevel(logging.INFO)
   
    # also try out this resolution: 640 x 360
    webcam = init_cam(640, 480)
    key = None

    model = load_model('../models/signs_model.keras')
    classes = ['speed_30','speed_50','speed_70', 'speed_80']
    print(model.summary())
    try:
        # q key not pressed 
        while key != 'q':
            # Capture frame-by-frame
            ret, frame = webcam.read()
            # fliping the image 
            frame = cv2.flip(frame, 1)
   
            # draw a [224x224] rectangle into the frame, leave some space for the black border 
            offset = 2
            width = 224
            x = 160
            y = 120
            cv2.rectangle(img=frame, 
                          pt1=(x-offset,y-offset), 
                          pt2=(x+width+offset, y+width+offset), 
                          color=(0, 0, 0), 
                          thickness=2
            )     
            
            # get key event
            key = key_action()
   
            if key == 'r':
                print('record')
                print(datetime.now().strftime('%d.%m.%Y %H:%M:%S'))
                image = frame[y:y+width, x:x+width, :]
                write_image(out_folder, image) 

            if key == 'space':
                # write the image without overlay
                # extract the [224x224] rectangle out of it
                print(datetime.now().strftime('%d.%m.%Y %H:%M:%S'))
                image = frame[y:y+width, x:x+width, :]
                image = cv2.resize(image, (224, 224))
                #image = image.astype("float32") / 255.0  
                # Add batch dimension → (1, 224, 224, 3)
                image_exp = np.expand_dims(image, axis=0)
                prediction = model.predict(image_exp)
                print("Raw prediction:", prediction)

                predicted_label = classes[np.argmax(prediction, axis=1)[0]]
                confidence = np.max(prediction) * 100
                print(f"Predicted: {predicted_label} ({confidence:.2f}%)")

                #write_image(out_folder, image) 

            # disable ugly toolbar
            cv2.namedWindow('frame', flags=cv2.WINDOW_GUI_NORMAL)              
            
            # display the resulting frame
            cv2.imshow('frame', frame)            
            
    finally:
        # when everything done, release the capture
        logging.info('quit webcam')
        webcam.release()
        cv2.destroyAllWindows()
