import numpy as np
import cv2
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
import os


def emotion_cv():
    # Load your model architecture
    model = Sequential()
    model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(48,48,1)))
    model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(1024, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(7, activation='softmax'))

    model.load_weights('pretrained/model.h5')

    # Prevents openCL usage and unnecessary logging messages
    cv2.ocl.setUseOpenCL(False)

    # Load the face cascade classifier
    face_cascade = cv2.CascadeClassifier('pretrained/haarcascade_frontalface_default.xml')

    # Dictionary which assigns each label an emotion (alphabetical order)
    emotion_dict = {0: "Angry", 1: "Disgusted", 2: "Fearful", 3: "Happy", 4: "Neutral", 5: "Sad", 6: "Surprised"}

    # Start the webcam feed
    cap = cv2.VideoCapture(0)
    faces=()

    while True:

        if faces==():
            # Capture a single frame
            ret, frame = cap.read()
            
            if not ret:
                break
            
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Detect faces in the frame
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)
            
            for (x, y, w, h) in faces:
                # Draw a rectangle around the detected face
                cv2.rectangle(frame, (x, y-50), (x+w, y+h+10), (255, 0, 0), 2)
                roi_gray = gray[y:y + h, x:x + w]
                cropped_img = np.expand_dims(np.expand_dims(cv2.resize(roi_gray, (48, 48)), -1), 0)
                
                # Get the emotion prediction for the detected face
                prediction = model.predict(cropped_img)
                max_index = int(np.argmax(prediction))
                
                # Display the emotion label on the frame
                cv2.putText(frame, emotion_dict[max_index], (x+20, y-60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
                
                # Store the output for later use (you can replace this with your database logic)
                output_to_store = emotion_dict[max_index]
                print("Detected emotion:", output_to_store)
                break
                # Add your database storage logic here
        else:        
            break

    # Release the webcam and close windows
    cap.release()
    cv2.destroyAllWindows()
