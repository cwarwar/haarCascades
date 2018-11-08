#Importing the module
import cv2

#Loading two examples, one will detect the face, and the another to the eyes
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')

#Opening the conection with your camera  
#the parameter (0) may vary, according how many video inputs you have
cap = cv2.VideoCapture(0)

try:
    #Looping
    while True:
        #Reading the camera
        ret, img = cap.read()

        #Converting the input to black & white
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        #Where the magic happens, here the face will be detected
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        #For each face
        for(x, y, w, h) in faces:
            #Draw a rectangle in the face
            cv2.rectangle(img, (x,y), (x+w, y+h), (255,0,0), 2)

            #Get both colored and black & white images
            roi_gray = gray[y:y+h, x:x+w]
            roi_color = img[y:y+h, x:x+w]

            #Where the magic happens again, here the eyes will be detected
            eyes = eye_cascade.detectMultiScale(roi_gray)

            #For each eye
            for(ex, ey, ew, eh) in eyes:
                #Draw a rectangle in the eye
                cv2.rectangle(roi_color, (ex, ey), (ex+ew, ey+eh), (0,255,0), 2)

        #Image output
        cv2.imshow('img', img)
        #Stop video capture (ctrl+c)
        k = cv2.waitKey(30) & 0xff
        if k == 27:
            break

except:
    print('An error has occurred')
finally:
    #Releasing the video input and closing the window
    cap.release()
    cv2.destroyAllWindows()
