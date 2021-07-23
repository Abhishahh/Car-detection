import cv2
video = cv2.VideoCapture('comp.mp4')
car_classifier = cv2.CascadeClassifier('car_detector.xml')
pedestrians_classifier = cv2.CascadeClassifier('haarcascade_fullbody.xml')

while True:
    successful_video_read, frame = video.read()
    grayscale = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    car_coordinates = car_classifier.detectMultiScale(grayscale) 
    pedestrians_cordinates = pedestrians_classifier.detectMultiScale(grayscale)
    for (x,y,w,h) in car_coordinates:
        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),5)
    for (x,y,w,h) in  pedestrians_cordinates:
        cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),10)
    cv2.imshow('live', frame)
    key = cv2.waitKey(1)
    if key == 81 or key == 113:
        break
    else:
        continue
video.release()