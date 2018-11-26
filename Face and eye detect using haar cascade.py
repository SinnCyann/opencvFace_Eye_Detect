import numpy as np
import cv2

face_cascade = cv2.CascadeClassifier('./haarcascade_frontalface_default.xml')

eye_cascade = cv2.CascadeClassifier('./haarcascade_eye.xml')


cap = cv2.VideoCapture(0)
found_iris_counter = 0
eye_counter = 0
if cap.isOpened():
    while True:
        check, frame = cap.read()
        if check:
            while 1:
                ret, img = cap.read()
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                faces = face_cascade.detectMultiScale(gray, 1.3, 5)

                for (x, y, w, h) in faces:
                    cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
                    roi_gray = gray[y:y + h, x:x + w]
                    roi_color = img[y:y + h, x:x + w]

                    eyes = eye_cascade.detectMultiScale(roi_gray)
                    for (ex, ey, ew, eh) in eyes:
                        # cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 10)

                        # here's your eye-roi, see, it's the very same pattern
                        # roi_color_eye = roi_color[ey:ey + eh, ex:ex + ew * 2]
                        roi_color_eye = roi_color[ey:ey + eh, ex:x + w]
                        # write image *before* drawing stuff on it
                        cv2.imwrite("eye_%d.png" % eye_counter, roi_color_eye)
                        eye_counter += 1
                        cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)

                    # success, photo = cap.read()
                    # finished, iris_pic = draw(photo)
                    # if iris_pic is not None:
                    #     timecode = time.strftime("%Y-%m-%d_%H;%M;%S", time.gmtime())
                    #     cv2.imwrite("iris_pic_"+str(found_iris_counter)+"_"+timecode+".jpg", iris_pic)
                    #     found_iris_counter += 1

                cv2.imshow('img', img)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

            cap.release()
            cv2.destroyAllWindows()




        else:
            print('Frame not available')
            print(cap.isOpened())



