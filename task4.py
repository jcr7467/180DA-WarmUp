import numpy as np
import cv2

cap = cv2.VideoCapture(0)

while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Our operations on the frame come here
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)



    #https://docs.opencv.org/4.x/df/d9d/tutorial_py_colorspaces.html
    lower_red = np.array([161, 100, 80])
    upper_red = np.array([179, 255, 255])
    mask = cv2.inRange(hsv, lower_red, upper_red)
    template = cv2.imread('templates/coffee_mask_template.png', cv2.IMREAD_GRAYSCALE)

    template_w, template_h = template.shape[::-1]

    template_match_result = cv2.matchTemplate(mask, template, cv2.TM_CCOEFF_NORMED)

    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(template_match_result)

    top_left = max_loc
    bottom_right = (top_left[0] + template_w, top_left[1] + template_h)

    frame = cv2.rectangle(frame, top_left, bottom_right, (0,255,0), 2)

    # gray_mask = cv2.cvtColor(mask, cv2.COLOR_HSV2GRAY)

    result = cv2.bitwise_and(frame, frame, mask=mask)





    # ret, thresh1 = cv2.threshold(gray, 127, 255, )

    # Display the resulting frame
    cv2.imshow('frame', frame)
    cv2.imshow('Bitwise mask', mask)
    cv2.imshow('HSV result', result)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()