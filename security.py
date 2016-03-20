#!/usr/bin/env python

import numpy as np
import cv2

cv2.namedWindow("+1", cv2.WINDOW_NORMAL)

cap = cv2.VideoCapture(0)

while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Our operations on the frame come here
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    height, width, _ = rgb.shape

    # Display the resulting frame
    cv2.imshow('+1', cv2.resize(rgb, (1024, 768)))

    cv2.imwrite('security/data/frames/screenshot.jpg', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()