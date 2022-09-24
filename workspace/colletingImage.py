import os
import uuid
import cv2

positive_image = os.makedirs(os.path.abspath(os.path.join('data', 'positive')))
negative_image = os.makedirs(os.path.abspath(os.path.join('data', 'negative')))
# for negative image can be downloaded on the link in readme, and then extract to negative image path above
anchor_image = os.makedirs(os.path.abspath(os.path.join('data', 'anchor')))

cap = cv2.VideoCapture(1)
while cap.isOpened():
    ret, frame = cap.read()
    frame = frame[120:120+250, 200:200+250, :]
    
    cv2.imshow('image collection', frame)
    
    # Collect Anchors    
    if cv2.waitKey(1) & 0XFF == ord('a'):
        imgname = os.path.join(os.path.abspath(os.path.join('data', 'anchor')), '{}.jpg'.format(uuid.uuid1()))
        cv2.imwrite(imgname, frame)
    
    # Collect Positives
    if cv2.waitKey(1) & 0XFF == ord('p'):
        imgname = os.path.join(os.path.abspath(os.path.join('data', 'positive')), '{}.jpg'.format(uuid.uuid1()))
        cv2.imwrite(imgname, frame)
    if cv2.waitKey(1) & 0XFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()