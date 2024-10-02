import cv2
import numpy as np
import face_recognition
import os
from PIL import Image, ImageDraw, ImageFont

font_path = 'C:/Users/MC/font.TTF'  # استبدل هذا بالمسار الصحيح للخط الذي قمت بتحميله
font_size = 32
font = ImageFont.truetype(font_path, font_size)

path = 'persons'
images = []
classNames = []
personsList = os.listdir(path)

for cl in personsList:
    curPersonn = cv2.imread(f'{path}/{cl}')
    images.append(curPersonn)
    classNames.append(os.path.splitext(cl)[0])
print(classNames)

def findEncodeings(images):
    encodeList = []
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)[0]
        encodeList.append(encode)
    return encodeList

encodeListKnown = findEncodeings(images)
print('Encoding Complete.')

cap = cv2.VideoCapture(0)

while True:
    success, img = cap.read()

    if not success:
        print("Failed to capture image")
        break

    imgS = cv2.resize(img, (0, 0), None, 0.25, 0.25)
    imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)

    faceCurentFrame = face_recognition.face_locations(imgS)
    encodeCurentFrame = face_recognition.face_encodings(imgS, faceCurentFrame)

    for encodeface, faceLoc in zip(encodeCurentFrame, faceCurentFrame):
        matches = face_recognition.compare_faces(encodeListKnown, encodeface)
        faceDis = face_recognition.face_distance(encodeListKnown, encodeface)
        matchIndex = np.argmin(faceDis)

        if matches[matchIndex]:
            name = classNames[matchIndex].upper()
            print(name)
            y1, x2, y2, x1 = faceLoc
            y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.rectangle(img, (x1, y2 - 35), (x2, y2), (0, 0, 255), cv2.FILLED)
            cv2.putText(img, name, (x1 + 6, y2 - 6), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)

        mage_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(image_pil)
        arabic_text = "مرحبا بكم في OpenCV"
        draw.text((10, 10), arabic_text, font=font, fill=(255, 255, 255)) 
        img = cv2.cvtColor(np.array(image_pil), cv2.COLOR_RGB2BGR)

        cv2.imshow('Face Recognition', img)

    if cv2.waitKey(1) & 0xFF == 27:  
     break

cap.release()
cv2.destroyAllWindows()