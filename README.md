Ahmed Al-Qarm       202073058
Osama Al-Dhawrani   202174053

---

# Face Recognition Project

## Description

This project implements a real-time face recognition system using OpenCV and the `face_recognition` library. It captures video from the webcam, detects faces, and recognizes them based on pre-stored images. When a recognized face is found, the name is displayed on the video feed.

## Requirements

- Python 3.x
- OpenCV
- NumPy
- face_recognition
- PIL (Pillow)

You can install the necessary libraries using pip:

```bash
pip install opencv-python numpy face_recognition Pillow
```

## Project Structure

```
/face_recognition_project
│
├── persons/                # Directory containing images of known persons
│   ├── person1.jpg
│   ├── person2.jpg
│   └── ...
│
└── face_recognition.py     # Main script
```

## Usage

1. Place the images of known persons in the `persons` directory. Ensure the names of the images are easily recognizable.
2. Run the main script:

```bash
python face_recognition.py
```

3. Press `Esc` to exit the application.

## Code Explanation

```python
import cv2
import numpy as np
import face_recognition
import os
```
- **Imports**: The necessary libraries are imported. `cv2` for image processing, `numpy` for numerical operations, `face_recognition` for facial recognition tasks, and `os` for file path operations.

```python
path = 'persons'
images = []
classNames = []
personsList = os.listdir(path)
```
- **Setup Paths**: The variable `path` specifies the directory containing the images of known persons. `images` and `classNames` lists are initialized to store the images and corresponding names of the persons.

```python
for cl in personsList:
    curPersonn = cv2.imread(f'{path}/{cl}')
    images.append(curPersonn)
    classNames.append(os.path.splitext(cl)[0])
print(classNames)
```
- **Load Images**: This loop reads each image in the `persons` directory, appending it to the `images` list. The name of each image (without the extension) is added to `classNames`. Finally, it prints the list of names for verification.

```python
def findEncodeings(images):
    encodeList = []
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)[0]
        encodeList.append(encode)
    return encodeList
```
- **Encoding Function**: The `findEncodeings` function converts images from BGR to RGB format and computes the face encodings using the `face_encodings` method. It returns a list of encodings, which are numerical representations of the faces.

```python
encodeListKnown = findEncodeings(images)
print('Encoding Complete.')
```
- **Generate Encodings**: The function is called to generate encodings for all known images, and a completion message is printed.

```python
cap = cv2.VideoCapture(0)
```
- **Open Webcam**: The webcam is opened for video capture using `cv2.VideoCapture(0)`, where `0` is the default camera.

```python
while True:
    success, img = cap.read()
```
- **Capture Video Frames**: A loop begins that continuously captures frames from the webcam.

```python
    if not success:
        print("Failed to capture image")
        break
```
- **Check Capture Success**: If the frame capture fails, an error message is printed, and the loop breaks.

```python
    imgS = cv2.resize(img, (0, 0), None, 0.25, 0.25)
    imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)
```
- **Resize and Convert**: The captured frame is resized to 25% of its original size for faster processing, and then it is converted to RGB color space.

```python
    faceCurentFrame = face_recognition.face_locations(imgS)
    encodeCurentFrame = face_recognition.face_encodings(imgS, faceCurentFrame)
```
- **Face Detection and Encoding**: The locations of faces in the frame are detected, and their encodings are computed.

```python
    for encodeface, faceLoc in zip(encodeCurentFrame, faceCurentFrame):
```
- **Match Detected Faces**: The loop iterates through each detected face's encoding and location.

```python
        matches = face_recognition.compare_faces(encodeListKnown, encodeface)
        faceDis = face_recognition.face_distance(encodeListKnown, encodeface)
        matchIndex = np.argmin(faceDis)
```
- **Compare Faces**: The detected face is compared to known encodings. The distances to each known encoding are calculated, and the index of the closest match is found.

```python
        if matches[matchIndex]:
            name = classNames[matchIndex].upper()
            print(name)
```
- **Recognize Face**: If a match is found, the corresponding name is retrieved and printed.

```python
            y1, x2, y2, x1 = faceLoc
            y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4
```
- **Get Face Location**: The coordinates of the bounding box around the detected face are extracted and scaled back to the original frame size.

```python
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.rectangle(img, (x1, y2 - 35), (x2, y2), (0, 0, 255), cv2.FILLED)
            cv2.putText(img, name, (x1 + 6, y2 - 6), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)
```
- **Draw Bounding Box and Name**: A rectangle is drawn around the face, and the recognized name is displayed above the rectangle.

```python
    cv2.imshow('Face Recognition', img)
```
- **Display Video Feed**: The modified frame (with rectangles and names) is shown in a window.

```python
    if cv2.waitKey(1) & 0xFF == 27:  
        break
```
- **Exit Condition**: The loop will exit if the `Esc` key is pressed.

```python
cap.release()
cv2.destroyAllWindows()
```
- **Cleanup**: Finally, the camera is released, and all OpenCV windows are closed.

## Conclusion

This project demonstrates how to create a simple face recognition system using Python, OpenCV, and the `face_recognition` library. You can enhance this project by adding features like saving recognized faces, improving accuracy with better models, or integrating it with a database.

---

Feel free to modify any part of this `README.md` to better fit your project's needs! If you have any further questions, let me know!