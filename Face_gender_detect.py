import cv2

model_folder = "ML_Models/"

age_model = model_folder + 'age_net.caffemodel'
age_proto = model_folder + 'age_deploy.prototxt'
gender_model = model_folder + 'gender_net.caffemodel'
gender_proto = model_folder + 'gender_deploy.prototxt'

face_model = model_folder + 'opencv_face_detector_uint8.pb'
face_proto = model_folder + 'opencv_face_detector.pbtxt'

# Load the pre-trained models
age_net = cv2.dnn.readNetFromCaffe(age_proto, age_model)
gender_net = cv2.dnn.readNetFromCaffe(gender_proto, gender_model)
face_net = cv2.dnn.readNetFromTensorflow(face_model, face_proto)

MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)
age_list = ['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']
gender_list = ['Male', 'Female']

# Initialize the video capture for the webcam
video_capture = cv2.VideoCapture(0)

def detect_face_gender_age(frame):
    height, width = frame.shape[:2]

    blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), [104, 117, 123], True, False)
    face_net.setInput(blob)
    detections = face_net.forward()

    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]

        if confidence > 0.7:
            box = detections[0, 0, i, 3:7] * [width, height, width, height]
            (x, y, x1, y1) = box.astype(int)

            cv2.rectangle(frame, (x, y), (x1, y1), (0, 255, 0), 2)

            face_roi = frame[y:y1, x:x1]

            blob = cv2.dnn.blobFromImage(face_roi, 1.0, (227, 227), MODEL_MEAN_VALUES, swapRB=False)

            gender_net.setInput(blob)
            gender_preds = gender_net.forward()
            gender = gender_list[gender_preds[0].argmax()]

            age_net.setInput(blob)
            age_preds = age_net.forward()
            age = age_list[age_preds[0].argmax()]

            label = f"{gender}, {age}"
            cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

    return frame


# Main loop to process each frame
while True:
    # Capture each frame from the webcam
    ret, frame = video_capture.read()

    if not ret:
        break

    # Call the detection function
    output_frame = detect_face_gender_age(frame)

    # Display the frame with detection results
    cv2.imshow('Face, Gender, and Age Detection', output_frame)

    # Break the loop on pressing 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture object and close all windows
video_capture.release()
cv2.destroyAllWindows()
