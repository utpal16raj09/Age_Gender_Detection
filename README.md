
# Age and Gender Detection System

This project implements a real-time age and gender detection system using OpenCV and pre-trained deep learning models. It captures video from a webcam, detects faces, and predicts the age and gender of each detected face, while highlighting the detected faces and predicting their attributes.

## Project Structure

```
Face_Gender_Detect/
│
├── ML_Models/                                # Directory containing pre-trained models
│   ├── age_deploy.prototxt                   # Caffe model architecture for age prediction
│   ├── age_net.caffemodel                    # Caffe model weights for age prediction
│   ├── gender_deploy.prototxt                # Caffe model architecture for gender prediction
│   ├── gender_net.caffemodel                 # Caffe model weights for gender prediction
│   ├── opencv_face_detector_uint8.pb         # TensorFlow model for face detection
│   └── opencv_face_detector.pbtxt            # Configuration file for TensorFlow face detection model
│
├── Face_gender_detect.py                     # Main Python script for face, gender, and age detection
└── README.md                                 # Project documentation
```

## Prerequisites

To run this project, you need Python and the following libraries:

- `opencv-python`
- `numpy`

Install these dependencies using pip:

```bash
pip install opencv-python numpy
```

## Setup and Usage

1. **Clone the Repository**

   Clone this repository to your local machine:

   ```bash
   git clone https://github.com/utpal16raj09/Age_Gender_Detection.git
   ```

2. **Navigate to the Project Directory**

   Change to the project directory:

   ```bash
   cd Face_Gender_Detect
   ```

3. **Run the Detection Script**

   Execute the Python script to start the real-time detection:

   ```bash
   python Face_gender_detect.py
   ```

   The script will open a window displaying the webcam feed with detected faces annotated with predicted age and gender.

## How It Works

### 1. Face Detection

- **Model Used:** `opencv_face_detector_uint8.pb` and `opencv_face_detector.pbtxt`
- **Framework:** TensorFlow
- **Purpose:** Detect faces in the video feed.
- **Process:** The TensorFlow model identifies faces in each frame. The `opencv_face_detector.pbtxt` file configures the model for face detection.

### 2. Age Prediction

- **Model Used:** `age_net.caffemodel` and `age_deploy.prototxt`
- **Framework:** Caffe
- **Purpose:** Predict the age range of each detected face.
- **Process:** Each detected face is processed by the Caffe model to estimate its age from predefined ranges.

### 3. Gender Prediction

- **Model Used:** `gender_net.caffemodel` and `gender_deploy.prototxt`
- **Framework:** Caffe
- **Purpose:** Predict the gender of each detected face.
- **Process:** The Caffe model predicts the gender based on the facial features extracted from the detected face.

### 4. Facial Landmarks

- **Feature:** 68 Facial Landmarks
- **Purpose:** Enhance facial feature detection and alignment.
- **Process:** The script uses facial landmarks to better position and align the detected faces. This can help improve the accuracy of age and gender predictions.

## Notes

- Ensure your webcam is properly connected and accessible.
- The pre-trained models may not be perfect for all scenarios. Adjustments may be necessary based on lighting and other conditions.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

