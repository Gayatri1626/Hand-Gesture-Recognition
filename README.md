# Hand-Gesture-Recognition

### README

# Gesture Recognition and Face Authentication System

This project involves a gesture recognition system combined with face authentication. It leverages OpenCV for computer vision tasks, CVZone for hand tracking, and the Twilio API for sending notifications.

## Table of Contents

- [Installation](#installation)
- [Setup](#setup)
- [Usage](#usage)
- [Dependencies](#dependencies)
- [Features](#features)
- [Acknowledgements](#acknowledgements)

## Installation

1. **Install the required Python packages:**
    ```sh
    pip install -r requirements.txt
    ```

2. **Download the model and labels:**
    Ensure that the model (`keras_model.h5`) and the labels (`labels.txt`) files are placed in the correct directory as specified in the code.

## Setup

1. **Twilio Setup:**
    - Sign up for a Twilio account and get your `account_sid` and `auth_token`.
    - Replace the placeholders in the script with your Twilio credentials:
        ```python
        account_sid = 'your_account_sid'
        auth_token = 'your_auth_token'
        ```

2. **Authorized People Setup:**
    - Add images of authorized people in the specified directory.
    - Update the `authorized_people` dictionary with the paths to these images and their names.

3. **Voice Engine Setup:**
    - Ensure that `pyttsx3` is correctly installed and configured on your machine for text-to-speech functionality.

## Usage

1. **Run the Script:**
    ```sh
    python gesture_recognition.py
    ```

2. **Face Authentication:**
    - The system will start with face authentication. Ensure that authorized persons' images are correctly loaded.
    - If an unauthorized person attempts to use the system, it will notify them and deny access.

3. **Gesture Recognition:**
    - Once authenticated, the system will start detecting hand gestures.
    - It supports gestures like "Hello", "Cancel", "Help-Cleaner", "Help-Technician", and "Yes".

4. **Interacting with the System:**
    - Perform gestures within the camera's field of view.
    - The system will perform corresponding actions based on the detected gesture (e.g., send an SMS, make a call, or provide a voice response).

## Dependencies

- **OpenCV**: For video capturing and image processing.
- **CVZone**: For hand tracking.
- **NumPy**: For numerical operations.
- **math**: For mathematical computations.
- **pyttsx3**: For text-to-speech conversion.
- **time**: For handling timing operations.
- **Twilio**: For sending SMS and making calls.
- **face_recognition**: For face detection and recognition.
- **threading**: For handling concurrent operations.

## Features

- **Face Authentication**: Only allows authorized persons to interact with the system.
- **Gesture Recognition**: Detects predefined hand gestures and performs actions.
- **SMS and Call Notifications**: Uses Twilio API to send notifications based on gestures.
- **Voice Feedback**: Provides voice responses using a text-to-speech engine.

## Acknowledgements

- **OpenCV**: For providing comprehensive tools for computer vision.
- **CVZone**: For simplifying hand tracking tasks.
- **Twilio**: For enabling seamless communication via SMS and calls.
- **Pyttsx3**: For converting text to speech in a simple manner.
- **Face Recognition**: For efficient face detection and recognition capabilities.

---

This project aims to demonstrate the integration of gesture recognition, face authentication, and communication APIs to create a responsive and secure interaction system. For any questions or further enhancements, feel free to contribute or contact the maintainer.

---

This README provides an overview and instructions for setting up and running the gesture recognition and face authentication system. For detailed implementation, refer to the source code and comments.
