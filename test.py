import cv2
from cvzone.HandTrackingModule import HandDetector
from cvzone.ClassificationModule import Classifier
import numpy as np
import math
import pyttsx3
import time
from twilio.rest import Client
import face_recognition
import threading
from user_log import log_user_interaction

# Initialize video capture with lower resolution
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# Initialize hand detector and classifier with the updated model and labels
classifier = Classifier(r"C:\Users\aryan\Downloads\SignDetectionCVProject\Model\keras_model.h5", 
                        r"C:\Users\aryan\Downloads\SignDetectionCVProject\Model\labels.txt")
detector = HandDetector(maxHands=1)

# Initialize text-to-speech engine
engine = pyttsx3.init()

# Labels for classification
labels = ["Hello", "Cancel", "Help-Cleaner", "Help-Technician","Yes"]

# Confidence threshold
confidence_threshold = 0.99

# Twilio credentials and client initialization
account_sid = 'xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx'
auth_token = 'xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx'
twilio_client = Client(account_sid, auth_token)
from_phone = '+1 567 xx-xxxx'
to_phone = '+91 9623xx xxxx'

# Path to the authorized people's images and names
authorized_people = {
    "Gayatri": {
        "image_path": r"C:\Users\aryan\Downloads\SignDetectionCVProject\Gayatri.jpg",
        "face_encoding": None  
    },
    "saurabh": {
        "image_path": r"C:\Users\aryan\Downloads\SignDetectionCVProject\saurabh.jpg",
        "face_encoding": None  },

    "Asit": {
        "image_path": r"C:\Users\aryan\Downloads\SignDetectionCVProject\asit.jpg",
        "face_encoding": None 

    }
}

# Function to load face encodings for authorized people
def load_authorized_encodings():
    for person, info in authorized_people.items():
        image_path = info["image_path"]
        authorized_img = face_recognition.load_image_file(image_path)
        face_encoding = face_recognition.face_encodings(authorized_img)[0]
        authorized_people[person]["face_encoding"] = face_encoding

# Function to send SMS notification
def send_sms_notification(authenticated_person):
    message = twilio_client.messages.create(
        body=f"{authenticated_person} requires cleaning assistance at the 5G Lab.",
        from_=from_phone,
        to=to_phone
    )
    print(f"SMS sent with SID: {message.sid}")

# Function to make a call using Twilio
def make_call():
    call = twilio_client.calls.create(
        to=to_phone,
        from_=from_phone,
        url='http://demo.twilio.com/docs/voice.xml'
    )
    print(f"Call initiated with SID: {call.sid}")

# Variables for gesture recognition and timing
last_gesture = None
gesture_times = {label: 0 for label in labels}
cooldown_time = 5  # Cooldown period in seconds
consistency_threshold = 5  # Number of consecutive frames required for consistency
consistency_counter = {label: 0 for label in labels}  # Counter for gesture consistency
authenticated = False
authenticated_person = None

# Function to capture and authenticate
def capture_and_authenticate():
    global authenticated, authenticated_person
    while not authenticated:
        # Capture a frame from the webcam
        success, frame = cap.read()
        if not success:
            print("Error: Failed to capture frame from webcam.")
            continue

        # Detect faces in the frame
        face_locations = face_recognition.face_locations(frame)
        if len(face_locations) == 0:
            continue

        # Loop through each detected face
        for face_location in face_locations:
            top, right, bottom, left = face_location

            # Crop and resize the face region
            face_img = frame[top:bottom, left:right]
            face_img = cv2.resize(face_img, (0, 0), fx=0.25, fy=0.25)

            # Compute face encoding
            face_encoding = face_recognition.face_encodings(face_img)
            if len(face_encoding) == 0:
                continue

            face_encoding = face_encoding[0]

            # Compare with authorized people's encodings
            for person, info in authorized_people.items():
                authorized_encoding = info["face_encoding"]
                distance = face_recognition.face_distance([authorized_encoding], face_encoding)[0]
                print(f"Face distance from {person}: {distance}")

                # Define a threshold for authentication success
                if distance < 0.6:  # Adjust threshold as needed
                    print(f"Face authentication successful for {person}!")
                    authenticated = True
                    authenticated_person = person
                    engine.say(f"Welcome {person}. You may proceed with gestures.")
                    engine.runAndWait()
                    return

        if not authenticated:
            print("Unknown person.")
            engine.say("You don't have authorization.")
            engine.runAndWait()
            authenticated = False

# Start a separate thread for loading face encodings and authentication
load_authorized_encodings()
auth_thread = threading.Thread(target=capture_and_authenticate)
auth_thread.start()

# Main loop for gesture detection
webcam_active = True

while webcam_active:
    try:
        # Read frame from webcam
        success, img = cap.read()
        if not success:
            break

        imgOutput = img.copy()

        # Display instructions on the webcam feed
        instructions = "  Gesture Recognition Model  "
        cv2.putText(imgOutput, instructions, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)

        if not authenticated:
            cv2.putText(imgOutput, "Authenticating...", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2, cv2.LINE_AA)
            cv2.imshow('Webcam', imgOutput)
            key = cv2.waitKey(1)
            if key == ord("q"):  # Press 'q' to quit
                webcam_active = False
            continue

        # Verify the person performing gestures is the authenticated user
        face_locations = face_recognition.face_locations(img)
        face_encodings = face_recognition.face_encodings(img, face_locations)
        authenticated_face_found = False

        for (face_location, face_encoding) in zip(face_locations, face_encodings):
            face_distance = face_recognition.face_distance([authorized_people[authenticated_person]["face_encoding"]], face_encoding)[0]
            if face_distance < 0.6:  # Threshold for matching faces
                authenticated_face_found = True
                break

        if not authenticated_face_found:
            cv2.putText(imgOutput, "Unauthenticated user", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2, cv2.LINE_AA)
            cv2.imshow('Webcam', imgOutput)
            key = cv2.waitKey(1)
            if key == ord("q"):  # Press 'q' to quit
                webcam_active = False
            continue

        # Detect hands in the frame
        hands, img = detector.findHands(img)

        if len(hands) == 0:
            # No hands detected
            gesture = "No Gesture"
            cv2.putText(imgOutput, gesture, (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2)
            cv2.imshow('Webcam', imgOutput)
            key = cv2.waitKey(1)
            if key == ord("q"):  # Press 'q' to quit
                break
            continue

        for hand in hands:
            x, y, w, h = hand['bbox']

            # Draw bounding box around the hand
            cv2.rectangle(imgOutput, (x, y), (x + w, y + h), (255, 0, 255), 2)

            imgWhite = np.ones((300, 300, 3), np.uint8) * 255

            imgCrop = img[y - 20:y + h + 20, x - 20:x + w + 20]

            if imgCrop.shape[0] == 0 or imgCrop.shape[1] == 0:
                continue

            aspectRatio = h / w

            if aspectRatio > 1:
                k = 300 / h
                wCal = math.ceil(k * w)
                imgResize = cv2.resize(imgCrop, (wCal, 300))
                wGap = math.ceil((300 - wCal) / 2)
                imgWhite[:, wGap: wCal + wGap] = imgResize
            else:
                k = 300 / w
                hCal = math.ceil(k * h)
                imgResize = cv2.resize(imgCrop, (300, hCal))
                hGap = math.ceil((300 - hCal) / 2)
                imgWhite[hGap: hCal + hGap, :] = imgResize

            # Allow recognition of gestures after authentication
            prediction, index = classifier.getPrediction(imgWhite, draw=False)
            confidence = prediction[index]  # Get confidence score
            print(prediction, index, confidence)

            # Check for recognized gestures and trigger corresponding actions
            if index < len(labels) and confidence >= confidence_threshold:
                gesture = labels[index]

                # Update consistency counter for the gesture
                consistency_counter[gesture] += 1
                for other_gesture in labels:
                    if other_gesture != gesture:
                        consistency_counter[other_gesture] = 0

                # Perform actions based on recognized gestures
                current_time = time.time()

                if consistency_counter[gesture] >= consistency_threshold:  # Check if the gesture is consistent
                    if gesture != last_gesture:
                        if (current_time - gesture_times.get(gesture, 0)) > cooldown_time:
                            if gesture == "Hello":
                                print("Hello Gesture Detected!")
                                engine.say(f"{gesture} {authenticated_person}. How can I assist you today?")
                                log_user_interaction(authenticated_person, "Hello action")
                                engine.runAndWait()

                            elif gesture == "Cancel":
                                print("Cancel Gesture Detected!")
                                engine.say("Cancel!")
                                log_user_interaction(authenticated_person, "Cancel action")
                                engine.runAndWait()
                                break  # Exit the gesture recognition loop

                            elif gesture == "Yes":
                               print("Yes Gesture Detected!")
                               engine.say(f"Thank you for your response {authenticated_person}.")
                               log_user_interaction(authenticated_person, "Yes action")
                               engine.runAndWait()

                            elif gesture == "Help-Cleaner":
                                print("Help-Cleaner Gesture Detected!")
                                send_sms_notification(authenticated_person)
                                engine.say("Help for cleaning is requested.")
                                log_user_interaction(authenticated_person, "Help-Cleaner action")
                                engine.runAndWait()

                            elif gesture == "Help-Technician":
                                print("Help-Technician Gesture Detected!")
                                make_call()
                                engine.say("Help for technical support is on the way.")
                                log_user_interaction(authenticated_person, "Help-Technician action")
                                engine.runAndWait()

                        # Update the last gesture and time
                        last_gesture = gesture
                        gesture_times[gesture] = current_time

                # Add the recognized gesture name to the frame
                cv2.putText(imgOutput, gesture, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            else:
                 gesture = "Unknown Gesture"
                 consistency_counter = {label: 0 for label in labels}  # Reset all consistency counters
                 cv2.putText(imgOutput, gesture, (x, y - 10), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2)

        cv2.imshow('Webcam', imgOutput)
        key = cv2.waitKey(1)
        if key == ord("q"):  # Press 'q' to quit
            webcam_active = False

    except Exception as e:
        print(f"An error occurred: {e}")
        break

cap.release()
cv2.destroyAllWindows()
