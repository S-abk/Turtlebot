import mediapipe as mp
import cv2
import math
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe.framework.formats import landmark_pb2

# Import drawing utilities and styles
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# Define the base options for MediaPipe tasks
BaseOptions = python.BaseOptions
GestureRecognizer = vision.GestureRecognizer
GestureRecognizerOptions = vision.GestureRecognizerOptions
GestureRecognizerResult = vision.GestureRecognizerResult
VisionRunningMode = vision.RunningMode

# Callback function to process and display results
def print_result(result: GestureRecognizerResult, output_image: mp.Image, timestamp_ms: int):
    # Display a placeholder message to confirm that the callback is triggered
    print("Gesture recognition result:", result)

# Configure GestureRecognizer with live stream mode
options = GestureRecognizerOptions(
    base_options=BaseOptions(model_asset_path='gesture_recognizer.task'),  # Update with your model path
    running_mode=VisionRunningMode.LIVE_STREAM,
    result_callback=print_result
)

# Start the gesture recognizer with a live camera stream
with GestureRecognizer.create_from_options(options) as recognizer:
    # Open the default camera
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Error: Could not open camera.")
        exit()

    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                print("Failed to capture frame.")
                break

            # Convert frame to RGB as MediaPipe requires RGB format
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Wrap the frame for MediaPipe input and use timestamp
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
            timestamp_ms = int(cap.get(cv2.CAP_PROP_POS_MSEC))

            # Process the frame with gesture recognizer
            recognizer.recognize_async(mp_image, timestamp_ms)

            # Show the original frame
            cv2.imshow('Live Camera Feed', frame)

            # Exit when ESC is pressed
            if cv2.waitKey(5) & 0xFF == 27:
                break
    finally:
        # Release the capture and close windows
        cap.release()
        cv2.destroyAllWindows()
