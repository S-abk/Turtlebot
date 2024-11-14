import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# Initialize OpenCV and MediaPipe Hands
cap = cv2.VideoCapture(0)
mp_hands = mp.solutions.hands
hands = mp_hands.Hands()
mp_drawing = mp.solutions.drawing_utils

# Set up Gesture Recognizer
BaseOptions = python.BaseOptions
GestureRecognizer = vision.GestureRecognizer
GestureRecognizerOptions = vision.GestureRecognizerOptions
GestureRecognizerResult = vision.GestureRecognizerResult
VisionRunningMode = vision.RunningMode

# Callback to display gesture recognition result
def print_gesture_result(result: GestureRecognizerResult, output_image: mp.Image, timestamp_ms: int):
    if result.gestures:
        # Print the recognized gesture
        gesture_label = result.gestures[0][0].category_name
        print(f"Detected Gesture: {gesture_label}")
    else:
        print("No gesture detected.")

gesture_options = GestureRecognizerOptions(
    base_options=BaseOptions(model_asset_path='gesture_recognizer.task'),  # Update with correct model path
    running_mode=VisionRunningMode.LIVE_STREAM,
    result_callback=print_gesture_result
)

with GestureRecognizer.create_from_options(gesture_options) as recognizer:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Flip and process the frame for hand landmarks
        frame = cv2.flip(frame, 1)
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(image)

        # Draw hand landmarks and prepare gesture recognition input
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Convert image to MediaPipe format and recognize gestures
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image)
            timestamp_ms = int(cap.get(cv2.CAP_PROP_POS_MSEC))
            recognizer.recognize_async(mp_image, timestamp_ms)

        # Display the frame
        cv2.imshow('Hand Gesture with Landmark and Recognition', frame)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
