import cv2
import mediapipe as mp

# Load MediaPipe Tasks for both Hand Landmarker and Gesture Recognizer
mp_tasks_vision = mp.tasks.vision
BaseOptions = mp.tasks.BaseOptions
GestureRecognizer = mp_tasks_vision.GestureRecognizer
GestureRecognizerOptions = mp_tasks_vision.GestureRecognizerOptions
GestureRecognizerResult = mp.tasks.vision.GestureRecognizerResult
HandLandmarker = mp_tasks_vision.HandLandmarker
HandLandmarkerOptions = mp_tasks_vision.HandLandmarkerOptions
VisionRunningMode = mp_tasks_vision.RunningMode

# Callback to print gesture recognition result
def print_result(result: GestureRecognizerResult, output_image: mp.Image, timestamp_ms: int):
    if result.gestures:
        gesture_label = result.gestures[0][0].category_name
        print(f"Detected Gesture: {gesture_label}")
    else:
        print("No gesture detected.")

# Initialize Hand Landmarker
hand_landmarker_options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path='hand_landmarker.task'),  # Update with correct model path
    running_mode=VisionRunningMode.LIVE_STREAM
)
hand_landmarker = HandLandmarker.create_from_options(hand_landmarker_options)

# Initialize Gesture Recognizer with live stream mode
gesture_recognizer_options = GestureRecognizerOptions(
    base_options=BaseOptions(model_asset_path='gesture_recognizer.task'),  # Update with correct model path
    running_mode=VisionRunningMode.LIVE_STREAM,
    result_callback=print_result
)

with hand_landmarker, GestureRecognizer.create_from_options(gesture_recognizer_options) as recognizer:
    cap = cv2.VideoCapture(0)
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Flip image for a mirror effect
        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Detect hand landmarks
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        timestamp_ms = int(cap.get(cv2.CAP_PROP_POS_MSEC))
        landmarks_result = hand_landmarker.detect_async(mp_image, timestamp_ms)

        # Draw hand landmarks if detected
        if landmarks_result.hand_landmarks:
            for hand_landmarks in landmarks_result.hand_landmarks:
                mp.drawing_utils.draw_landmarks(
                    frame, hand_landmarks, mp.solutions.hands.HAND_CONNECTIONS,
                    mp.solutions.drawing_styles.get_default_hand_landmarks_style(),
                    mp.solutions.drawing_styles.get_default_hand_connections_style()
                )

            # Recognize gestures based on the landmarks detected
            recognizer.recognize_async(mp_image, timestamp_ms)

        # Display the frame
        cv2.imshow("Gesture Recognition with Landmarks", frame)

        # Press 'ESC' to exit
        if cv2.waitKey(5) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()
