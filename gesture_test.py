import cv2
import mediapipe as mp

# Load MediaPipe Tasks for Hand Landmarker
mp_tasks_vision = mp.tasks.vision
BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp_tasks_vision.HandLandmarker
HandLandmarkerOptions = mp_tasks_vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

# Initialize a global variable to hold the annotated frame
annotated_frame = None

# Callback for hand landmark results
def hand_landmark_result_callback(result, output_image, timestamp_ms):
    global annotated_frame
    # Convert output_image to OpenCV format
    annotated_frame = output_image.numpy_view()

    # Draw hand landmarks if detected
    if result.hand_landmarks:
        for hand_landmarks in result.hand_landmarks:
            mp.drawing_utils.draw_landmarks(
                annotated_frame, hand_landmarks, mp.solutions.hands.HAND_CONNECTIONS,
                mp.solutions.drawing_styles.get_default_hand_landmarks_style(),
                mp.solutions.drawing_styles.get_default_hand_connections_style()
            )

# Initialize Hand Landmarker with live-stream mode and callback
hand_landmarker_options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path='hand_landmarker.task'),  # Update with correct model path
    running_mode=VisionRunningMode.LIVE_STREAM,
    result_callback=hand_landmark_result_callback
)

# Start video capture and initialize the hand landmarker
with HandLandmarker.create_from_options(hand_landmarker_options) as hand_landmarker:
    cap = cv2.VideoCapture(0)
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Flip image for a mirror effect
        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Use the Hand Landmarker to detect landmarks
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        timestamp_ms = int(cap.get(cv2.CAP_PROP_POS_MSEC))
        hand_landmarker.detect_async(mp_image, timestamp_ms)

        # Display the annotated frame if landmarks were drawn, otherwise show the original frame
        if annotated_frame is not None:
            cv2.imshow("Hand Landmarks", annotated_frame)
        else:
            cv2.imshow("Hand Landmarks", frame)

        # Press 'ESC' to exit
        if cv2.waitKey(5) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()
