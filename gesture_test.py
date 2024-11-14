import mediapipe as mp
import cv2

# Import necessary MediaPipe classes
BaseOptions = mp.tasks.BaseOptions
GestureRecognizer = mp.tasks.vision.GestureRecognizer
GestureRecognizerOptions = mp.tasks.vision.GestureRecognizerOptions
GestureRecognizerResult = mp.tasks.vision.GestureRecognizerResult
VisionRunningMode = mp.tasks.vision.RunningMode
DrawingUtils = mp.solutions.drawing_utils
DrawingStyles = mp.solutions.drawing_styles

# Callback function to display results with hand landmarks
def print_result(result: GestureRecognizerResult, output_image: mp.Image, timestamp_ms: int):
    # Convert MediaPipe image to OpenCV format
    output_frame = output_image.numpy_view()
    
    # Draw hand landmarks on the frame if they exist
    if result.hand_landmarks:
        for hand_landmarks in result.hand_landmarks:
            DrawingUtils.draw_landmarks(
                output_frame,
                hand_landmarks,
                mp.solutions.hands.HAND_CONNECTIONS,
                DrawingStyles.get_default_hand_landmarks_style(),
                DrawingStyles.get_default_hand_connections_style()
            )

    # Display the frame with landmarks
    cv2.imshow('Gesture Recognition with Landmarks', output_frame)

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

            # Exit when ESC is pressed
            if cv2.waitKey(5) & 0xFF == 27:
                break
    finally:
        # Release the capture and close windows
        cap.release()
        cv2.destroyAllWindows()
