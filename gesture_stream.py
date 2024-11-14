import mediapipe as mp
import cv2

# MediaPipe classes and constants
BaseOptions = mp.tasks.BaseOptions
GestureRecognizer = mp.tasks.vision.GestureRecognizer
GestureRecognizerOptions = mp.tasks.vision.GestureRecognizerOptions
GestureRecognizerResult = mp.tasks.vision.GestureRecognizerResult
VisionRunningMode = mp.tasks.vision.RunningMode

# Callback function to handle gesture recognition results
def print_result(result: GestureRecognizerResult, output_image: mp.Image, timestamp_ms: int):
    print('Gesture recognition result:', result)

# Gesture Recognizer options setup for live stream
options = GestureRecognizerOptions(
    base_options=BaseOptions(model_asset_path='gesture_recognizer.task'),  # model path
    running_mode=VisionRunningMode.LIVE_STREAM,
    result_callback=print_result
)

# Start gesture recognizer with the given options
with GestureRecognizer.create_from_options(options) as recognizer:
    # Open a video capture stream from the default camera (or specify the camera index)
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

            # Convert the frame to RGB (MediaPipe requires RGB images)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Wrap the frame in a MediaPipe Image and provide a timestamp
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
            timestamp_ms = int(cap.get(cv2.CAP_PROP_POS_MSEC))

            # Process the frame with the gesture recognizer
            recognizer.recognize_async(mp_image, timestamp_ms)

            # Display the frame (optional)
            cv2.imshow('Gesture Recognition', frame)
            if cv2.waitKey(5) & 0xFF == 27:  # Exit on pressing ESC
                break
    finally:
        # Release the camera and close OpenCV windows
        cap.release()
        cv2.destroyAllWindows()
