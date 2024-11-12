# STEP 1: Import the necessary modules.
import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# Helper function to draw landmarks (you may need to define or import this function)
def draw_landmarks_on_image(image, detection_result):
    # Add code here to draw landmarks from `detection_result` onto `image`
    # For example, you might use cv2.circle to mark each landmark point.
    for hand in detection_result.multi_hand_landmarks:
        for landmark in hand.landmark:
            x = int(landmark.x * image.shape[1])
            y = int(landmark.y * image.shape[0])
            cv2.circle(image, (x, y), 5, (0, 255, 0), -1)  # Draw landmarks
    return image

# STEP 2: Create a HandLandmarker object.
base_options = python.BaseOptions(model_asset_path='hand_landmarker.task')
options = vision.HandLandmarkerOptions(base_options=base_options, num_hands=2)
detector = vision.HandLandmarker.create_from_options(options)

# STEP 3: Start capturing video from the webcam.
cap = cv2.VideoCapture(0)  # Change the number if using another camera

# Loop for processing each frame
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Convert the frame to RGB as MediaPipe expects RGB images.
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Convert the frame to a MediaPipe Image object.
    image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

    # STEP 4: Detect hand landmarks in the current frame.
    detection_result = detector.detect(image)

    # STEP 5: Draw landmarks on the frame.
    annotated_frame = draw_landmarks_on_image(frame, detection_result)

    # STEP 6: Display the annotated frame.
    cv2.imshow('Hand Gesture Control', annotated_frame)

    # Exit the loop if the user presses the 'q' key.
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

# Release the webcam and close OpenCV windows.
cap.release()
cv2.destroyAllWindows()
