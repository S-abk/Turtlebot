import cv2
import mediapipe as mp
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# Initialize MediaPipe and ROS 2 components
mp_hands = mp.solutions.hands
BaseOptions = python.BaseOptions
GestureRecognizer = vision.GestureRecognizer
GestureRecognizerOptions = vision.GestureRecognizerOptions
GestureRecognizerResult = vision.GestureRecognizerResult
VisionRunningMode = vision.RunningMode

# Define the GestureControlNode class for ROS 2
class GestureControlNode(Node):
    def __init__(self):
        super().__init__('gesture_control_node')
        self.publisher_ = self.create_publisher(Twist, '/cmd_vel', 10)
        self.velocity_msg = Twist()

    def stop(self):
        self.velocity_msg.linear.x = 0.0
        self.velocity_msg.angular.z = 0.0
        self.publisher_.publish(self.velocity_msg)

    def move_forward(self):
        self.velocity_msg.linear.x = 0.2
        self.velocity_msg.angular.z = 0.0
        self.publisher_.publish(self.velocity_msg)

    def turn_left(self):
        self.velocity_msg.linear.x = 0.0
        self.velocity_msg.angular.z = 0.5
        self.publisher_.publish(self.velocity_msg)

    def turn_right(self):
        self.velocity_msg.linear.x = 0.0
        self.velocity_msg.angular.z = -0.5
        self.publisher_.publish(self.velocity_msg)

# Callback function for gesture recognition results
def print_gesture_result(result: GestureRecognizerResult, output_image: mp.Image, timestamp_ms: int):
    if result.gestures:
        gesture_label = result.gestures[0][0].category_name
        print(f"Detected Gesture: {gesture_label}")

        # Map gestures to robot commands
        if gesture_label == "open_palm" or gesture_label == "none":
            gesture_control_node.stop()
        elif gesture_label == "closed_fist":
            gesture_control_node.move_forward()
        elif gesture_label == "pointing_up":
            gesture_control_node.turn_left()
        elif gesture_label == "victory":
            gesture_control_node.turn_right()

# Initialize the Gesture Recognizer with live-stream mode
gesture_options = GestureRecognizerOptions(
    base_options=BaseOptions(model_asset_path='gesture_recognizer.task'),  # Update with the correct path
    running_mode=VisionRunningMode.LIVE_STREAM,
    result_callback=print_gesture_result
)

# Set up the video capture and ROS 2 node
rclpy.init()
gesture_control_node = GestureControlNode()

with GestureRecognizer.create_from_options(gesture_options) as recognizer:
    cap = cv2.VideoCapture(0)
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Flip and process the frame
        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Use the gesture recognizer to detect gestures
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        timestamp_ms = int(cap.get(cv2.CAP_PROP_POS_MSEC))
        recognizer.recognize_async(mp_image, timestamp_ms)

        # Display the frame
        cv2.imshow('Gesture Control', frame)

        # Exit on 'q' key press
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
rclpy.shutdown()
