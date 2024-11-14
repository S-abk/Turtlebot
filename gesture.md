
**1. Install Required Libraries**

Ensure you have the necessary packages installed:

```bash
pip install opencv-python mediapipe
```

**2. Import Necessary Modules**

Import the required modules from OpenCV and MediaPipe:

```python
import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
```

**3. Define the Result Callback Function**

This function processes the results asynchronously:

```python
def print_result(result, output_image, timestamp_ms):
    if result.gestures:
        for gesture in result.gestures[0]:
            print(f"Gesture recognized: {gesture.category_name} with score {gesture.score:.2f}")
```

**4. Set Up the Gesture Recognizer**

Configure the Gesture Recognizer with the live stream mode and the result callback:

```python
# Path to the gesture recognizer model
model_path = 'gesture_recognizer.task'

# Base options for the model
base_options = python.BaseOptions(model_asset_path=model_path)

# Configure the gesture recognizer options
options = vision.GestureRecognizerOptions(
    base_options=base_options,
    running_mode=vision.RunningMode.LIVE_STREAM,
    result_callback=print_result
)

# Create the gesture recognizer
with vision.GestureRecognizer.create_from_options(options) as recognizer:
    # Initialize the webcam
    cap = cv2.VideoCapture(0)

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            print("Ignoring empty camera frame.")
            continue

        # Convert the frame to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Create a MediaPipe Image object
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

        # Get the current timestamp in milliseconds
        frame_timestamp_ms = int(cap.get(cv2.CAP_PROP_POS_MSEC))

        # Send the image to the gesture recognizer
        recognizer.recognize_async(mp_image, frame_timestamp_ms)

        # Display the frame
        cv2.imshow('Gesture Recognition', frame)

        if cv2.waitKey(5) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()
```


- **Model Path**: Ensure the `gesture_recognizer.task` model file is in your working directory or provide the correct path.
- **Result Callback**: The `print_result` function is called asynchronously with the recognition results.
- **Running Mode**: The `LIVE_STREAM` mode processes input data from a live stream, such as a webcam.
- **Frame Processing**: Each frame is captured, converted to RGB, and processed by the recognizer. The results are handled by the callback function.

For more detailed information, refer to the [Gesture Recognition Guide for Python](https://ai.google.dev/edge/mediapipe/solutions/vision/gesture_recognizer/python#live-stream). 


[media pipe repo](https://github.com/google-ai-edge/mediapipe-samples/blob/main/examples/gesture_recognizer/python/gesture_recognizer.ipynb)
