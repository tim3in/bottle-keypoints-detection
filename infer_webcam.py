import cv2
import json
import math
from inference_sdk import InferenceHTTPClient

# Initialize the client
CLIENT = InferenceHTTPClient(
    api_url="https://detect.roboflow.com",
    api_key="YOUR_API_KEY"
)

# Function to calculate the angle
def calculate_angle(x1, y1, x2, y2):
    # Calculate the differences in coordinates
    delta_x = x1 - x2
    delta_y = y1 - y2

    # Calculate the angle using arctan2 and convert it to degrees
    angle_rad = math.atan2(delta_y, delta_x)
    angle_deg = math.degrees(angle_rad)

    # Ensure the angle is between 0 and 360 degrees
    mapped_angle = angle_deg % 360
    if mapped_angle < 0:
        mapped_angle += 360  # Ensure angle is positive

    return mapped_angle

# Open the webcam
cap = cv2.VideoCapture(0)

# Loop to capture frames from the webcam
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Perform inference on the current frame
    json_data = CLIENT.infer(frame, model_id="bottle-keypoints/1")
    
    # Convert JSON data to dictionary
    data = json.loads(json.dumps(json_data))
    
    # Variables to store bottom and top keypoint coordinates
    bottom_x, bottom_y = None, None
    top_x, top_y = None, None
    
    # Iterate through predictions
    for prediction in data['predictions']:
        x = int(prediction['x'])
        y = int(prediction['y'])
        width = int(prediction['width'])
        height = int(prediction['height'])

        x1 = int(x - (width / 2))
        y1 = int(y - (height / 2))
        x2 = int(x + (width / 2))
        y2 = int(y + (height / 2))

        # Draw bounding box
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # Draw keypoints
        for keypoint in prediction['keypoints']:
            keypoint_x = int(keypoint['x'])
            keypoint_y = int(keypoint['y'])
            class_name = keypoint['class_name']
            if class_name == 'top':
                color = (0, 0, 255)  # Red color for top keypoints
                top_x, top_y = keypoint_x, keypoint_y
            elif class_name == 'bottom':
                color = (255, 0, 0)  # Blue color for bottom keypoints
                bottom_x, bottom_y = keypoint_x, keypoint_y
            else:
                color = (0, 255, 0)  # Green color for other keypoints
            cv2.circle(frame, (keypoint_x, keypoint_y), 5, color, -1)
    
    # If both bottom and top keypoints are found, calculate the angle
    if bottom_x is not None and bottom_y is not None and top_x is not None and top_y is not None:
        angle = calculate_angle(bottom_x, bottom_y, top_x, top_y)
        
        # Display the angle on the frame
        cv2.putText(frame, "Angle: {:.2f} degrees".format(angle), (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, ( 251, 241, 25), 2)
        
        # Check for orientation
        if 0 <= angle <= 85 or 95 <= angle <= 185:  # Angle close to 0 or 180 degrees
            cv2.putText(frame, "Wrong orientation", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        elif 85 <= angle <= 95 or 265 <= angle <= 275:  # Angle close to 90 degrees
            cv2.putText(frame, "Correct orientation", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    # Display the frame with predictions and angle
    cv2.imshow('Webcam', frame)
    
    # Check for 'q' key press to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close OpenCV windows
cap.release()
cv2.destroyAllWindows()
