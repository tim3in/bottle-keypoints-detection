from inference_sdk import InferenceHTTPClient
import cv2
import json

# initialize the client
CLIENT = InferenceHTTPClient(
    api_url="https://detect.roboflow.com",
    api_key="YOUR_API_KEY"
)

# infer on a local image
json_data= CLIENT.infer("bottle.jpg", model_id="bottle-keypoints/1")
print(json_data)
json_string = json.dumps(json_data)
data = json.loads(json_string )

# Load the image
image = cv2.imread("bottle.jpg")  # Replace "your_image.jpg" with your image file path

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
    cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
    
    # Draw keypoints
    for keypoint in prediction['keypoints']:
        keypoint_x = int(keypoint['x'])
        keypoint_y = int(keypoint['y'])
        class_name = keypoint['class_name']
        if class_name == 'top':
            color = (0, 0, 255)  # Red color for top keypoints
        elif class_name == 'bottom':
            color = (255, 0, 0)  # Blue color for bottom keypoints
        else:
            color = (0, 255, 0)  # Green color for other keypoints
        cv2.circle(image, (keypoint_x, keypoint_y), 5, color, -1)

# Display the result
cv2.imshow("Image with Bounding Boxes and Keypoints", image)
cv2.waitKey(0)
cv2.destroyAllWindows()


