import cv2
import os

# Path
image_file = '/content/20150919_174151_image6.png'
label_file = '/content/labels/20150919_174151_image6.txt'

# Read the image
image = cv2.imread(image_file)

# Get image dimensions
height, width = image.shape[:2]

# Open the label file and read each line
with open(label_file, 'r') as file:
    for line in file:
        class_id, x_center, y_center, box_width, box_height = map(float, line.split())

        # Convert normalized coordinates back to pixel coordinates
        x_center = int(x_center * width)
        y_center = int(y_center * height)
        box_width = int(box_width * width)
        box_height = int(box_height * height)

        # Calculate the top-left and bottom-right coordinates
        top_left = (int(x_center - box_width / 2), int(y_center - box_height / 2))
        bottom_right = (int(x_center + box_width / 2), int(y_center + box_height / 2))

        # Draw the bounding box
        cv2.rectangle(image, top_left, bottom_right, (0, 255, 0), 2)

# Display the image
cv2_imshow(image)
cv2.waitKey(0)
cv2.destroyAllWindows()


