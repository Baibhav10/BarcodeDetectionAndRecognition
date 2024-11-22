import cv2
import numpy as np
import os

def run_task2(image_path, config):
    # Use paths from config
    input_dir = image_path
    output_dir = config.get('task2_output', './output/task2')

    # List all image files in the input directory
    image_files = [f for f in os.listdir(input_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]

    # Looping through each image file
    for image_file in image_files:
        # Reading the image
        image_path = os.path.join(input_dir, image_file)
        image = cv2.imread(image_path)

        # Converting the image to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Applying Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)

        # Applying adaptive thresholding to get a binary image
        thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                       cv2.THRESH_BINARY_INV, 11, 2)

        # Using Connected Component Labeling (CCL)
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(thresh, connectivity=8)

        # Create an output directory for the image
        image_name = os.path.splitext(image_file)[0]
        image_output_dir = os.path.join(output_dir, image_name)
        if not os.path.exists(image_output_dir):
            os.makedirs(image_output_dir)

        digit_contours = []
        image_height, image_width = image.shape[:2]

        for i in range(1, num_labels):  # Skip the background (label 0)
            x, y, w, h, area = stats[i]

            # Filter based on aspect ratio and size to exclude bars/noise
            aspect_ratio = w / float(h)
            if (area > 100 and 0.3 < aspect_ratio < 1.0 and h > image_height * 0.3):  
                # These thresholds focus on digit-like components
                digit_contours.append((x, y, w, h))

        # Sorting the contours from left to right for sequential digit extraction
        digit_contours = sorted(digit_contours, key=lambda c: c[0])

        # Making sure there are 13 digits (truncate extra ones if necessary)
        digit_contours = digit_contours[:13]

        # Extracting digits and saving them
        for idx, (x, y, w, h) in enumerate(digit_contours):
            # Extract each digit
            digit_image = image[y:y + h, x:x + w]
            digit_filename = f"d{idx + 1:02d}.png"
            digit_filepath = os.path.join(image_output_dir, digit_filename)  # Save in image-specific folder
            cv2.imwrite(digit_filepath, digit_image)

            # Save bounding box coordinates in a text file
            bbox_filename = f"d{idx + 1:02d}.txt"
            bbox_filepath = os.path.join(image_output_dir, bbox_filename)  # Save in image-specific folder
            with open(bbox_filepath, 'w') as f:
                f.write(f"{x},{y},{x + w},{y + h}")
            print(f"Saved {digit_filename} and {bbox_filename} in {image_output_dir}")

            # Drawing the bounding box on the image for visualization
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Save the image with bounding boxes for reference in the image-specific folder
        #output_image_path = os.path.join(image_output_dir, f"{image_name}_boxed.png")
        #cv2.imwrite(output_image_path, image)

    print(f"Task 2 processing completed successfully. Output saved to: {output_dir}")
