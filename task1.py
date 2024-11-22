# Author: Baibhav Shrestha
# Last Modified: 2024-10-5

import numpy as np
from skimage.transform import hough_line, hough_line_peaks, rotate
from skimage.feature import canny
from skimage.io import imread, imsave
from skimage.color import rgb2gray
from skimage import img_as_ubyte
from scipy.stats import mode
import os
import cv2
from ultralytics import YOLO
import yaml
import matplotlib.pyplot as plt
import torch

# Function to display image for visualization
def display_image(image, title):
    plt.imshow(image, cmap='gray')
    plt.title(title)
    plt.axis('off')
    plt.show()

# Function to train YOLO
def train_yolo(yaml_path,config):
    model = YOLO('yolov8n.pt')  # Loading a pre-trained YOLO model 

    # Fetching hyperparameters from config.txt
    epochs = int(config.get('task1_epochs', 50))
    imgsz = int(config.get('task1_imgsz', 640))
    augment = config.get('task1_augment', 'True').lower() == 'true'
    patience = int(config.get('task1_patience', 15))

    # Training the model with hyperparameters
    model.train(
        data=yaml_path,        # Path to the YAML file
        epochs=epochs,         # Number of training epochs
        imgsz=imgsz,           # Image size for training
        augment=augment,       # Data augmentation flag
        patience=patience,     # Early stopping patience
        name='train4'          # Name of the training run to match existing runs
    )
    
    return model

#Function to create yaml file if training the model
def create_yaml_file(yaml_path, task1_input):
    # Ensuring paths for train and val directories exist
    train_dir = os.path.join(task1_input, 'Training').replace("\\", "/")
    val_dir = os.path.join(task1_input, 'Validation').replace("\\", "/")
    
    # Creating the YAML content
    yaml_content = {
        'train': train_dir,  # Training dataset directory
        'val': val_dir,      # Validation dataset directory
        'nc': 1,             # Number of classes 
        'names': ['barcode'] # Class names 
    }
    
    # Saving the YAML file
    with open(yaml_path, 'w') as yaml_file:
        yaml.dump(yaml_content, yaml_file, default_flow_style=False)
    print(f"YAML file created at {yaml_path}")


# Function to add padding to the image to prevent losing data on rotation
def add_padding(image, pad_color=255):
    rows, cols = image.shape[:2]
    diagonal = int(np.ceil(np.sqrt(rows**2 + cols**2)))
    pad_top = (diagonal - rows) // 2
    pad_bottom = diagonal - rows - pad_top
    pad_left = (diagonal - cols) // 2
    pad_right = diagonal - cols - pad_left
    if len(image.shape) == 2:
        padded_image = cv2.copyMakeBorder(image, pad_top, pad_bottom, pad_left, pad_right, cv2.BORDER_CONSTANT, value=pad_color)
    else:
        padded_image = cv2.copyMakeBorder(image, pad_top, pad_bottom, pad_left, pad_right, cv2.BORDER_CONSTANT, value=(pad_color, pad_color, pad_color))
    padding = (pad_top, pad_bottom, pad_left, pad_right)
    return padded_image, (rows, cols), padding

def skew_angle_hough_transform(image):
    edges = canny(image)
    tested_angles = np.deg2rad(np.arange(0.1, 180.0))
    h, theta, d = hough_line(edges, theta=tested_angles)
    accum, angles, dists = hough_line_peaks(h, theta, d)
    if len(angles) > 0:
        most_common_angle = mode(np.around(angles, decimals=2))[0]
        skew_angle = np.rad2deg(most_common_angle - np.pi / 2)
        if abs(skew_angle) > 80:
            skew_angle = skew_angle - 90 if skew_angle > 0 else skew_angle + 90
        print(f"Detected skew angle: {skew_angle} degrees")
        return skew_angle
    else:
        print("No lines detected in the image.")
    return 0


# Function to fix skew and save corrected image if barcode is detected
def correct_image_skew(image_path, output_dir, model, confidence_threshold=0.7):
    image_color = imread(image_path)
    image_gray = rgb2gray(image_color)
    
    # Display original grayscale image before correction
    # display_image(image_gray, 'Original Image (Before Skew Correction)')
    
    padded_image, original_dims, padding = add_padding(image_color)
    angle = skew_angle_hough_transform(image_gray)
    
    # Adding another 90 degrees to images to fix rotation
    if abs(angle) > 15:
        angle += 90
        #print(f"Additional +90 degree rotation applied for {os.path.basename(image_path)}")
    
    corrected_image = rotate(padded_image, angle, cval=1, resize=False, mode='edge')
    corrected_image_8bit = img_as_ubyte(corrected_image)
    
    # Display corrected image
    # display_image(corrected_image_8bit, 'Corrected Image (After Skew Correction)')
    
    # Saving the image first to the corrected_images folder
    output_path = os.path.join(output_dir, 'corrected_' + os.path.basename(image_path))
    imsave(output_path, corrected_image_8bit)

    # Using YOLO to check if barcode exists
    results = model(corrected_image_8bit)
    has_barcode = False
    for result in results:
        if result.boxes.conf is not None:
            confidences = result.boxes.conf.cpu().numpy()
            if any(conf > confidence_threshold for conf in confidences):
                has_barcode = True

    if has_barcode:
        print(f"Barcode detected in {os.path.basename(image_path)}. Image saved to {output_path}")
        return angle, output_path, original_dims, padding
    else:
        if os.path.exists(output_path):
            os.remove(output_path)
            print(f"No barcode detected in {os.path.basename(image_path)}. Image removed from {output_path}")
        return None, None, None, None

# Function to draw bounding box around the number below barcode
def draw_additional_box(image, original_bbox, reduction_factor=0.03):
    # Unpack original_bbox to x1, y1, x2, y2, x3, y3, x4, y4
    x1, y1, x2, y2, x3, y3, x4, y4 = original_bbox

    # Display image before drawing the bounding box
    # display_image(image, 'Image Before Drawing Bounding Box')

    # Defining a region below the barcode to detect the number
    number_region_y = y1 + int(0.8 * (y3 - y1))
    number_region_h = int(0.2 * (y3 - y1))

    reduction_w = int((x2 - x1) * reduction_factor)
    reduction_h = int(number_region_h * reduction_factor)

    # Adjusting the corners for reduction factor and convert to integers
    x1_new = int(x1 + reduction_w)
    y1_new = int(number_region_y + reduction_h)
    x2_new = int(x2 - reduction_w)
    y2_new = y1_new
    x3_new = int(x3 - reduction_w)
    y3_new = int(y1_new + number_region_h - reduction_h)
    x4_new = int(x4 + reduction_w)
    y4_new = y3_new

    new_bbox = [x1_new, y1_new, x2_new, y2_new, x3_new, y3_new, x4_new, y4_new]

    # Drawing bounding box
    image_with_box = cv2.polylines(
        image.copy(),
        [np.array([[x1_new, y1_new], [x2_new, y2_new], [x3_new, y3_new], [x4_new, y4_new]], dtype=np.int32)],
        isClosed=True,
        color=(0, 255, 0), 
        thickness=2
    )
    
    # Display image after drawing bounding box
    # display_image(image_with_box, 'Image After Drawing Bounding Box')

    return image_with_box, new_bbox

# Function to refine the bounding box using CCL
def refine_bbox_with_ccl(image, initial_bbox, min_area=50):
    # Unpack initial_bbox to get coordinates
    x_coords = initial_bbox[::2]
    y_coords = initial_bbox[1::2]
    x_min = max(int(min(x_coords)), 0)
    x_max = min(int(max(x_coords)), image.shape[1])
    y_min = max(int(min(y_coords)), 0)
    y_max = min(int(max(y_coords)), image.shape[0])
    
    # Extract ROI from the image
    roi = image[y_min:y_max, x_min:x_max]
    
    # Convert to grayscale if needed
    if len(roi.shape) == 3:
        gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    else:
        gray_roi = roi.copy()
    
    # Apply thresholding to get binary image
    _, binary_roi = cv2.threshold(gray_roi, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    # Apply morphological operations to clean up noise
    kernel = np.ones((3, 3), np.uint8)
    binary_roi = cv2.morphologyEx(binary_roi, cv2.MORPH_OPEN, kernel, iterations=2)  # Increase iterations for stronger effect
    binary_roi = cv2.morphologyEx(binary_roi, cv2.MORPH_CLOSE, kernel, iterations=2)
    
    # Apply connected components labeling
    num_labels, labels_im = cv2.connectedComponents(binary_roi)
    output = cv2.connectedComponentsWithStats(binary_roi, connectivity=8)
    num_labels = output[0]
    labels = output[1]
    stats = output[2]
    centroids = output[3]
    
    # Collect bounding boxes of connected components
    components = []
    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        if area >= min_area:
            x_comp = stats[i, cv2.CC_STAT_LEFT]
            y_comp = stats[i, cv2.CC_STAT_TOP]
            w_comp = stats[i, cv2.CC_STAT_WIDTH]
            h_comp = stats[i, cv2.CC_STAT_HEIGHT]
            components.append((x_comp, y_comp, w_comp, h_comp))
    
    # Compute the tightest bounding box around all components
    if components:
        x_min_comp = min([c[0] for c in components])
        y_min_comp = min([c[1] for c in components])
        x_max_comp = max([c[0] + c[2] for c in components])
        y_max_comp = max([c[1] + c[3] for c in components])
        
        # Adjust the coordinates to the original image
        x1_new = x_min + x_min_comp
        y1_new = y_min + y_min_comp
        x2_new = x_min + x_max_comp
        y2_new = y1_new
        x3_new = x2_new
        y3_new = y_min + y_max_comp
        x4_new = x1_new
        y4_new = y3_new
        
        refined_bbox = [x1_new, y1_new, x2_new, y2_new, x3_new, y3_new, x4_new, y4_new]
        return refined_bbox
    else:
        # If no components found, return the initial bbox
        return initial_bbox

# Function to crop the image to the area inside the bounding box and save it
def crop_image_to_bbox(image, bbox, output_path):
    x_coords = bbox[::2]
    y_coords = bbox[1::2]
    x_min = max(int(min(x_coords)), 0)
    x_max = min(int(max(x_coords)), image.shape[1])
    y_min = max(int(min(y_coords)), 0)
    y_max = min(int(max(y_coords)), image.shape[0])
    cropped_image = image[y_min:y_max, x_min:x_max]
    cv2.imwrite(output_path, cropped_image)
    return

# Function to transform bounding box coordinates back to the original image after reverse rotation and padding removal
def transform_bbox_back(bbox, angle, original_dims, padding):
    # Unpack padding
    pad_top, pad_bottom, pad_left, pad_right = padding
    original_h, original_w = original_dims

    # Calculate center of the padded image (before reverse rotation)
    padded_h = original_h + pad_top + pad_bottom
    padded_w = original_w + pad_left + pad_right
    center = ((padded_w - 1) / 2.0, (padded_h - 1) / 2.0)  # Center of the padded image

    # Preparing the rotation matrix to reverse the rotation
    angle_rad = np.deg2rad(-angle)  # Reverse the angle 
    cos_a = np.cos(angle_rad)
    sin_a = np.sin(angle_rad)

    # Rotation matrix for reverse rotation
    M = np.array([[cos_a, -sin_a],
                  [sin_a,  cos_a]])

    # Transforming each point in the bounding box
    transformed_bbox = []
    for i in range(0, len(bbox), 2):
        x = bbox[i]
        y = bbox[i + 1]

        # Shifting point to center of padded image
        x_shifted = x - center[0]
        y_shifted = y - center[1]

        # Applying reverse rotation
        x_rotated = M[0, 0] * x_shifted + M[0, 1] * y_shifted
        y_rotated = M[1, 0] * x_shifted + M[1, 1] * y_shifted

        # Shifting back and adjust for padding removal
        x_unpadded = x_rotated + center[0] - pad_left
        y_unpadded = y_rotated + center[1] - pad_top

        # Ensure coordinates are within bounds (prevent negatives)
        x_final = max(0, min(x_unpadded, original_w - 1))
        y_final = max(0, min(y_unpadded, original_h - 1))

        transformed_bbox.extend([x_final, y_final])

    return transformed_bbox

# Function to reverse the skew correction and save the final image without padding
def reverse_image_skew(image_path, angle, original_dims, padding, final_output_dir):
    image = imread(image_path)  # Read the corrected image

    # Display image before reversing skew
    # display_image(image, 'Image Before Reversing Skew')

    # Reverse the rotation by applying the negative of the correction angle
    reversed_image = rotate(image, -angle, cval=1, resize=False, mode='edge')

    # Crop the image back to its original dimensions to remove padding
    original_rows, original_cols = original_dims
    pad_top, pad_bottom, pad_left, pad_right = padding
    
    row_start = pad_top
    col_start = pad_left
    cropped_image = reversed_image[row_start:row_start+original_rows, col_start:col_start+original_cols]

    # Converting to 8-bit before saving
    cropped_image_8bit = img_as_ubyte(cropped_image)

    if not os.path.exists(final_output_dir):
        os.makedirs(final_output_dir)

    # Getting the original image name by removing 'corrected_' prefix if present
    image_name = os.path.basename(image_path)
    if image_name.startswith('corrected_'):
        original_image_name = image_name[len('corrected_'):]
    else:
        original_image_name = image_name

    # Saving the final output image
    output_path = os.path.join(final_output_dir, 'final_' + original_image_name)
    imsave(output_path, cropped_image_8bit)

    # Display image after reversing skew
    # display_image(cropped_image_8bit, 'Image After Reversing Skew')

    #print(f"Image reversed and saved at: {output_path}")
    return output_path

# Function to use YOLO to detect barcodes, draw bounding boxes, and save modified images
def process_corrected_images_with_additional_boxes(model, corrected_image_paths, output_dir, task1_output, bbox_info_list, angle_dict, padding_dict, image_shape_dict):
    confidence_threshold = 0.8  # Threshold for detection confidence

    # Ensuring cropped output directory exists
    os.makedirs(task1_output, exist_ok=True)

    for img_path in corrected_image_paths:
        img = cv2.imread(img_path)
        if img is None:
            print(f"Error loading image {img_path}")
            continue

        corrected_base_name = os.path.basename(img_path)

        # Removing 'corrected_' prefix to get original image name
        if corrected_base_name.startswith('corrected_'):
            original_base_name = corrected_base_name[len('corrected_'):]
        else:
            original_base_name = corrected_base_name

        # Applying YOLO model for barcode detection
        results = model(img)

        has_barcode = False
        for result in results:
            if result.boxes.conf is not None:
                confidences = result.boxes.conf.cpu().numpy()
                if any(conf > confidence_threshold for conf in confidences):
                    has_barcode = True
                    for box in result.boxes:
                        bbox = box.xyxy[0].cpu().numpy().astype(int)

                        # Convert (x1, y1, x2, y2) to (x1, y1, x2, y2, x3, y3, x4, y4)
                        x1, y1 = bbox[0], bbox[1]
                        x2, y2 = bbox[2], bbox[1]
                        x3, y3 = bbox[2], bbox[3]
                        x4, y4 = bbox[0], y3
                        original_bbox = [x1, y1, x2, y2, x3, y3, x4, y4]

                        # Drawing additional bounding box for numbers below barcode
                        img_with_box, additional_bbox = draw_additional_box(img, original_bbox)

                        # Refining the additional bounding box using CCL
                        refined_bbox = refine_bbox_with_ccl(img, additional_bbox)

                        # Drawing the refined bounding box on the image
                        img_with_boxes = img.copy()

                        # Drawing the refined bounding box
                        img_with_boxes = cv2.polylines(
                            img_with_boxes,
                            [np.array([
                                [refined_bbox[0], refined_bbox[1]],
                                [refined_bbox[2], refined_bbox[3]],
                                [refined_bbox[4], refined_bbox[5]],
                                [refined_bbox[6], refined_bbox[7]]
                            ], dtype=np.int32)],
                            isClosed=True,
                            color=(0, 255, 0),  # Green color for the refined bounding box
                            thickness=2
                        )

                        # Saving the image with bounding boxes
                        output_image_path = os.path.join(output_dir, os.path.basename(img_path))
                        cv2.imwrite(output_image_path, img_with_boxes)
                        #print(f"Saved image with bounding boxes as {output_image_path}")

                        # **Modified Part: Rename the cropped image to 'barcodeX.png'**
                        # Extract the base name without extension (e.g., 'img1' from 'img1.png')
                        base_name_no_ext = os.path.splitext(original_base_name)[0]

                        # Removing 'img' prefix if present and add 'barcode' prefix
                        if base_name_no_ext.startswith('img'):
                            barcode_suffix = base_name_no_ext[3:]  # Remove 'img'
                        else:
                            barcode_suffix = base_name_no_ext  # Keep as is if no 'img' prefix

                        # Constructing the new filename
                        barcode_name = f'barcode{barcode_suffix}.png'

                        # Defining the new cropped output path
                        cropped_output_path = os.path.join(task1_output, barcode_name)

                        # Cropping and saving the image with the new name
                        crop_image_to_bbox(img, refined_bbox, cropped_output_path)
                        #print(f"Cropped image saved as {cropped_output_path}")

                        # Saving bbox information for later transformation
                        bbox_info_list.append({
                            'image_name': original_base_name,  # Keeps 'imgX.png' for txt file
                            'bbox': refined_bbox
                        })

                        # Saving image shape
                        image_shape_dict[original_base_name] = img.shape

                        break  # Only process the first detected barcode
            if has_barcode:
                break  # Only process the first image with a barcode

        if not has_barcode:
            print(f"No barcode detected in {img_path}, skipping.")


# Function to save transformed bounding boxes to .txt files
def save_bounding_boxes(bbox_info_list, angle_dict, padding_dict, final_output_dir, task1_output):
    os.makedirs(task1_output, exist_ok=True)
    
    for info in bbox_info_list:
        image_name = info['image_name']
        bbox = info['bbox']
        angle = angle_dict[image_name]  # Get the angle used for correction
        padding = padding_dict[image_name]  # Get the padding used for correction
        
        # Constructing the correct filename for the final image
        final_image_path = os.path.join(final_output_dir, 'final_' + image_name)
        
        if not os.path.exists(final_image_path):
            print(f"Error: Final image not found for {image_name}")
            continue
        
        # Loading the final image to get the shape
        original_dims = imread(final_image_path).shape[:2]  # Get the original dimensions
        
        # Transforming the bbox coordinates back to the final image's orientation
        transformed_bbox = transform_bbox_back(bbox, angle, original_dims, padding)

        # Save to .txt file
        txt_filename = os.path.splitext(image_name)[0] + '.txt'
        txt_path = os.path.join(task1_output, txt_filename)
        with open(txt_path, 'w') as f:
            # Join coordinates with commas
            bbox_str = ','.join(map(str, transformed_bbox))
            f.write(bbox_str)
        print(f"Saved bounding box coordinates to {txt_path}")

def run_task1(image_path, config):
    base_dir = os.getcwd()

    # Paths for test images and corrected images
    task1_input = image_path
    corrected_images_dir = os.path.join(base_dir, 'data', 'task1', 'corrected_images')
    final_output_dir = os.path.join(base_dir, 'data', 'task1', 'final_images')
    task1_output = config.get('task1_output', './output/task1')

    # Creating corrected images directory if it does not exist
    os.makedirs(corrected_images_dir, exist_ok=True)

    # Setting the path for the trained YOLO file
    model_path = config.get('task1_model_path', './data/task1/best.pt')

    # Print out the model_path for debugging
    print(f"Checking for model at: {model_path}")

    if not os.path.exists(model_path):
        print(f"{model_path} not found. Starting model training...")
        
        # Retrieve yaml_path from the config
        yaml_path = config.get('yaml_path', None)

        if not yaml_path:
            print("Error: yaml_path is not defined in config.")
            return

        # Check if YAML file exists, if not create it
        if not os.path.exists(yaml_path):
            print(f"YAML file not found at {yaml_path}. Creating YAML file...")
            create_yaml_file(yaml_path, task1_input)
        
        # Define training and validation directories (if needed)
        train_dir = os.path.join(task1_input, 'Training')
        val_dir = os.path.join(task1_input, 'Validation')
        
        # Train the model with both yaml_path and config
        model = train_yolo(yaml_path, config)
    else:
        try:
            # Load the trained model without yaml_path
            model = YOLO(model_path)
            print(f"Model loaded from {model_path}")
        except Exception as e:
            print(f"Failed to load the model: {e}")
            return

    # Ensuring test image directory exists
    if not os.path.exists(task1_input):
        print(f"Test image directory not found: {task1_input}")
        return

    # Loading all test images from the input folder
    image_names = [
        os.path.join(task1_input, f) 
        for f in os.listdir(task1_input) 
        if f.endswith('.jpg') or f.endswith('.png')
    ]

    # Dictionaries to store information about corrected images
    corrected_images_info = []
    angle_dict = {}
    padding_dict = {}
    image_shape_dict = {}

    # Correcting skew and saving images with barcodes in corrected_images
    for image_name in image_names:
        angle, corrected_image_path, original_dims, padding = correct_image_skew(
            image_name, corrected_images_dir, model
        )
        if angle is not None and corrected_image_path is not None:
            corrected_images_info.append({
                'corrected_image_path': corrected_image_path,
                'angle': angle,
                'original_dims': original_dims,
                'padding': padding
            })
            # Saving angle and padding info
            base_name = os.path.basename(image_name)
            angle_dict[base_name] = angle
            padding_dict[base_name] = padding

    # Processing the corrected images to draw bounding box, and crop the images
    corrected_image_names = [
        info['corrected_image_path'] for info in corrected_images_info
    ]
    bbox_info_list = []
    process_corrected_images_with_additional_boxes(
        model, corrected_image_names, corrected_images_dir, 
        task1_output, bbox_info_list, angle_dict, padding_dict, image_shape_dict
    )

    # Reversing the skew correction and save final images without padding to get bbox in original image form
    for info in corrected_images_info:
        reverse_image_skew(
            info['corrected_image_path'], info['angle'], 
            info['original_dims'], info['padding'], final_output_dir
        )

    # Save the transformed bounding box coordinates to .txt files
    save_bounding_boxes(
        bbox_info_list, angle_dict, padding_dict, 
        final_output_dir, task1_output
    )

    print("Task 1 processing completed successfully.")