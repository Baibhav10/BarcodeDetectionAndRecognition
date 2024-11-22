# task4.py

# Author: Baibhav Shrestha
# Last Modified: 2024-10-5

import os
from task1 import run_task1
from task2 import run_task2
from task3 import run_task3

# Function to concatenate digits from recognized digit files
def concatenate_digits(input_dir, output_dir):
    barcode_folders = [
        f
        for f in os.listdir(input_dir)
        if os.path.isdir(os.path.join(input_dir, f))
    ]

    for folder in barcode_folders:
        # Remove the word 'barcode' from the folder name if it exists
        folder_number = folder.replace('barcode', '')
        # Change the output filename to imgX.txt (where X is the folder number)
        img_output_path = os.path.join(output_dir, f"img{folder_number}.txt")
        barcode_folder_path = os.path.join(input_dir, folder)
        digit_txt_files = [
            f
            for f in os.listdir(barcode_folder_path)
            if f.endswith(".txt") and f.startswith("d")
        ]

        # Sorting the digit files in order
        digit_txt_files.sort()

        barcode_number = ""
        for txt_file in digit_txt_files:
            txt_file_path = os.path.join(barcode_folder_path, txt_file)
            with open(txt_file_path, "r") as f:
                digit = f.read().strip()
                barcode_number += digit

        # Saving the concatenated barcode number with imgX.txt filename
        with open(img_output_path, "w") as f:
            f.write(barcode_number)
        print(
            f"Barcode {folder}: {barcode_number} saved to {img_output_path}"
        )


# Function to run task4
def run_task4(image_path, config):
    base_dir = os.getcwd()

    # Paths for test images and outputs
    test_images_dir = image_path  # Passed in as argument from the command line
    final_output_dir = config.get(
        "task4_output", os.path.join(base_dir, "output", "task4")
    )

    # Run Task 1
    run_task1(test_images_dir, config)

    # Task 1 outputs
    task1_output_dir = config.get(
        "task1_output", os.path.join(base_dir, "output", "task1")
    )

    # Run Task 2 on Task 1 output
    run_task2(task1_output_dir, config)

    # Task 2 outputs
    task2_output_dir = config.get(
        "task2_output", os.path.join(base_dir, "output", "task2")
    )

    # Run Task 3 on Task 2 output
    run_task3(task2_output_dir, config)

    # Task 3 outputs
    task3_output_dir = config.get(
        "task3_output", os.path.join(base_dir, "output", "task3")
    )
    os.makedirs(final_output_dir, exist_ok=True)
    # Concatenate digits and save final output
    concatenate_digits(task3_output_dir, final_output_dir)


    print("Task 4 pipeline completed successfully.")
