import os
import math
import random

def get_distance_class(distance: float):
    if distance < 0.1:
        return 0
    elif distance < 0.5:
        return 1
    elif distance < 1:
        return 2
    elif distance < 2:
        return 3
    return 4

def get_angle_class(angle: float): # TODO: This is not correct
    if angle < -0.5:
        return 0
    elif angle < -0.1:
        return 1
    elif angle < 0.1:
        return 2
    elif angle < 0.5:
        return 3
    return 4

def convert_line(line: str):
    # ['1 445 139 36 24  -1.4480862438197035 0.0 0.027528700497758685 0.413120234994226']
    label, x, y, w, h, relative_x, relative_y, relative_z, angle = line.split(" ")

    center_x = float(x) + float(w) / 2.0
    center_y = float(y) + float(h) / 2.0
    x = float(center_x) / 640.0
    y = float(center_y) / 480.0
    w = float(w) / 640.0
    h = float(h) / 480.0
    
    relative_distance = math.sqrt(float(relative_x) ** 2 +  float(relative_z) ** 2)
    distance_class = get_distance_class(relative_distance)
    angle_class = get_angle_class(float(angle))
    return f"{label} {x} {y} {w} {h} {distance_class} {angle_class}\n"
    
def convert_dir(input_dir, output_dir):
    # Copy imgs to output dir
    input_imgs = os.path.join(root_dir_path, "images")
    output_imgs = os.path.join(output_dir_path, "images")
    os.makedirs(output_imgs, exist_ok=True)
    os.system(f"cp -r {input_imgs}/* {output_imgs}")

    # Convert labels
    input_dir = os.path.join(root_dir_path, "labels")
    output_dir = os.path.join(output_dir_path, "labels")
    os.makedirs(output_dir, exist_ok=True)
    
    for file_name in os.listdir(input_dir):
        file_path = os.path.join(input_dir, file_name)
        output_file_path = os.path.join(output_dir, file_name)
        with open(file_path, "r") as file:
            lines = file.readlines()
            with open(output_file_path, "w") as output_file:
                for line in lines:
                    output_file.write(convert_line(line))

def split_valid(input_dir, output_dir, valid_ratio=0.2):
    input_imgs = os.path.join(input_dir, "images")
    output_imgs = os.path.join(output_dir, "images")
    os.makedirs(output_imgs, exist_ok=True)
    input_labels = os.path.join(input_dir, "labels")
    output_labels = os.path.join(output_dir, "labels")
    os.makedirs(output_labels, exist_ok=True)

    file_names = os.listdir(input_imgs)
    random.shuffle(file_names)

    for file_name in file_names[:int(len(file_names)*valid_ratio)]: # Move valid_ratio of files to valid dir
        file_path = os.path.join(input_imgs, file_name)
        output_file_path = os.path.join(output_imgs, file_name)
        os.system(f"mv {file_path} {output_file_path}")

        file_path = os.path.join(input_labels, file_name).replace(".png", ".txt")
        output_file_path = os.path.join(output_labels, file_name)
        os.system(f"mv {file_path} {output_file_path}")

root_dir_path = "../datasets/duckietown2"
output_dir_path = "../datasets/duckietown2_processed/train"

convert_dir(root_dir_path, output_dir_path)

random.seed(1)
split_valid(output_dir_path, "../datasets/duckietown2_processed/valid")