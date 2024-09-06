import os
import shutil
import random

# Paths to images and labels folder
images_dir = 'images/'  # Path to folder containing all images
labels_dir = 'labels/'  # Path to folder containing all label .txt files

# Output folders for split data
train_images_dir = 'split/train/images/'
val_images_dir = 'split/val/images/'
test_images_dir = 'split/test/images/'
train_labels_dir = 'split/train/labels/'
val_labels_dir = 'split/val/labels/'
test_labels_dir = 'split/test/labels/'

# Create directories if they don't exist
os.makedirs(train_images_dir, exist_ok=True)
os.makedirs(val_images_dir, exist_ok=True)
os.makedirs(test_images_dir, exist_ok=True)
os.makedirs(train_labels_dir, exist_ok=True)
os.makedirs(val_labels_dir, exist_ok=True)
os.makedirs(test_labels_dir, exist_ok=True)

# Get list of image files with multiple extensions
valid_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff')  # Add any other extensions you need
image_files = [f for f in os.listdir(images_dir) if f.lower().endswith(valid_extensions)]

# Shuffle the dataset
random.shuffle(image_files)

# Split dataset (70% train, 15% validation, 15% test)
train_split = int(0.7 * len(image_files))
val_split = int(0.15 * len(image_files))

train_files = image_files[:train_split]
val_files = image_files[train_split:train_split + val_split]
test_files = image_files[train_split + val_split:]

# Function to move files
def move_files(file_list, source_dir, dest_dir):
    for file_name in file_list:
        shutil.copy(os.path.join(source_dir, file_name), os.path.join(dest_dir, file_name))

# Move image and corresponding label files
def move_image_and_label_files(file_list, images_src_dir, labels_src_dir, images_dst_dir, labels_dst_dir):
    for file_name in file_list:
        # Move image files
        shutil.copy(os.path.join(images_src_dir, file_name), os.path.join(images_dst_dir, file_name))
        # Move corresponding label files
        label_file = os.path.splitext(file_name)[0] + '.txt'
        if os.path.exists(os.path.join(labels_src_dir, label_file)):
            shutil.copy(os.path.join(labels_src_dir, label_file), os.path.join(labels_dst_dir, label_file))

move_image_and_label_files(train_files, images_dir, labels_dir, train_images_dir, train_labels_dir)
move_image_and_label_files(val_files, images_dir, labels_dir, val_images_dir, val_labels_dir)
move_image_and_label_files(test_files, images_dir, labels_dir, test_images_dir, test_labels_dir)

print(f'Total images: {len(image_files)}')
print(f'Training images: {len(train_files)}')
print(f'Validation images: {len(val_files)}')
print(f'Test images: {len(test_files)}')
