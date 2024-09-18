
fishlens-modelv1 - v2 2024-09-05 5:02am
==============================

This dataset was exported via roboflow.com on September 5, 2024 at 5:13 AM GMT

Roboflow is an end-to-end computer vision platform that helps you
* collaborate with your team on computer vision projects
* collect & organize images
* understand and search unstructured image data
* annotate, and create datasets
* export, train, and deploy computer vision models
* use active learning to improve your dataset over time

For state of the art Computer Vision training notebooks you can use with this dataset,
visit https://github.com/roboflow/notebooks

To find over 100k other datasets and pre-trained models, visit https://universe.roboflow.com

The dataset includes 7500 images.
Fish-skin-diseases are annotated in YOLO v5 PyTorch format.

The following pre-processing was applied to each image:
* Auto-orientation of pixel data (with EXIF-orientation stripping)
* Resize to 640x640 (Fit within)
* Auto-contrast via histogram equalization

The following augmentation was applied to create 3 versions of each source image:
* Equal probability of one of the following 90-degree rotations: none, clockwise, counter-clockwise, upside-down
* Random rotation of between -45 and +45 degrees
* Random brigthness adjustment of between -15 and +15 percent
* Random exposure adjustment of between -15 and +15 percent
* Salt and pepper noise was applied to 0.1 percent of pixels


