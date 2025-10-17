
cellphone detector drivers - v2 2023-02-01 2:45pm
==============================

This dataset was exported via roboflow.com on February 1, 2023 at 5:56 PM GMT

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

The dataset includes 3661 images.
Cellphone-in-drivers are annotated in YOLOv8 format.

The following pre-processing was applied to each image:
* Auto-orientation of pixel data (with EXIF-orientation stripping)
* Resize to 640x640 (Stretch)
* Grayscale (CRT phosphor)

The following augmentation was applied to create 3 versions of each source image:
* Equal probability of one of the following 90-degree rotations: none, clockwise, counter-clockwise, upside-down
* Random rotation of between -37 and +37 degrees
* Random shear of between -33° to +33° horizontally and -15° to +15° vertically
* Random exposure adjustment of between -25 and +25 percent

The following transformations were applied to the bounding boxes of each image:
* Random rotation of between -34 and +34 degrees
* Random shear of between -15° to +15° horizontally and -24° to +24° vertically


