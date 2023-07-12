# Pseudo-colorize masked cells
TL;DR: Self-supervised pre-training method for cell detection, which combines masked autoencoding and pseudo-colorization.

## Getting started
Coming soon...

## Fine-tuning on cell detection

![Alt Text](detection.gif)

Input fluorescence microscopy video, predicted centroid heatmaps, and detections (predicted bboxes in green, gt bboxes in red)

## TensorFlow implementation
This [repo](https://github.com/roydenwa/cell-centroid-former) contains the code for our CellCentroidFormer model with an EfficientNet backbone.

## Acknowledgements
The code for vision transformer (ViT) models and masked autoencoders (MAEs) is based on lucidrain's [vit_pytorch](https://github.com/lucidrains/vit-pytorch) library.
