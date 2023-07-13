# Pseudo-colorize masked cells
TL;DR: Self-supervised pre-training method for cell detection, which combines masked autoencoding and pseudo-colorization.

## Getting started
Coming soon...

## Self-supervised pre-training

![Pre-training](pre-training.gif)

Masking ratio: 0.33, pre-training target: fluorescence microscopy video pseudo-colorized with the nipy_spectral colormap

## Fine-tuning on cell detection

![Detection](detection.gif)

Input fluorescence microscopy video, predicted centroid heatmaps, and detections (predicted bboxes in green, target bboxes in red)

## TensorFlow implementation
This [repo](https://github.com/roydenwa/cell-centroid-former) contains the code for our CellCentroidFormer model with an EfficientNet backbone.

## Acknowledgements
The code for vision transformer (ViT) models and masked autoencoders (MAEs) is based on lucidrain's [vit_pytorch](https://github.com/lucidrains/vit-pytorch) library.
