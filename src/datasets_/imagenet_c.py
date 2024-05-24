import os
import torch
import torchvision.datasets as datasets
from .imagenet import ImageNet
import numpy as np


class ImageNetCDatasetWithPaths(datasets.ImageFolder):
    def __init__(self, path, transform):
        super().__init__(path, transform)

    def __getitem__(self, index):
        image, label = super(ImageNetCDatasetWithPaths, self).__getitem__(index)
        return {
            'images': image,
            'labels': label,
            'image_paths': self.samples[index][0]
        }


class ImageNetC(ImageNet):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.train_corruptions = [
            "brightness",
            "elastic_transform",
            "impulse_noise",
            "pixelate",
            "snow",
            "zoom_blur",
            "contrast",
            "fog",
            "gaussian_noise",
            "jpeg_compression",
            "defocus_blur",
            "frost",
            "glass_blur",
            "motion_blur",
            "shot_noise"]
        
        self.test_corruptions = ["gaussian_blur",
                                 "saturate",
                                 "spatter",
                                 "speckle_noise"]

        self.all_corruption_types = self.train_corruptions + self.test_corruptions
        self.max_severity = 5

    def get_test_dataset(self):
        return ImageNetCDatasetWithPaths(path=self.location, transform=self.preprocess)


mCE_normalizer = { ### noise
                    'gaussian_noise': 0.886428,
                    'shot_noise': 0.894468,
                    'impulse_noise': 0.922640,

                    ### blur
                    'defocus_blur': 0.819880,
                    'glass_blur': 0.826268,
                    'motion_blur': 0.785948,
                    'zoom_blur': 0.798360,

                    ### weather
                    'snow':0.866816,
                    'frost':0.826572,
                    'fog':0.819324,
                    'brightness':0.564592,
                    'contrast':0.853204,

                    ### digital
                    'elastic_transform': 0.646056,
                    'pixelate': 0.717840,
                    'jpeg_compression': 0.606500,

                    ### Hold-out
                    'speckle_noise': 0.845388,
                    'saturate':0.658248,
                    'gaussian_blur':0.787108,
                    'spatter':0.717512,
                }
