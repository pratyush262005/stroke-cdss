"""
transforms.py

Step 1E

Medical Image Augmentation Pipeline
"""

import albumentations as A
from albumentations.pytorch import ToTensorV2


# ============================================================
# Train Transform
# ============================================================

def get_train_transforms():

    return A.Compose(

        [

            A.HorizontalFlip(
                p=0.5
            ),

            A.Rotate(
                limit=10,
                p=0.5
            ),

            A.RandomBrightnessContrast(
                brightness_limit=0.1,
                contrast_limit=0.1,
                p=0.5
            ),

            A.GaussNoise(
                p=0.2
            ),

        ],

        additional_targets={
            "symmetry_map": "image",
            "mask": "mask"
        }

    )


# ============================================================
# Validation Transform
# ============================================================

def get_val_transforms():

    return A.Compose(

        [],

        additional_targets={
            "symmetry_map": "image",
            "mask": "mask"
        }

    )
