"""
mask_utils.py

Production-grade lesion mask processing.

Brain Stroke CT Dataset
-----------------------
Background = 0
Lesion = 200

Output:
    Binary mask:
        0 -> Background
        1 -> Lesion
"""

from pathlib import Path
from typing import Dict, Any

import cv2
import numpy as np
import matplotlib.pyplot as plt


# ============================================================
# Exceptions
# ============================================================

class MaskProcessingError(Exception):
    pass


# ============================================================
# Main Processor
# ============================================================

class LesionMaskProcessor:

    def __init__(
        self,
        lesion_value=200
    ):
        self.lesion_value = lesion_value

    # --------------------------------------------------------
    # Load Mask
    # --------------------------------------------------------

    def load_mask(self, mask_path):

        mask_path = Path(mask_path)

        # ----------------------------------------------------
        # Normal scans do not have lesion masks.
        # Return empty mask.
        # ----------------------------------------------------

        if not mask_path.exists():

            return np.zeros(
                (224, 224),
                dtype=np.uint8
            )

        mask = cv2.imread(
            str(mask_path),
            cv2.IMREAD_GRAYSCALE
        )

        if mask is None:

            raise MaskProcessingError(
                f"Failed reading mask: {mask_path}"
            )

        return mask

    # --------------------------------------------------------
    # Convert to Binary
    # --------------------------------------------------------

    def binarize(self, mask):

        binary = (
            mask > 0
        ).astype(np.uint8)

        return binary

    # --------------------------------------------------------
    # Resize
    # --------------------------------------------------------

    def resize(
        self,
        mask,
        size=(224, 224)
    ):

        return cv2.resize(
            mask,
            size,
            interpolation=cv2.INTER_NEAREST
        )

    # --------------------------------------------------------
    # Empty Check
    # --------------------------------------------------------

    @staticmethod
    def is_empty(mask):

        return np.sum(mask) == 0

    # --------------------------------------------------------
    # Lesion Area
    # --------------------------------------------------------

    @staticmethod
    def lesion_area(mask):

        return int(np.sum(mask))

    # --------------------------------------------------------
    # Bounding Box
    # --------------------------------------------------------

    @staticmethod
    def bounding_box(mask):

        coords = np.argwhere(mask > 0)

        if len(coords) == 0:
            return None

        y_min, x_min = coords.min(axis=0)
        y_max, x_max = coords.max(axis=0)

        return {
            "x_min": int(x_min),
            "y_min": int(y_min),
            "x_max": int(x_max),
            "y_max": int(y_max)
        }

    # --------------------------------------------------------
    # Centroid
    # --------------------------------------------------------

    @staticmethod
    def centroid(mask):

        coords = np.argwhere(mask > 0)

        if len(coords) == 0:
            return None

        cy, cx = coords.mean(axis=0)

        return {
            "x": float(cx),
            "y": float(cy)
        }

    # --------------------------------------------------------
    # Statistics
    # --------------------------------------------------------

    def compute_statistics(self, mask):

        return {

            "empty":
                self.is_empty(mask),

            "lesion_area":
                self.lesion_area(mask),

            "bbox":
                self.bounding_box(mask),

            "centroid":
                self.centroid(mask)
        }

    # --------------------------------------------------------
    # Full Pipeline
    # --------------------------------------------------------

    def process(
        self,
        mask_path,
        output_size=(224, 224)
    ) -> Dict[str, Any]:

        mask = self.load_mask(
            mask_path
        )

        mask = self.binarize(
            mask
        )

        stats = self.compute_statistics(
            mask
        )

        mask = self.resize(
            mask,
            output_size
        )

        return {

            "mask":
                mask.astype(np.float32),

            "statistics":
                stats
        }


# ============================================================
# Visualization
# ============================================================

def visualize_mask(
    mask,
    title="Mask"
):

    plt.figure(
        figsize=(6, 6)
    )

    plt.imshow(
        mask,
        cmap="gray"
    )

    plt.title(title)

    plt.axis("off")

    plt.show()


# ============================================================
# Overlay Helper
# ============================================================

def overlay_mask_on_image(
    image,
    mask,
    alpha=0.4
):

    image = image.copy()

    if image.max() <= 1:

        image = (
            image * 255
        ).astype(np.uint8)

    rgb = cv2.cvtColor(
        image,
        cv2.COLOR_GRAY2RGB
    )

    rgb[mask > 0] = [
        255, 0, 0
    ]

    return rgb


# ============================================================
# Unit Test
# ============================================================

if __name__ == "__main__":

    TEST_MASK = (
        r"D:\vscode\SWIM\Brain_Stroke_CT_Dataset"
        r"\Bleeding\OVERLAY\10002.png"
    )

    processor = (
        LesionMaskProcessor()
    )

    result = processor.process(
        TEST_MASK
    )

    mask = result["mask"]

    stats = result["statistics"]

    print(
        "\nMask Shape:",
        mask.shape
    )

    print("\nStatistics")

    print("-" * 50)

    for k, v in stats.items():

        print(
            k,
            ":",
            v
        )

    visualize_mask(
        mask,
        title="Processed Lesion Mask"
    )