"""
dicom_utils.py

Production-grade DICOM processing module for
Brain Stroke CT Dataset.

Author: Stroke CDSS Project
"""

from pathlib import Path
from typing import Dict, Any

import logging

import cv2
import numpy as np
import pydicom
import matplotlib.pyplot as plt


# ============================================================
# Logger
# ============================================================

logger = logging.getLogger("DICOM_UTILS")

if not logger.handlers:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s"
    )


# ============================================================
# Exceptions
# ============================================================

class DICOMProcessingError(Exception):
    pass


# ============================================================
# CT DICOM Processor
# ============================================================

class CTDicomProcessor:
    """
    Production-grade CT DICOM processor.

    Pipeline:
        DICOM
           ↓
        Pixel Array
           ↓
        HU Conversion
           ↓
        Brain Window
           ↓
        Normalize
           ↓
        Output
    """

    def __init__(
        self,
        window_center: int = 40,
        window_width: int = 80
    ):
        self.window_center = window_center
        self.window_width = window_width

    # --------------------------------------------------------
    # Read DICOM
    # --------------------------------------------------------

    def read_dicom(self, dicom_path: str):

        dicom_path = Path(dicom_path)

        if not dicom_path.exists():
            raise FileNotFoundError(
                f"DICOM file not found: {dicom_path}"
            )

        try:
            ds = pydicom.dcmread(str(dicom_path))

            return ds

        except Exception as e:
            raise DICOMProcessingError(
                f"Failed reading DICOM: {e}"
            )

    # --------------------------------------------------------
    # Convert to Hounsfield Units
    # --------------------------------------------------------

    def to_hu(self, ds):

        image = ds.pixel_array.astype(np.float32)

        slope = float(
            getattr(ds, "RescaleSlope", 1)
        )

        intercept = float(
            getattr(ds, "RescaleIntercept", 0)
        )

        hu = image * slope + intercept

        return hu

    # --------------------------------------------------------
    # MONOCHROME Fix
    # --------------------------------------------------------

    def fix_photometric(self, image, ds):

        photometric = getattr(
            ds,
            "PhotometricInterpretation",
            "MONOCHROME2"
        )

        if photometric == "MONOCHROME1":
            image = np.max(image) - image

        return image

    # --------------------------------------------------------
    # Windowing
    # --------------------------------------------------------

    def apply_window(
        self,
        image,
        center=None,
        width=None
    ):

        center = center or self.window_center
        width = width or self.window_width

        lower = center - width / 2
        upper = center + width / 2

        image = np.clip(
            image,
            lower,
            upper
        )

        return image

    # --------------------------------------------------------
    # Normalize
    # --------------------------------------------------------

    @staticmethod
    def normalize(image):

        image = image.astype(np.float32)

        min_val = image.min()
        max_val = image.max()

        if max_val - min_val < 1e-8:
            return np.zeros_like(image)

        image = (
            image - min_val
        ) / (
            max_val - min_val
        )

        return image

    # --------------------------------------------------------
    # Resize
    # --------------------------------------------------------

    @staticmethod
    def resize(
        image,
        size=(224, 224)
    ):

        return cv2.resize(
            image,
            size,
            interpolation=cv2.INTER_LINEAR
        )

    # --------------------------------------------------------
    # Metadata
    # --------------------------------------------------------

    def extract_metadata(self, ds):

        metadata = {

            "rows":
                getattr(ds, "Rows", None),

            "cols":
                getattr(ds, "Columns", None),

            "pixel_spacing":
                getattr(ds, "PixelSpacing", None),

            "window_center":
                getattr(ds, "WindowCenter", None),

            "window_width":
                getattr(ds, "WindowWidth", None),

            "rescale_slope":
                getattr(ds, "RescaleSlope", None),

            "rescale_intercept":
                getattr(ds, "RescaleIntercept", None),

            "photometric":
                getattr(
                    ds,
                    "PhotometricInterpretation",
                    None
                )
        }

        return metadata

    # --------------------------------------------------------
    # Full Pipeline
    # --------------------------------------------------------

    def process(
        self,
        dicom_path,
        output_size=(224, 224)
    ) -> Dict[str, Any]:

        ds = self.read_dicom(dicom_path)

        metadata = self.extract_metadata(ds)

        image = self.to_hu(ds)

        image = self.fix_photometric(
            image,
            ds
        )

        image = self.apply_window(
            image
        )

        image = self.normalize(
            image
        )

        image = self.resize(
            image,
            output_size
        )

        return {
            "image": image.astype(np.float32),
            "metadata": metadata
        }


# ============================================================
# Visualization
# ============================================================

def visualize_ct(image, title="CT"):

    plt.figure(figsize=(6, 6))

    plt.imshow(
        image,
        cmap="gray"
    )

    plt.title(title)

    plt.axis("off")

    plt.show()


# ============================================================
# Unit Test
# ============================================================

if __name__ == "__main__":

    TEST_DICOM = (
        "/kaggle/input/"
        "brain-stroke-ct-dataset/"
        "Brain_Stroke_CT_Dataset/"
        "Ischemia/"
        "DICOM/"
        "15633.dcm"
    )

    processor = CTDicomProcessor(
        window_center=40,
        window_width=80
    )

    result = processor.process(
        TEST_DICOM
    )

    image = result["image"]

    metadata = result["metadata"]

    print("\nMetadata")
    print("-" * 50)

    for k, v in metadata.items():
        print(f"{k}: {v}")

    print("\nImage Shape:", image.shape)
    print("Image Dtype:", image.dtype)
    print("Min:", image.min())
    print("Max:", image.max())

    visualize_ct(
        image,
        title="Brain Window CT"
    )
