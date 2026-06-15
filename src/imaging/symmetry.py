"""
symmetry.py

Production-grade symmetry analysis module.

Input:
    CT image (normalized)

Output:
    Symmetry Map
    Asymmetry Score
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt


# ============================================================
# Exceptions
# ============================================================

class SymmetryError(Exception):
    pass


# ============================================================
# Symmetry Analyzer
# ============================================================

class SymmetryAnalyzer:

    def __init__(self):
        pass

    # --------------------------------------------------------
    # Midline Estimation
    # --------------------------------------------------------

    def estimate_midline(self, image):

        h, w = image.shape

        return w // 2

    # --------------------------------------------------------
    # Create Mirror Image
    # --------------------------------------------------------

    def mirror_image(
        self,
        image,
        midline
    ):

        mirrored = np.fliplr(image)

        return mirrored

    # --------------------------------------------------------
    # Absolute Difference Matrix
    # --------------------------------------------------------

    def compute_adm(
        self,
        image,
        mirrored
    ):

        adm = np.abs(
            image.astype(np.float32)
            -
            mirrored.astype(np.float32)
        )

        return adm

    # --------------------------------------------------------
    # Asymmetry Score
    # --------------------------------------------------------

    def asymmetry_score(
        self,
        adm
    ):

        return float(
            np.mean(adm)
        )

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

        return (
            image - min_val
        ) / (
            max_val - min_val
        )

    # --------------------------------------------------------
    # Full Pipeline
    # --------------------------------------------------------

    def process(self, image):

        if image.ndim != 2:
            raise SymmetryError(
                "Expected grayscale image."
            )

        midline = self.estimate_midline(
            image
        )

        mirrored = self.mirror_image(
            image,
            midline
        )

        adm = self.compute_adm(
            image,
            mirrored
        )

        score = self.asymmetry_score(
            adm
        )

        adm = self.normalize(adm)

        return {

            "symmetry_map": adm,

            "asymmetry_score": score,

            "midline": midline
        }


# ============================================================
# Visualization
# ============================================================

def visualize_symmetry(
    image,
    symmetry_map
):

    fig, ax = plt.subplots(
        1,
        2,
        figsize=(10, 5)
    )

    ax[0].imshow(
        image,
        cmap="gray"
    )

    ax[0].set_title(
        "CT Image"
    )

    ax[1].imshow(
        symmetry_map,
        cmap="hot"
    )

    ax[1].set_title(
        "Symmetry Map"
    )

    for a in ax:
        a.axis("off")

    plt.tight_layout()
    plt.show()


# ============================================================
# Unit Test
# ============================================================

if __name__ == "__main__":

    import sys

    sys.path.append(
        "/content/stroke_project/src"
    )

    from imaging.dicom_utils import (
        CTDicomProcessor
    )

    TEST_DICOM = (
       "/kaggle/input/brain-stroke-ct-dataset/Brain_Stroke_CT_Dataset/Ischemia/DICOM/15633.dcm"

    )

    processor = CTDicomProcessor()

    result = processor.process(
        TEST_DICOM
    )

    image = result["image"]

    analyzer = SymmetryAnalyzer()

    output = analyzer.process(
        image
    )

    print(
        "Asymmetry Score:",
        output["asymmetry_score"]
    )

    print(
        "Midline:",
        output["midline"]
    )

    visualize_symmetry(
        image,
        output["symmetry_map"]
    )
