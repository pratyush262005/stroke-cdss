"""
segmentation_dataset.py

Dataset for:
1. Stroke Classification
2. Lesion Segmentation

Author: Stroke CDSS Project
"""

import sys

sys.path.append(
    r"D:\vscode\SWIM\stroke-cdss\src"
)

from pathlib import Path

import torch
from torch.utils.data import Dataset

from imaging.dicom_utils import CTDicomProcessor
from imaging.mask_utils import LesionMaskProcessor
from imaging.symmetry import SymmetryAnalyzer


class StrokeSegmentationDataset(Dataset):

    LABEL_MAP = {
        "Normal": 0,
        "Bleeding": 1,
        "Ischemia": 2
    }

    def __init__(self, root_dir):

        self.root_dir = Path(root_dir)

        self.processor = CTDicomProcessor()

        self.mask_processor = (
            LesionMaskProcessor()
        )

        self.symmetry = (
            SymmetryAnalyzer()
        )

        self.samples = []

        # ----------------------------------------
        # Build dataset
        # ----------------------------------------

        for cls in self.LABEL_MAP.keys():

            dcm_dir = (
                self.root_dir /
                cls /
                "DICOM"
            )

            if not dcm_dir.exists():
                continue

            for dcm_file in dcm_dir.glob("*.dcm"):

                self.samples.append({

                    "dcm":
                        dcm_file,

                    "label":
                        self.LABEL_MAP[cls],

                    "class_name":
                        cls
                })

        print(
            f"Loaded {len(self.samples)} samples."
        )

    # ========================================================
    # Dataset length
    # ========================================================

    def __len__(self):

        return len(self.samples)

    # ========================================================
    # Get sample
    # ========================================================

    def __getitem__(self, idx):

        sample = self.samples[idx]

        dcm_path = sample["dcm"]

        label = sample["label"]

        class_name = sample["class_name"]

        # ----------------------------------------------------
        # CT Image
        # ----------------------------------------------------

        image = self.processor.process(
            dcm_path
        )["image"]

        # ----------------------------------------------------
        # Symmetry Map
        # ----------------------------------------------------

        symmetry = self.symmetry.process(
            image
        )["symmetry_map"]

        # ----------------------------------------------------
        # Lesion Mask
        # ----------------------------------------------------

        overlay_path = (

            dcm_path.parent.parent /

            "OVERLAY" /

            f"{dcm_path.stem}.png"
        )

        # For normal scans overlay may not exist.
        # mask_utils.py handles this automatically.

        mask = self.mask_processor.process(
            overlay_path
        )["mask"]

        # ----------------------------------------------------
        # Convert to tensors
        # ----------------------------------------------------

        image = torch.tensor(
            image,
            dtype=torch.float32
        ).unsqueeze(0)

        symmetry = torch.tensor(
            symmetry,
            dtype=torch.float32
        ).unsqueeze(0)

        mask = torch.tensor(
            mask,
            dtype=torch.float32
        ).unsqueeze(0)

        label = torch.tensor(
            label,
            dtype=torch.long
        )

        return {

            "image":
                image,

            "symmetry":
                symmetry,

            "mask":
                mask,

            "label":
                label,

            "class_name":
                class_name
        }


# ==========================================================
# UNIT TEST
# ==========================================================

if __name__ == "__main__":

    dataset = StrokeSegmentationDataset(
        r"D:\vscode\SWIM\Brain_Stroke_CT_Dataset"
    )

    sample = None

    # Find first abnormal sample

    for i in range(len(dataset)):

        if dataset.samples[i][
            "class_name"
        ] != "Normal":

            print(
                "\nFound abnormal sample "
                f"at index: {i}"
            )

            sample = dataset[i]

            break

    if sample is None:

        print(
            "No abnormal samples found!"
        )

    else:

        print("\nSample Keys:")

        print(sample.keys())

        print(
            "\nImage Shape:",
            sample["image"].shape
        )

        print(
            "Symmetry Shape:",
            sample["symmetry"].shape
        )

        print(
            "Mask Shape:",
            sample["mask"].shape
        )

        print(
            "Mask Min:",
            sample["mask"].min()
        )

        print(
            "Mask Max:",
            sample["mask"].max()
        )

        print(
            "Mask Sum:",
            sample["mask"].sum()
        )

        print(
            "Label:",
            sample["label"]
        )

        print(
            "Class:",
            sample["class_name"]
        )