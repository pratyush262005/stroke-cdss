
"""
dataset.py

Step 1D:
Stroke Dataset Integration

Combines:
    - CTDicomProcessor
    - SymmetryAnalyzer

Output:
{
    image,
    symmetry_map,
    label,
    sample_id
}
"""

from pathlib import Path
import os

import torch
from torch.utils.data import Dataset

from imaging.dicom_utils import CTDicomProcessor
from imaging.symmetry import SymmetryAnalyzer


# ============================================================
# Label Mapping
# ============================================================

LABEL_MAP = {
    "Normal": 0,
    "Bleeding": 1,
    "Ischemia": 2
}


# ============================================================
# Dataset
# ============================================================

class StrokeDataset(Dataset):

    def __init__(
        self,
        dataset_root,
        image_size=(224, 224)
    ):

        self.dataset_root = Path(dataset_root)

        self.image_size = image_size

        self.dicom_processor = CTDicomProcessor()

        self.symmetry_analyzer = SymmetryAnalyzer()

        self.samples = []

        self._build_index()

    # --------------------------------------------------------
    # Build Dataset Index
    # --------------------------------------------------------

    def _build_index(self):

        classes = [
            "Normal",
            "Bleeding",
            "Ischemia"
        ]

        for cls in classes:

            dicom_dir = (
                self.dataset_root
                / cls
                / "DICOM"
            )

            if not dicom_dir.exists():
                continue

            for file in os.listdir(dicom_dir):

                if not file.endswith(".dcm"):
                    continue

                sample_id = Path(file).stem

                self.samples.append(
                    {
                        "sample_id": sample_id,
                        "class_name": cls,
                        "dicom_path": str(
                            dicom_dir / file
                        )
                    }
                )

        print(
            f"Indexed {len(self.samples)} scans."
        )

    # --------------------------------------------------------
    # Length
    # --------------------------------------------------------

    def __len__(self):
        return len(self.samples)

    # --------------------------------------------------------
    # Get Item
    # --------------------------------------------------------

    def __getitem__(self, idx):

        sample = self.samples[idx]

        # ----------------------------------
        # CT Processing
        # ----------------------------------

        ct_output = (
            self.dicom_processor.process(
                sample["dicom_path"],
                output_size=self.image_size
            )
        )

        image = ct_output["image"]

        # ----------------------------------
        # Symmetry Analysis
        # ----------------------------------

        sym_output = (
            self.symmetry_analyzer.process(
                image
            )
        )

        symmetry_map = (
            sym_output["symmetry_map"]
        )

        # ----------------------------------
        # Label
        # ----------------------------------

        label = LABEL_MAP[
            sample["class_name"]
        ]

        # ----------------------------------
        # Tensor Conversion
        # ----------------------------------

        image = torch.tensor(
            image,
            dtype=torch.float32
        ).unsqueeze(0)

        symmetry_map = torch.tensor(
            symmetry_map,
            dtype=torch.float32
        ).unsqueeze(0)

        label = torch.tensor(
            label,
            dtype=torch.long
        )

        return {

            "image":
                image,

            "symmetry_map":
                symmetry_map,

            "label":
                label,

            "sample_id":
                sample["sample_id"]
        }
