"""Unified dataset and checkpoint paths. Edit DATA_ROOT only."""
import os

DATA_ROOT = "/path/to/data"

CCDIFF_DATASET = os.path.join(DATA_ROOT, "CC-Diff", "dataset")
HRSC_ROOT = os.path.join(DATA_ROOT, "dataset", "HRSC2016", "HRSC2016")

# HED weights (ControlNet Annotators)
HED_CKPT = os.path.join(DATA_ROOT, "ckpts", "lllyasviel", "Annotators", "ControlNetHED.pth")

# DOTA
DOTA_IMAGES_TRAIN = os.path.join(CCDIFF_DATASET, "DOTA", "images", "train")
DOTA_TRAIN_HED = os.path.join(CCDIFF_DATASET, "DOTA", "train_hed")
DOTA_LABELS_TRAIN = os.path.join(CCDIFF_DATASET, "DOTA", "labels", "train_original")
DOTA_LABELS_VAL = os.path.join(CCDIFF_DATASET, "DOTA", "labels", "val_original")
DOTA_IMAGES_VAL = os.path.join(CCDIFF_DATASET, "DOTA", "images", "val")
DOTA_HED_REF_VAL_OUT = os.path.join(CCDIFF_DATASET, "DOTA", "dota_hed_train_ref_val")

# filter_train
FILTER_TRAIN = os.path.join(CCDIFF_DATASET, "filter_train")
FILTER_TRAIN_HED = os.path.join(FILTER_TRAIN, "train_hed")
FILTER_LABEL_TXT = os.path.join(FILTER_TRAIN, "labelTxt")
FILTER_RESULTS_OBB = os.path.join(FILTER_TRAIN, "results_obb_full")
FILTER_TRAIN_REF_ONLY = os.path.join(FILTER_TRAIN, "train_only_ref_train")

# DIOR_NOT_800
DIOR_ROOT = os.path.join(CCDIFF_DATASET, "DIOR_NOT_800")
DIOR_TRAIN_HED = os.path.join(DIOR_ROOT, "train_hed")
DIOR_ANNOTATIONS = os.path.join(DIOR_ROOT, "Annotations")
DIOR_TRAIN_REF_ONLY = os.path.join(DIOR_ROOT, "train_only_train_ref")
DIOR_RESULTS_OBB = os.path.join(DIOR_ROOT, "results_obb")
DIOR_VAL_IMG = os.path.join(DIOR_ROOT, "val")
DIOR_VAL_ANNO = os.path.join(
    CCDIFF_DATASET,
    "DIOR_NOT_800_correct_obbox_no_hbbox",
    "Filter_Add_Oriented_Bounding_Boxes_RSVG",
)
DIOR_HED_REF_VAL_OUT = os.path.join(DIOR_ROOT, "dior_hed_train_ref_val_obbox")

# HRSC2016
HRSC_TRAIN_IMG = os.path.join(HRSC_ROOT, "Train", "AllImages")
HRSC_TRAIN_HED = os.path.join(HRSC_ROOT, "Train", "train_hed")
HRSC_TRAIN_ANNO = os.path.join(HRSC_ROOT, "Train", "Annotations")
HRSC_TRAIN_REF_OBB = os.path.join(HRSC_ROOT, "Train", "train_ref_obb_train")
HRSC_CROP_OUT = os.path.join(HRSC_ROOT, "Train", "hrsc_results_obb_val")
HRSC_TRAIN_RESULT_ROOT = os.path.join(HRSC_ROOT, "Train", "hrsc_results_obb")
HRSC_VAL_IMG = os.path.join(HRSC_ROOT, "Test", "AllImages")
HRSC_VAL_ANNO = os.path.join(HRSC_ROOT, "Test", "Annotations")
HRSC_HED_REF_VAL_OUT = os.path.join(HRSC_ROOT, "Train", "hrsc_hed_ref_val")
