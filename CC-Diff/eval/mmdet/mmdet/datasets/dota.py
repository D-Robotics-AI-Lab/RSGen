from .builder import DATASETS
from .coco import CocoDataset

@DATASETS.register_module()
class DOTADataset(CocoDataset):
    
    CLASSES = ("plane","ship","storage-tank","baseball-diamond","tennis-court",
                "basketball-court","ground-track-field","harbor","bridge","large-vehicle",
                "small-vehicle","helicopter","roundabout","soccer-ball-field","swimming-pool")
    
    PALETTE = [(220, 20, 60), (119, 11, 32), (0, 0, 142), (0, 0, 230),
               (106, 0, 228), (0, 60, 100), (0, 80, 100), (0, 0, 70),
               (0, 0, 192), (250, 170, 30), (100, 170, 30), (220, 220, 0),
               (175, 116, 175), (250, 0, 30), (165, 42, 42)]