_base_ = [
    '../_base_/models/faster_rcnn_r50_fpn.py',
    '../_base_/datasets/rsvg_detection.py',
    '../_base_/schedules/schedule_1x.py', '../_base_/default_runtime.py'
]
classes = ('vehicle','chimney','golffield','Expressway-toll-station','stadium',
    'groundtrackfield','windmill','trainstation','harbor','overpass',
    'baseballfield','tenniscourt','bridge','basketballcourt','airplane',
    'ship','storagetank','Expressway-Service-area','airport','dam')

data_root = '/data/filter_val/'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(800, 800),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]

data = dict(
    test=dict(
        type='CocoDataset', 
        classes=classes,   
        ann_file=data_root + 'dota_mapped_to_dior.json', 
        img_prefix=data_root + 'images',
        pipeline=test_pipeline
    )
)
evaluation = dict(interval=1, metric='bbox', classwise=True)
model = dict(
    roi_head=dict(
        bbox_head=dict(num_classes=20)))