"""New-run downstream1:1 contracts. No submissions and no old noise oracle."""
import hashlib
import ast
import importlib.util
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
RUN = HERE.parent.name
OLD = ROOT/'CC-Diff/experimental/fg_rotation_rerender_60k_2x2_20261104_v2/downstream'
OLD_ASYNC = OLD.parent/'downstream_async_v1'
NEW = Path('/data/efs/datasets/xianbao01.hou')
STATE = NEW/'evaluations/RSGen'/RUN/'downstream_dior_mixed_recovery_cpu_v3'
DATASETS = NEW/'datasets/RSGen'/RUN/'downstream_dior_mixed_recovery_cpu_v3'
CHECKPOINTS = NEW/'checkpoints/RSGen'/RUN/'downstream_dior_mixed_recovery_cpu_v3'
METHODS = ('MIGC','GLIGEN','FICGen')
CONDITIONS = ('ours',)
EFS = Path('/data/efs/vepfs_20260622/public/xianbao01.hou')
HRSC = EFS/'dataset/HRSC2016/HRSC2016'
STACK = EFS/'new/RSGen_data/CC-Diff/eval'
R50 = '/data/efs/vepfs_20260622/xianbao01.hou/.cache/torch/hub/checkpoints/resnet50-0676ba61.pth'
DOMAINS = {
 'hrsc':dict(model='S2ANet',K=2808,steps=[1872,2574],lr=.0025,batch=2,
             stack=STACK/'mmrotate',config=STACK/'mmrotate/configs/s2anet/s2anet_r50_fpn_3x_hrsc_le135.py',
             image='d-robotics-image-dev-cn-beijing.cr.volces.com/d-robotics-images/ccdiff:mmrotate_l4',primary='mAP'),
 'dior':dict(model='FasterRCNN',K=9264,steps=[6176,8492],lr=.04,batch=4,
             stack=STACK/'mmdetection',config=STACK/'mmdetection/configs/faster_rcnn/faster_rcnn_r50_fpn_1x_rsvg.py',
             image='d-robotics-image-dev-cn-beijing.cr.volces.com/d-robotics-images/ccdiff:mmdet_l4',primary='bbox_mAP'),
 'dota':dict(model='RotatedFasterRCNN',K=26388,steps=[17592,24189],lr=.005,batch=2,
             stack=STACK/'mmrotate',config=STACK/'mmrotate/configs/rotated_faster_rcnn/rotated_faster_rcnn_r50_fpn_1x_dota_le90.py',
             image='d-robotics-image-dev-cn-beijing.cr.volces.com/d-robotics-images/ccdiff:mmrotate_l4',primary='mAP')}


# HRSC remains with its existing frozen consumer; never route it here.
DOMAINS = {key:value for key,value in DOMAINS.items() if key=='dior'}
METHODS = ('MIGC',)

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()


def digest(obj):
    return hashlib.sha256(json.dumps(obj,sort_keys=True,separators=(',',':')).encode()).hexdigest()


def bind(path):return dict(path=str(Path(path).resolve()),sha256=sha(path))


def write_new(path,obj):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x') as f:
        json.dump(obj,f,indent=2);f.write('\n');f.flush();__import__('os').fsync(f.fileno())


def read_bound(receipt):
    assert sha(receipt['path'])==receipt['sha256'],receipt
    return json.loads(Path(receipt['path']).read_text())


def paths(method,domain,condition):
    assert method in METHODS and domain in DOMAINS and condition in CONDITIONS
    suffix=Path(method)/domain/condition
    return dict(dataset=DATASETS/suffix,evidence=STATE/suffix,checkpoint=CHECKPOINTS/suffix)


def admission_path(method,domain,condition):return HERE/'admissions'/method/domain/condition/'PREFLIGHT_SPEC.json'


def source_pins():
    files=list(HERE.glob('*.py'))+[OLD/name for name in ['contracts.py','audit_combine.py','native_filter_audit.py','exactk_guard.py','verify_result.py']]+[ROOT/'combine_train_gen.py']
    mixed=HERE.parent/'adapters/migc_dior_mixed_recovery_pool_pair_v3'
    files+=list(mixed.glob('*.py'))
    return {str(path.resolve()):bind(path) for path in files}


def assert_cell(record,method,domain,condition,condition_key='condition'):
    assert record['method']==method and record['domain']==domain and record[condition_key]==condition,'Cross-cell receipt rejected'


def verify_pool(method,domain,condition):
    own=paths(method,domain,condition);file=own['dataset']/'COMBINE_COMPLETE.json'
    assert not own['dataset'].is_symlink(),'Cell dataset root must be its own entity'
    record=json.loads(file.read_text());assert record['status']=='PASS_COMPLETE_1TO1_POOL'
    assert_cell(record,method,domain,condition)
    generation=read_bound(record['generation'])
    data,_=shared_protocol(domain)
    from chunk_receipt_contract import validate_ready
    validate_ready(generation,method,domain,condition,sha(data['train_metadata']),data['expected_active']['train'])
    assert generation['method'].upper()==method.upper() and generation['domain']==domain and generation['split']=='train'
    assert generation['condition']==('native_noFG' if condition=='native_noFG' else 'scratchFG_rotation_rerender_no_reference_H_FFT_on')
    for receipt in record['producer_bindings']:read_bound(receipt)
    data,_=shared_protocol(domain)
    assert Path(record['metadata']['path']).resolve()==Path(data['train_metadata']).resolve()
    assert sha(record['metadata']['path'])==record['metadata']['sha256']
    for receipt in record['numerical_sources']:assert sha(receipt['path'])==receipt['sha256']
    return record


def config_dependencies(path,seen=None):
    seen=set() if seen is None else seen;path=Path(path).resolve()
    if path in seen:return {}
    seen.add(path);result={str(path):bind(path)}
    for node in ast.parse(path.read_text()).body:
        if isinstance(node,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='_base_' for t in node.targets):
            bases=ast.literal_eval(node.value)
            for base in ([bases] if isinstance(bases,str) else bases):result.update(config_dependencies(path.parent/base,seen))
    return result


def verify_spec(method,domain,condition):
    path=admission_path(method,domain,condition);record=json.loads(path.read_text())
    assert record['status']=='IMMUTABLE_TARGET_PREFLIGHT_SPEC';assert_cell(record,method,domain,condition)
    assert record['execution_profile']=='CPU_GLOO8_STRUCTURE_ONLY' and record['GPU_count']==0
    own=paths(method,domain,condition)
    assert record['dataset']==str(own['dataset'].resolve()) and record['combine']['path']==str((own['dataset']/'COMBINE_COMPLETE.json').resolve())
    assert sha(record['combine']['path'])==record['combine']['sha256']
    verify_pool(method,domain,condition)
    for receipt in record['source_pins'].values():assert sha(receipt['path'])==receipt['sha256']
    assert record['K']==DOMAINS[domain]['K'] and record['lr_steps']==DOMAINS[domain]['steps']
    return record


def verify_actual(method,domain,condition):
    path=paths(method,domain,condition)['evidence']/'CPU_GLOO8_STRUCTURE.json';record=json.loads(path.read_text())
    assert record['status']=='PASS_CPU_GLOO8_STRUCTURE_ONLY';assert_cell(record,method,domain,condition,'arm')
    assert record['execution_profile']=='CPU_GLOO8_STRUCTURE_ONLY' and record['backend']=='gloo' and record['device']=='cpu'
    assert record['declared_GPU_count']==0 and record['actual_GPU_training_verified'] is False
    assert record['CUDA_guard']['initialized'] is False and record['CUDA_guard']['forbidden_attempts']==[]
    bridge=record['CPU_seed_bridge'];assert sha(bridge['source']['path'])==bridge['source']['sha256']
    for receipt in bridge['native_sync_sources']:assert sha(receipt['path'])==receipt['sha256']
    spec=verify_spec(method,domain,condition)
    assert record['admission']['path']==str(admission_path(method,domain,condition).resolve())
    assert record['admission']['sha256']==sha(record['admission']['path'])
    assert record['combine']==spec['combine']
    assert record['actual_measured_loaders'] and record['all_rank_measurements_equal'] and record['world']==8
    for group in ['native_dependencies','preflight_sources']:
        for receipt in record[group].values():assert sha(receipt['path'])==receipt['sha256']
    observation=record['observations'][condition]
    assert observation['sampler_seed']==42
    assert all(call['input_seed']==call['returned_seed']==42 and call['actual_device']=='cpu'
               for call in bridge['calls'])
    for key in ['dataset_source','sampler_source','dataloader_source','LR_hook_source','IterLoader_source','native_bbox_source']:
        receipt=observation[key];assert sha(receipt['path'])==receipt['sha256']
    assert digest(record['resolved_native_config_literal'])==record['resolved_native_config_sha256']
    for receipt in record['native_eval']['annotation_sources'].values():assert sha(receipt['path'])==receipt['sha256']
    assert sha(record['native_eval']['dataset_source']['path'])==record['native_eval']['dataset_source']['sha256']
    assert record['native_eval']['train_test_disjoint'] is True
    return record


def eval_identity(domain,dataset,train_names):
    import inspect
    config=test_data(domain);ann=Path(config['ann_file'])
    names=[row['filename'] for row in dataset.data_infos]
    assert len(names)==len(set(names))
    def source_stem(name):
        stem=Path(name).stem
        suffix='_1' if domain=='hrsc' else '__synthetic'
        return stem[:-len(suffix)] if stem.endswith(suffix) else stem
    train={source_stem(name) for name in train_names};test={Path(name).stem for name in names}
    assert not train.intersection(test),'Native train/test overlap'
    sources={}
    if ann.is_file():sources['ann_file']=bind(ann)
    if domain=='hrsc':
        for name in names:
            path=Path(config['ann_subdir'])/(Path(name).stem+'.xml');sources[path.name]=bind(path)
    elif domain=='dota':
        for name in names:
            path=ann/(Path(name).stem+'.txt');sources[path.name]=bind(path)
    result=dict(names=names,name_order_sha256=digest(names),name_set_sha256=digest(sorted(names)),
                annotation_sources=sources,dataset_source=bind(inspect.getsourcefile(type(dataset))),
                train_source_name_set_sha256=digest(sorted(train)),train_test_disjoint=True)
    return result


def shared_protocol(domain):
    pointer=json.loads((HERE.parent/'INPUT_AUDIT_INDEX.json').read_text())
    index=read_bound(pointer['canonical_index'])
    assert index['status']=='PASS_SHARED_TRAIN_AND_FROZEN_H_INPUTS'
    data=read_bound(index['data_protocol'])['domains'][domain]
    train=read_bound(index['domains'][domain]['train'])
    assert train['status']=='PASS_ORIGINAL_CANVAS_RGB_TARGET_HED'
    return data,train


def old_contracts():
    spec=importlib.util.spec_from_file_location('_new_run_old_native_contracts',OLD/'contracts.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


def train_data(domain,dataset):
    if domain=='hrsc':
        return dict(type='HRSCDataset',ann_file=str(dataset/'ImageSets/trainval.txt'),
                    img_prefix=str(dataset/'AllImages'),img_subdir=str(dataset/'AllImages'),
                    ann_subdir=str(dataset/'Annotations'),version='le135',classwise=False)
    old=old_contracts();cfg=old.DOMAINS[domain]
    if domain=='dior':return dict(ann_file=str(dataset/'train.json'),img_prefix=str(dataset/'images'),classes=tuple(cfg['names']))
    return dict(ann_file=str(dataset/'labelTxt'),img_prefix=str(dataset/'images'),classes=tuple(cfg['names']),
                version='le90',difficulty=100,filter_empty_gt=True)


def test_data(domain):
    if domain!='hrsc':return old_contracts().test_data(domain)
    return dict(type='HRSCDataset',ann_file=str(HRSC/'ImageSets/test.txt'),
                img_prefix=str(HRSC/'FullDataSet/AllImages'),img_subdir=str(HRSC/'FullDataSet/AllImages'),
                ann_subdir=str(HRSC/'FullDataSet/Annotations'),version='le135',classwise=False)


def load_old_numerics():
    # Only GT construction and native parser/filter helpers are imported.
    # downstream_async_v1/common.py and its CCDiff noise oracle are never used.
    sys.path.insert(0,str(OLD))
    spec=importlib.util.spec_from_file_location('_new_run_old_numerical_combine',OLD/'audit_combine.py')
    mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
    return mod
