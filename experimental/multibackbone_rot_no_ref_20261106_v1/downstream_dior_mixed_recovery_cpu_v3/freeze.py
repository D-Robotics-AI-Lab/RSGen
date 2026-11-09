"""Freeze one actual-ready native/ours detector configuration; no GPU work."""
import argparse
import ast
import json
import pprint
from pathlib import Path
from protocol import DOMAINS,METHODS,CONDITIONS,HERE,R50,OLD,paths,bind,sha,write_new,train_data,test_data,verify_actual


def merge_resolved(base,patch):
    """Preserve original values/tuples while applying old Config merge semantics."""
    for key,value in patch.items():
        if isinstance(value,dict):
            value=dict(value)
            replace=value.pop('_delete_',False)
            if replace or not isinstance(base.get(key),dict):base[key]={}
            merge_resolved(base[key],value)
        else:base[key]=value
    return base


def main():
    p=argparse.ArgumentParser();p.add_argument('--method',choices=METHODS,required=True);p.add_argument('--domain',choices=DOMAINS,required=True)
    p.add_argument('--condition',choices=CONDITIONS,required=True);a=p.parse_args()
    own=paths(a.method,a.domain,a.condition);actual=own['evidence']/'CPU_GLOO8_STRUCTURE.json'
    record=verify_actual(a.method,a.domain,a.condition);cfg=DOMAINS[a.domain]
    assert record['execution_profile']=='CPU_GLOO8_STRUCTURE_ONLY' and record['actual_GPU_training_verified'] is False
    assert record['status']=='PASS_CPU_GLOO8_STRUCTURE_ONLY' and record['actual_measured_loaders'] and record['all_rank_measurements_equal']
    assert record['K']==cfg['K'] and record['lr_steps']==cfg['steps'] and record['set_epoch_rollover']=='PASS'
    assert sha(record['combine']['path'])==record['combine']['sha256']
    assert sha(record['native_config']['path'])==record['native_config']['sha256']
    assert sha(record['measurement_source']['path'])==record['measurement_source']['sha256']
    assert sha(HERE/'exactk_guard.py')==sha(OLD/'exactk_guard.py')
    root=HERE/'configs'/a.method/a.domain/a.condition;assert not root.exists()
    lr=dict(record['native_LR'],by_epoch=False,step=cfg['steps'])
    evaluation=dict(_delete_=True,interval=cfg['K']+1,metric='bbox' if a.domain=='dior' else 'mAP')
    if a.domain=='dota':evaluation['iou_thr']=[.5,.75]
    if a.domain=='hrsc':evaluation['iou_thr']=[v/100 for v in range(50,100,5)]
    values=dict(custom_imports=dict(imports=['exactk_guard'],allow_failed_imports=False),
                model=dict(backbone=dict(init_cfg=dict(type='Pretrained',checkpoint=R50))),
                data=dict(train=train_data(a.domain,own['dataset']),val=test_data(a.domain),test=test_data(a.domain),
                          train_dataloader=dict(runner_type='EpochBasedRunner')),
                runner=dict(_delete_=True,type='IterBasedRunner',max_iters=cfg['K']),lr_config=lr,
                checkpoint_config=None,custom_hooks=[dict(type='FourCellExactKGuard',priority='VERY_LOW',K=cfg['K'],
                loader_length=record['observations'][a.condition]['loader_length'],lr_steps=cfg['steps'],
                expected_lr=cfg['lr'],evidence=str(own['evidence']))],evaluation=evaluation,
                workflow=[('train',1)],resume_from=None,load_from=None,auto_resume=False)
    if a.domain=='dior':values['auto_scale_lr']=dict(enable=True,base_batch_size=16)
    values=merge_resolved(ast.literal_eval(record['resolved_native_config_literal']),values)
    values.pop('_base_',None)  # Train consumes a fully resolved sealed config.
    assert values.get('dist_params',{}).get('backend','nccl')=='nccl','Future8GPU training must retain native NCCL'
    assert values['custom_imports']['imports']==['exactk_guard'],'Never install CPU bridge in training'
    root.mkdir(parents=True);config=root/'detector.py'
    config.write_text('# New immutable one-condition fixed-endpoint native detector config.\n'+'\n'.join(k+' = '+pprint.pformat(v,width=110) for k,v in values.items())+'\n')
    frozen=dict(status='FROZEN_FROM_CPU_GLOO8_STRUCTURE',preflight_scope='CPU_GLOO8_STRUCTURE_ONLY',real_8GPU_training_still_requires_separate_admission=True,method=a.method,domain=a.domain,condition=a.condition,
                config=bind(config),actual=bind(actual),combine=record['combine'],guard=bind(HERE/'exactk_guard.py'),
                admission=record['admission'],resolved_native_source_sha256=record['resolved_native_config_sha256'],
                verifier=bind(HERE/'verify_result.py'),K=cfg['K'],lr_steps=cfg['steps'],seed=42,
                work_dir=str(own['checkpoint']),evidence=str(own['evidence']),old_CCDiff_noise_oracle_used=False)
    frozen['support_sources']={p.name:bind(p) for p in HERE.glob('*.py')}
    frozen['support_sources']['original_exactK_verifier']=bind(OLD/'verify_result.py')
    write_new(root/'CONFIG_FROZEN.json',frozen);print(config)


if __name__=='__main__':main()
