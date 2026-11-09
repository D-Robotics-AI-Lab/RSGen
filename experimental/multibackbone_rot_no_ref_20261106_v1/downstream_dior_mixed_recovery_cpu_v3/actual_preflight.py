"""Actual8rank target-image native dataset/loader/LR probe; no detector training."""
import argparse
import ast
from collections import Counter
import inspect
import json
import pprint
from pathlib import Path
from protocol import (HERE,DOMAINS,METHODS,CONDITIONS,paths,train_data,test_data,sha,bind,digest,write_new,
                      verify_pool,verify_spec,admission_path,config_dependencies,source_pins,eval_identity)


def identity(dataset,loader,domain):
    import numpy as np
    names=[];parsed={};dims={};objects=0;real=0;synthetic=0
    for i,info in enumerate(dataset.data_infos):
        name=info['filename'];assert Path(name).name==name
        ann=dataset.get_ann_info(i);payload={}
        for key in ['bboxes','labels','bboxes_ignore','polygons','polygons_ignore']:
            if key in ann:
                value=np.ascontiguousarray(ann[key]);payload[key]=dict(dtype=str(value.dtype),shape=list(value.shape),sha256=__import__('hashlib').sha256(value.tobytes()).hexdigest())
        names.append(name);parsed[name]=digest(payload);objects+=len(ann['labels'])
        is_syn=Path(name).stem.endswith('_1' if domain=='hrsc' else '__synthetic')
        synthetic+=int(is_syn);real+=int(not is_syn)
        dims[name]=[info.get('width'),info.get('height')]
    assert len(names)==len(set(names))==len(dataset)
    assert real==synthetic,'Native-filtered real:synthetic ratio changed'
    sampler=loader.sampler;assert type(sampler).__name__=='DistributedGroupSampler'
    return dict(dataset_length=len(dataset),loader_length=len(loader),sampler=type(sampler).__name__,
                sampler_length=len(sampler),world=8,classes=list(dataset.CLASSES),native_GT_objects=objects,
                effective_real=real,effective_synthetic=synthetic,name_set_sha256=digest(sorted(names)),
                native_name_order_sha256=digest(names),parsed_GT_sha256=digest(parsed),dimensions_sha256=digest(dims),
                flags_sha256=digest(dataset.flag.tolist()),flags={str(k):v for k,v in Counter(dataset.flag.tolist()).items()},
                dataset_source=bind(inspect.getsourcefile(type(dataset))),sampler_source=bind(inspect.getsourcefile(type(sampler))))


def main():
    p=argparse.ArgumentParser();p.add_argument('--method',choices=METHODS,required=True);p.add_argument('--domain',choices=DOMAINS,required=True)
    p.add_argument('--condition',choices=CONDITIONS,required=True);p.add_argument('--local-rank','--local_rank',type=int,default=0)
    a=p.parse_args();own=paths(a.method,a.domain,a.condition);complete=own['dataset']/'COMBINE_COMPLETE.json'
    receipt=verify_pool(a.method,a.domain,a.condition)
    admission=verify_spec(a.method,a.domain,a.condition)
    output=own['evidence']/'CPU_GLOO8_STRUCTURE.json';assert not output.exists()
    from cpu_bridge import install_no_cuda_guard, install_seed_bridge
    cuda_guard=install_no_cuda_guard()
    import torch
    import torch.distributed as dist
    torch.set_num_threads(1)
    import mmcv
    from mmcv import Config
    from mmcv.runner import get_dist_info
    from mmcv.runner.iter_based_runner import IterLoader
    from mmcv.runner.hooks import StepLrUpdaterHook
    if a.domain=='dior':from mmdet.utils import compat_cfg
    else:
        import mmrotate
        from mmrotate.utils import compat_cfg
    from mmdet.datasets import build_dataset,build_dataloader
    dist.init_process_group(backend='gloo',init_method='env://')
    rank,world=get_dist_info();assert world==8 and dist.get_backend()=='gloo'
    seed_bridge=install_seed_bridge(build_dataloader)
    cuda_guard.check()
    native=DOMAINS[a.domain];cfg=Config.fromfile(str(native['config']));assert cfg.model.type==native['model']
    cfg.merge_from_dict({'data.train.'+k:v for k,v in train_data(a.domain,own['dataset']).items()})
    cfg.merge_from_dict({'data.test.'+k:v for k,v in test_data(a.domain).items()})
    cfg=compat_cfg(cfg);options=dict(cfg.data.train_dataloader)
    options.update(num_gpus=8,dist=True,seed=42,runner_type='EpochBasedRunner',persistent_workers=False,workers_per_gpu=0)
    assert options['samples_per_gpu']==native['batch']
    dataset=build_dataset(cfg.data.train);loader=build_dataloader(dataset,**options)
    assert loader.sampler.seed==42 and loader.sampler.num_replicas==8
    measured=identity(dataset,loader,a.domain)
    measured['sampler_seed']=42
    measured['dataloader_source']=bind(inspect.getsourcefile(build_dataloader))
    measured['LR_hook_source']=bind(inspect.getsourcefile(StepLrUpdaterHook))
    measured['IterLoader_source']=bind(inspect.getsourcefile(IterLoader))
    measured['native_bbox_source']=bind(native['stack']/('mmdet' if a.domain=='dior' else 'mmrotate')/'core/bbox/transforms.py')
    expected_length={'hrsc':78,'dior':772,'dota':2199}[a.domain]
    assert measured['loader_length']==expected_length,('Actual mixed-loader differs from fixed protocol',measured)
    # Match tools/test.py; otherwise build_dataset defaults to training mode
    # and can silently filter empty held-out images in this measurement.
    cfg.data.test.test_mode=True
    testset=build_dataset(cfg.data.test)
    expected_test={'hrsc':453,'dior':3372,'dota':5657}[a.domain]
    assert len(testset)==expected_test
    native_eval=eval_identity(a.domain,testset,[info['filename'] for info in dataset.data_infos])
    class Sampler:
        epoch=0
        def set_epoch(self,value):self.epoch=value
    class Tiny:
        sampler=Sampler()
        def __iter__(self):return iter([0,1])
        def __len__(self):return 2
    tiny=Tiny();iterator=IterLoader(tiny)
    assert [next(iterator),next(iterator),next(iterator)]==[0,1,0] and iterator.epoch==tiny.sampler.epoch==1
    effective=float(cfg.optimizer.lr)
    if a.domain=='dior':effective*=8*native['batch']/cfg.auto_scale_lr.base_batch_size
    assert abs(effective-native['lr'])<1e-12
    lr_cfg=dict(cfg.lr_config);lr_cfg.pop('policy');lr_cfg.update(by_epoch=False,step=native['steps'])
    hook=StepLrUpdaterHook(**lr_cfg)
    optimizer=torch.optim.SGD([torch.nn.Parameter(torch.zeros(1))],lr=effective)
    from types import SimpleNamespace
    runner=SimpleNamespace(optimizer=optimizer,iter=0);hook.before_run(runner)
    values={};K=native['K'];steps=native['steps']
    for point in sorted({0,499,500,K-1,*steps,*(s-1 for s in steps)}):
        runner.iter=point;hook.before_train_iter(runner)
        factor=hook.gamma**sum(point>=s for s in steps)
        if point<500:factor*=hook.warmup_ratio+(1-hook.warmup_ratio)*point/500
        actual=optimizer.param_groups[0]['lr'];assert abs(actual-effective*factor)<1e-12;values[point]=actual
    rank_identity=dict(train=measured,native_eval_sha256=digest(native_eval))
    records=[None]*world;dist.all_gather_object(records,rank_identity);assert all(v==rank_identity for v in records)
    resolved=pprint.pformat(cfg._cfg_dict.to_dict(),width=110)
    assert isinstance(ast.literal_eval(resolved),dict)
    cuda_guard.check()
    bridge_receipt=seed_bridge.receipt()
    result=dict(status='PASS_CPU_GLOO8_STRUCTURE_ONLY',execution_profile='CPU_GLOO8_STRUCTURE_ONLY',backend='gloo',device='cpu',declared_GPU_count=0,actual_GPU_training_verified=False,real_loader_batches_iterated=0,CUDA_guard=cuda_guard.receipt(),CPU_seed_bridge=bridge_receipt,method=a.method,domain=a.domain,arm=a.condition,world=8,
                K=K,lr_steps=steps,observations={a.condition:measured},native_LR=dict(cfg.lr_config),
                native_eval_images=len(testset),native_eval_classes=list(testset.CLASSES),effective_lr=effective,
                native_eval=native_eval,
                measured_LR_boundaries=values,set_epoch_rollover='PASS',all_rank_measurements_equal=True,
                combine=bind(complete),measurement_source=bind(__file__),native_config=bind(native['config']),
                admission=bind(admission_path(a.method,a.domain,a.condition)),
                native_dependencies=config_dependencies(native['config']),
                preflight_sources=source_pins(),
                resolved_native_config_literal=resolved,resolved_native_config_sha256=digest(resolved),
                actual_measured_loaders=True,detector_model_constructed=False,old_CCDiff_noise_oracle_used=False)
    dist.barrier();cuda_guard.check()
    dist.destroy_process_group();cuda_guard.check()
    result['CUDA_guard']=cuda_guard.receipt()
    if rank==0:write_new(output,result);print('CPU_GLOO8_STRUCTURE_PASS',json.dumps(result))


if __name__=='__main__':main()
