"""Reuse exact-K/metric parsing; add HRSC native10IoU and actualsplit gates."""
import argparse
import importlib.util
import json
import math
import pickle
from pathlib import Path
import sys
from protocol import OLD,HERE,DOMAINS,METHODS,CONDITIONS,paths,sha,bind,write_new,verify_actual,digest


def legacy():
    sys.path.insert(0,str(OLD))
    spec=importlib.util.spec_from_file_location('_new_run_original_exactK_verifier',OLD/'verify_result.py')
    mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod);return mod


def main():
    p=argparse.ArgumentParser();p.add_argument('phase',choices=['checkpoint','evaluation']);p.add_argument('--method',choices=METHODS,required=True)
    p.add_argument('--domain',choices=DOMAINS,required=True);p.add_argument('--condition',choices=CONDITIONS,required=True);a=p.parse_args()
    own=paths(a.method,a.domain,a.condition);evidence=own['evidence'];actual=verify_actual(a.method,a.domain,a.condition)
    assert actual['status']=='PASS_CPU_GLOO8_STRUCTURE_ONLY' and actual['K']==DOMAINS[a.domain]['K']
    functions=legacy()
    if a.phase=='checkpoint':
        # Same proven function: exact checkpointmeta, optimizer, all LR boundaries,
        # actual sampler epochs/length and terminal K-1 runtime observation.
        functions.checkpoint(actual,own['checkpoint']/'exactk_final.pth',evidence,a.condition)
        return
    verified=json.loads((evidence/'CHECKPOINT_VERIFIED.json').read_text());assert verified['status']=='PASS'
    assert verified['expected_K']==actual['K'] and verified['arm']==a.condition
    assert sha(verified['checkpoint'])==verified['checkpoint_sha256']
    log=evidence/'eval_final.log';text=log.read_text(errors='replace')
    assert 'Traceback (most recent call last)' not in text and 'ChildFailedError' not in text
    with (evidence/'eval_final.pkl').open('rb') as f:predictions=pickle.load(f)
    assert len(predictions)==actual['native_eval_images']
    # Same count is insufficient: rebuild the unchanged native test dataset and
    # require the exact original filename order emitted by its actual parser.
    from mmcv import Config
    if a.domain=='dior':
        from mmdet.datasets import build_dataset
    else:
        import mmrotate
        from mmdet.datasets import build_dataset
    config=Config.fromfile(str(HERE/'configs'/a.method/a.domain/a.condition/'detector.py'))
    config.data.test.test_mode=True
    testset=build_dataset(config.data.test)
    names=[row['filename'] for row in testset.data_infos]
    assert names==actual['native_eval']['names'] and digest(names)==actual['native_eval']['name_order_sha256']
    metrics=functions.metrics_from_log(log);primary=DOMAINS[a.domain]['primary']
    assert all(math.isfinite(float(v)) and 0<=float(v)<=1 for k,v in metrics.items() if k in [primary,'AP50','AP75','mAP_75','bbox_mAP_50','bbox_mAP_75'])
    if a.domain=='hrsc':
        assert all('AP'+str(v) in metrics for v in range(50,100,5)),'HRSC must emit all10IoUs'
        assert actual['native_eval_images']==453 and actual['native_eval_classes']==['ship']
        # Native mAP is evaluator's10IoU mean; not AP50 or a reconstructed rounded average.
        semantics='native HRSC OBB mean mAP over10IoU .50:.05:.95; nativeuse_07_metric=True'
    else:semantics='COCO bbox mAP50:95' if a.domain=='dior' else 'native DOTA mAP=AP50; mAP_75=AP75'
    result=dict(status='PASS_NATIVE_FIXED_ENDPOINT',method=a.method,domain=a.domain,condition=a.condition,
                K=actual['K'],seed=42,native_metrics=metrics,primary_percent=float(metrics[primary])*100,
                metric_semantics=semantics,native_eval_images=len(predictions),
                predictions=bind(evidence/'eval_final.pkl'),checkpoint_verification=bind(evidence/'CHECKPOINT_VERIFIED.json'),
                actual_loader=bind(evidence/'CPU_GLOO8_STRUCTURE.json'),generation=json.loads((own['dataset']/'COMBINE_COMPLETE.json').read_text())['generation'],
                native_eval_name_order_sha256=actual['native_eval']['name_order_sha256'],
                online_validation=False,fixed_terminal_checkpoint=True,old_CCDiff_noise_oracle_used=False)
    write_new(evidence/'RESULT_VERIFIED.json',result);print(json.dumps(result))


if __name__=='__main__':main()
