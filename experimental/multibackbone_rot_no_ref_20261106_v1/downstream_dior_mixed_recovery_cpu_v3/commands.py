"""Print target-image CPU combine /8rank preflight / train→verify→eval commands.

No API, submission, environment mutation or execution occurs in this printer.
"""
import argparse
import ast
import json
from pathlib import Path
import shlex
from protocol import ROOT,HERE,DOMAINS,METHODS,CONDITIONS,paths,sha,verify_actual


def cpu_bootstrap():
    # Printed only for combine/structural preflight, never for training/test.
    return ('export CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 '
            'NUMEXPR_NUM_THREADS=1 OPENCV_FOR_THREADS_NUM=1 && '
            'export TORCH_HOME=/data/efs/vepfs_20260622/xianbao01.hou/.cache/torch && ')


def gpu_training_visibility_check():
    # Refuse a leaked CPU-job mask; never unmask GPUs outside assigned devices.
    return 'python -c '+shlex.quote('import os,torch; assert os.environ.get("CUDA_VISIBLE_DEVICES")!=""; assert torch.cuda.device_count()==8, "Separate normal8GPU training admission required"')


def bootstrap():
    # Evaluate only these three pinned literal/f-string assignments. Do not
    # import the legacy submitter (which imports credential-bearing API code).
    source=ROOT/'dag_downstream_train.py'
    selected=[]
    for node in ast.parse(source.read_text()).body:
        if isinstance(node,ast.Assign) and len(node.targets)==1 and isinstance(node.targets[0],ast.Name) and node.targets[0].id in ['TORCH_HOME','PATCH_MMCV','ENV']:
            selected.append(node)
    assert len(selected)==3
    namespace={};exec(compile(ast.fix_missing_locations(ast.Module(body=selected,type_ignores=[])),str(source),'exec'),{'__builtins__':{}},namespace)
    original='MMPATH=$(python -c "import mmcv,os;print(os.path.dirname(mmcv.__file__))")'
    probe=('import mmcv,os; p=os.path.realpath(os.path.dirname(mmcv.__file__)); '
           'assert p.startswith(("/usr/","/opt/","/root/")) and not p.startswith(("/data/efs/","/data/vepfs/")), '
           '"Refuse compatibility patches outside target container-local mmcv"; print(p)')
    env=namespace['ENV'];assert env.count(original)==1
    env=env.replace(original,'MMPATH=$(python -c '+shlex.quote(probe)+')')
    env=env.replace('$MMPATH/parallel/_functions.py','"$MMPATH/parallel/_functions.py"')
    env=env.replace('$MMPATH/parallel/distributed.py','"$MMPATH/parallel/distributed.py"')
    return env


def frozen(method,domain,condition):
    file=HERE/'configs'/method/domain/condition/'CONFIG_FROZEN.json'
    record=json.loads(file.read_text());assert record['status']=='FROZEN_FROM_CPU_GLOO8_STRUCTURE'
    assert record['method']==method and record['domain']==domain and record['condition']==condition
    assert record['actual']['path']==str((paths(method,domain,condition)['evidence']/'CPU_GLOO8_STRUCTURE.json').resolve())
    actual=verify_actual(method,domain,condition)
    assert record['admission']==actual['admission'] and record['combine']==actual['combine']
    for key in ['config','actual','combine','guard','verifier']:
        assert sha(record[key]['path'])==record[key]['sha256'],('Frozen input changed',key)
    for receipt in record['support_sources'].values():assert sha(receipt['path'])==receipt['sha256']
    return record


def command(phase,method,domain,condition,generation=None,pairing=None,attempt=1):
    q=lambda v:shlex.quote(str(v));cfg=DOMAINS[domain];own=paths(method,domain,condition)
    common=f' --method {method} --domain {domain} --condition {condition}'
    if phase=='check-frozen':return 'python '+q(HERE/'commands.py')+' check-frozen'+common
    start=['set -euo pipefail','export PYTHONPATH='+q(HERE)+':'+q(cfg['stack'])+':"${PYTHONPATH:-}"',(cpu_bootstrap() if phase in ['combine','preflight'] else bootstrap())+'cd '+q(cfg['stack'])]
    if phase=='combine':
        assert generation is not None
        return '\n'.join(start+['python '+q(HERE/'cpu_entry.py')+' combine'+common+' --generation '+q(generation)+(' --producer-pairing '+q(pairing) if pairing else '')])
    if phase=='preflight':
        assert attempt>=1
        log=own['evidence']/f'actual_preflight_attempt{attempt:03d}.stdout.log'
        return '\n'.join(start+['python '+q(HERE/'admit.py')+common,'mkdir -p '+q(own['evidence']),
            'test ! -e '+q(own['evidence']/'CPU_GLOO8_STRUCTURE.json'),'test ! -e '+q(log),
            'python '+q(HERE/'cpu_entry.py')+' torchrun --nproc_per_node=8 --master_port=29567 '+q(HERE/'actual_preflight.py')+common+' > '+q(log)+' 2>&1'])
    if phase=='freeze':return 'python '+q(HERE/'freeze.py')+common
    record=frozen(method,domain,condition);config=record['config']['path'];checkpoint=own['checkpoint']/'exactk_final.pth'
    evidence=own['evidence'];verify='python '+q(HERE/'verify_result.py')
    return '\n'.join(start+[gpu_training_visibility_check(),'python '+q(HERE/'commands.py')+' check-frozen'+common,
        'mkdir -p '+q(own['checkpoint'])+' '+q(evidence),
        'test ! -e '+q(checkpoint),'test ! -e '+q(evidence/'train_stdout.log'),
        'bash tools/dist_train.sh '+q(config)+' 8 --seed 42 --no-validate '+('--auto-scale-lr ' if domain=='dior' else '')+'--work-dir '+q(own['checkpoint'])+' > '+q(evidence/'train_stdout.log')+' 2>&1',
        verify+' checkpoint'+common+' > '+q(evidence/'verify_checkpoint.log')+' 2>&1',
        'bash tools/dist_test.sh '+q(config)+' '+q(checkpoint)+' 8 --out '+q(evidence/'eval_final.pkl')+' --tmpdir '+q(evidence/'distributed_eval')+' --eval '+('bbox' if domain=='dior' else 'mAP')+(' --eval-options jsonfile_prefix='+q(evidence/'coco_predictions') if domain=='dior' else '')+' > '+q(evidence/'eval_final.log')+' 2>&1',
        verify+' evaluation'+common])


def main():
    p=argparse.ArgumentParser();p.add_argument('phase',choices=['combine','preflight','freeze','check-frozen','train-eval']);p.add_argument('--method',choices=METHODS,required=True)
    p.add_argument('--domain',choices=DOMAINS,required=True);p.add_argument('--condition',choices=CONDITIONS,required=True)
    p.add_argument('--generation',type=Path);p.add_argument('--producer-pairing',type=Path);p.add_argument('--attempt',type=int,default=1);a=p.parse_args()
    if a.phase=='check-frozen':print('FROZEN_CONFIG_PASS',frozen(a.method,a.domain,a.condition)['config']['path'])
    else:print(command(a.phase,a.method,a.domain,a.condition,a.generation,a.producer_pairing,a.attempt))


if __name__=='__main__':main()
