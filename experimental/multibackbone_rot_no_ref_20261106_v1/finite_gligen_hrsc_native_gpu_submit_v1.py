"""Main-only fixed-K8L4 GLIGEN HRSC native detector train/eval from sealed CPU continuation.

New source lives in a linked worktree (supervisor decision 2026-11-08 15:08 UTC); all
ROOT/MON/EFS references are explicit absolute paths into the original project checkout,
never derived from __file__. Dynamic run-data (spec, manifest, canonical, logs) stays at
the original MON/EFS absolute paths.
"""
import argparse, contextlib, fcntl, hashlib, importlib.util, io, json, os, shlex, sys
from datetime import datetime, timezone
from pathlib import Path

RUN = 'multibackbone_rot_no_ref_20261106_v1'
ROOT = Path('/data/vepfs/users/xianbao01.hou/RSGen')
MON = ROOT / '.train_monitor_multibackbone_rot_no_ref_20261106'
REG = MON / 'finite_detector_submissions'
ROUTES = {'jrwjn': ('q-20260429225420-jrwjn', 'ml.gni3cl.45xlarge')}

STAGE = 'gligen_hrsc_native_gpu_train_a001'
TARGET_MAIN = MON / 'execution_acceptances/gligen_hrsc_native_cpu_continuation_target_main_k5c7q.json'
ROOT_REVIEW = MON / 'source_reviews/ROOT_GLIGEN_HRSC_NATIVE_ACTUAL_CPU_CONTINUATION_ACCEPTANCE_REVIEW_20261108_v1.json'
GPU_SCRIPT = Path('/data/efs/datasets/xianbao01.hou/evaluations/RSGen/multibackbone_rot_no_ref_20261106_v1/gligen_hrsc_native_cpu_continuation_v1/gligen_hrsc_native_cpu_continuation_owner_bound_a002/GPU_TRAIN_EVAL.sh')
PROFILE = MON / 'resource_audits/S2ANET_HRSC_TRAINING_40CPU128_PROFILE_20261107_v1.json'
IMAGE_RECEIPT = MON / 'CPU_PREFLIGHT_TARGET_IMAGE_RECEIPT_V1.json'
NATIVE_FULL_MAIN = MON / 'GLIGEN_HRSC_NATIVE_TRAIN_FULL617_GENERATION_ACCEPTANCE.json'
RESOURCE_PLAN = MON / 'resource_audits/MAIN_ADMITTED_GLIGEN_HRSC_NATIVE_GPU_TRAIN_20261108_v1.json'
LAUNCHER = MON / f'runtime_preparation/{STAGE}/launcher.sh'
MANIFEST = MON / f'runtime_preparation/{STAGE}/SOURCE_MANIFEST.json'
SPEC = MON / f'runtime_preparation/{STAGE}/ACTUAL_FINAL_FINITE_SPEC.json'
WORK_DIR = Path('/data/efs/datasets/xianbao01.hou/checkpoints/RSGen/multibackbone_rot_no_ref_20261106_v1/gligen_hrsc_typed_downstream_v1/GLIGEN/hrsc/native_noFG')
EVIDENCE = Path('/data/efs/datasets/xianbao01.hou/evaluations/RSGen/multibackbone_rot_no_ref_20261106_v1/gligen_hrsc_typed_downstream_v1/GLIGEN/hrsc/native_noFG')
LAUNCHER_OUT = EVIDENCE.parent / 'native_noFG_train_launcher_a001'

TARGET_INSTALLATION_PINS = {
    '/root/miniconda3/lib/python3.9/site-packages/mmdet/datasets/samplers/group_sampler.py': '44e5684c3ccbeea26e4fb87d457576cf01a131afb0f746057608f2969405e09f',
    '/root/miniconda3/lib/python3.9/site-packages/mmdet/datasets/builder.py': 'cee8ce4dc3a550ebcc2f7fdb792c05074b4ae3a6be4e5ee141814cb0a9bad5ee',
    '/root/miniconda3/lib/python3.9/site-packages/mmcv/runner/hooks/lr_updater.py': 'a9d9f5691d7880c9dccd6b031dba75eb8198c0696d8a8265e7a904c689436b31',
    '/root/miniconda3/lib/python3.9/site-packages/mmcv/runner/iter_based_runner.py': 'c9c814d40aff5c560bc9aa9bd34f1b7a163e4434d04e0092b257c72b7f4d6ddd',
}


def sha(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()


def read(p):
    return json.loads(Path(p).read_text())


def now():
    return datetime.now(timezone.utc).isoformat()


def atomic(p, j, exclusive=False):
    if exclusive:
        with Path(p).open('x') as f:
            json.dump(j, f, indent=2)
            f.write('\n')
            f.flush()
            os.fsync(f.fileno())
    else:
        t = Path(p).with_name(Path(p).name + '.preparing')
        with t.open('x') as f:
            json.dump(j, f, indent=2)
            f.write('\n')
            f.flush()
            os.fsync(f.fileno())
        os.replace(t, p)


def validate(s):
    assert s['run'] == RUN and s['method'] == 'GLIGEN' and s['domain'] == 'hrsc' and s['condition'] == 'native_noFG'
    assert s['stage'] == STAGE and s['queue'] in ROUTES and s['finite'] and s['holding_time_seconds'] is None
    assert s['K'] == 2808 and s['seed'] == 42
    assert s['resources'] == dict(gpu_count=8, cpu=40, memory_gib=128, gpu_type='NVIDIA-L4', family=ROUTES[s['queue']][1], replicas=1)
    p = Path(s['resource_plan'])
    assert sha(p) == s['resource_plan_sha256']
    r = read(p)
    assert r['main_admitted'] and s['queue'] in r['admitted_queues'] and datetime.now(timezone.utc) < datetime.fromisoformat(r['valid_until'])
    assert sha(PROFILE) == '5503e921abeba2a07086620d4ad477c7c0324c56b917acf1539d35cc31532b26'
    assert sha(ROOT_REVIEW) == '46161c4be294a7ce3ec036dfd252c40929f07ab0d668eb55d1034033cf05c82f'
    tm = read(TARGET_MAIN)
    assert sha(TARGET_MAIN) == '076a3d3bfa394c0d7060ccb46bccecbe5bacf6f2aaeb385819ec301d0a30b03d'
    assert tm['status'] == 'OWNER_TARGET_CPU_MAIN_ACCEPTED_SEPARATE_8GPU_ADMISSION_NEXT'
    assert tm['job'] == 't-20261108215606-k5c7q'
    assert tm['owner_guards_at_accept']['gpu_script_is_print_only_not_executed'] is True
    assert read(IMAGE_RECEIPT)['image_url'] == s['image'] == 'd-robotics-image-dev-cn-beijing.cr.volces.com/d-robotics-images/ccdiff:mmrotate_l4'
    assert sha(NATIVE_FULL_MAIN) == '09c0b40ad6ecd38f93b91d19c4c68e160a7d2513640388ed8b4d31f2a828c987'
    assert sha(GPU_SCRIPT) == 'd490bf51574d7cbab7ba3aca3656fa057c14da4bb3d91b1bba468b6202714898'
    m = read(MANIFEST)
    assert m['files'] and m['stage'] == STAGE
    seen = set()
    for x in m['files']:
        assert x['path'] not in seen
        seen.add(x['path'])
        if x['path'] in TARGET_INSTALLATION_PINS:
            assert x['sha256'] == TARGET_INSTALLATION_PINS[x['path']], 'container pin drift'
        else:
            assert sha(x['path']) == x['sha256'], ('sealed source drift', x['path'])
    assert any(x['path'] == str(LAUNCHER) for x in m['files'])
    assert any(x['path'] == str(GPU_SCRIPT) for x in m['files'])
    launch = Path(LAUNCHER).read_text()
    assert 'dist_train' not in launch and 'GPU_TRAIN_EVAL.sh' in launch and 'set -Eeuo pipefail' in launch
    for q in [WORK_DIR / 'exactk_final.pth', EVIDENCE / 'train_stdout.log', LAUNCHER_OUT]:
        assert not Path(q).exists(), ('New detector attempt required', str(q))
    return s


def load_dag():
    spec = importlib.util.spec_from_file_location('rsgen_dag_train_gen', str(ROOT / 'dag_train_gen.py'))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--spec', type=Path, required=True)
    ap.add_argument('--submit', action='store_true')
    a = ap.parse_args()
    s = validate(read(a.spec))
    if not a.submit:
        print(json.dumps(dict(status='GLIGEN_HRSC_NATIVE_GPU_SPEC_VALIDATED_NOT_SUBMITTED', K=s['K'], resources=s['resources'])))
        return
    REG.mkdir(exist_ok=True)
    with (REG / '.dispatch.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        out = REG / (s['stage'] + '.json')
        assert not out.exists(), 'Prior intent: reconcile exactJob before retry'
        for p in REG.glob('*.json'):
            old = read(p)
            if old.get('method') == 'GLIGEN' and old.get('domain') == 'hrsc' and old.get('condition') == 'native_noFG' and old.get('kind') == 'detector_train':
                assert old.get('confirmed_terminal') is True, 'Prior same producer unresolved'
        code = ("import hashlib,json,pathlib,os,sys;h=lambda b:hashlib.sha256(b).hexdigest();"
                "b=pathlib.Path(sys.argv[1]).read_bytes();assert h(b)==sys.argv[2];"
                "m=json.loads(b);assert all(h(pathlib.Path(x['path']).read_bytes())==x['sha256'] for x in m['files']);"
                "c=pathlib.Path(sys.argv[3]).read_bytes();assert h(c)==sys.argv[4];"
                "os.execv('/bin/bash',['/bin/bash','-c',c.decode()])")
        command = 'exec ' + shlex.join(['python', '-B', '-c', code, str(MANIFEST), sha(MANIFEST), str(LAUNCHER), sha(LAUNCHER)])
        row = dict(run=RUN, method=s['method'], domain=s['domain'], condition=s['condition'], stage=s['stage'], kind='detector_train',
                   job_name=s['job_name'], workload='FIXED_K_NATIVE_DETECTOR_TRAIN_EVAL',
                   spec=str(SPEC), spec_sha256=sha(SPEC), queue_id=ROUTES[s['queue']][0], image=s['image'],
                   resources=s['resources'], K=s['K'], seed=42, command_sha256=hashlib.sha256(command.encode()).hexdigest(),
                   status='intent_recorded', created_at=now(),
                   submitter_source=str(Path(__file__).resolve()), submitter_source_sha256=sha(__file__))
        atomic(out, row, True)
        try:
            dag = load_dag()
            kw = dict(dag.SUBMIT_KW, resource_queue_id=ROUTES[s['queue']][0], image_url=s['image'], instance_type_id='', **s['resources'])
            kw.pop('holding_time_seconds', None)
            with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                response = dag.create_ml_job(command=command, job_name=s['job_name'], **kw)
            jid = getattr(response, 'id', None) or (response.get('id') if isinstance(response, dict) else None)
            row.update(id=jid, status='submitted' if jid else 'submit_result_uncertain', submitted_at=now())
        except Exception as e:
            row.update(status='submit_result_uncertain', error_type=type(e).__name__)
            atomic(out, row)
            raise RuntimeError('Create uncertain; reconcile exact job name before retry') from None
        atomic(out, row)
        assert row.get('id'), 'No returned jobID; reconcile'
    print(json.dumps({k: row[k] for k in ('stage', 'id', 'status', 'command_sha256')}))


if __name__ == '__main__':
    main()
