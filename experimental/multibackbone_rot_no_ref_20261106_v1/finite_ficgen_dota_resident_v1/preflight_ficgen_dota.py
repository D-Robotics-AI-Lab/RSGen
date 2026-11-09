"""CPU source/spec/storage preparation only for FICGen DOTA smoke/formal resident dispatch.

Routine same-setting adaptation of the accepted FICGen HRSC resident preflight
(finite_ficgen_hrsc_resident_v1/preflight_ficgen_hrsc.py, commit e549c07) to the
same sealed ficgen_sealed_v4 adapter with the sealed DOTA specs. All numerics
stay bound to the frozen sealed specs/COMMANDS; this file adds no math, imports
no models, publishes nothing. New source lives in a linked worktree; every
ROOT/MON/EFS reference is an explicit absolute path into the original project.
"""
import argparse, hashlib, json, os, sys, tempfile
from pathlib import Path

ROOT = Path('/data/vepfs/users/xianbao01.hou/RSGen')
RUN = 'multibackbone_rot_no_ref_20261106_v1'
SEALED = ROOT / 'experimental' / RUN / 'adapters/ficgen_sealed_v4'
MON = ROOT / '.train_monitor_multibackbone_rot_no_ref_20261106'
HERE = MON / 'runtime_preparation/ficgen_dota_smoke_formal_resident_b001'
BUNDLE_SHA = '34e7032e42997b299068c368240122baf434b4a14b84a032d3d56cd48e6913ed'
SEALED_MANIFEST_SHA = '6018ae0895d4f70ce945e63b93c887afd0e44252543223eb865e82034aff92e6'
SPEC = {
    'smoke': (SEALED / 'dota_smoke_a002_SPEC.json', 'ef181a759d7c59eb3aa0a402c803717914856fdf773ebbfea27da64b180c89b7'),
    'formal': (SEALED / 'dota_formal_a002_SPEC.json', '07006d47132a53e5821cf69f5846877846915fb158c58b117833fbf9a22aafa5'),
}


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def validate(phase, runtime=False):
    assert __debug__
    assert sha(SEALED / 'SOURCE_BUNDLE.json') == BUNDLE_SHA
    command = json.loads((HERE / ('SMOKE_COMMAND.json' if phase == 'smoke' else 'FORMAL_COMMAND.json')).read_text())
    specpath, specsha = SPEC[phase]
    spec = json.loads(specpath.read_text())
    assert sha(specpath) == command['spec_sha256'] == specsha
    assert sha(SEALED / 'SMOKE_SOURCE_MANIFEST.json') == command['source_manifest_sha256'] == SEALED_MANIFEST_SHA
    manifest = json.loads((SEALED / 'SMOKE_SOURCE_MANIFEST.json').read_text())
    for entry in manifest['files']:
        assert sha(entry['path']) == entry['sha256']
    sys.path.insert(0, str(SEALED))
    from execution_spec import validate_spec, training_argv
    assert Path(sys.modules['execution_spec'].__file__).resolve() == SEALED / 'execution_spec.py'
    validate_spec(spec)
    assert spec['mode'] == phase and spec['domain'] == 'dota' and spec['steps'] == (4 if phase == 'smoke' else 60000) and spec['formal_endpoint'] == 60000
    assert spec['world_size'] == 8 and spec['seed'] == 1234 and spec['learning_rate'] == 8e-5 and spec['mixed_precision'] == 'no'
    assert spec['initializer'] == 'init_original' and spec['optimizer'] == 'new_AdamW' and spec['old_fg_loaded'] is False
    argv = training_argv(spec)
    assert '--resume_from_checkpoint' not in argv and '--controlnet_weights_path' not in argv
    log = Path(command['launcher_output'])
    assert log.resolve().is_relative_to(Path('/data/efs/datasets/xianbao01.hou/evaluations/RSGen') / RUN)
    assert not log.exists(), 'Fresh launcher attempt required'
    if runtime:
        log.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(prefix='ficgen_dota_launcher_private_probe_', dir=log.parent) as stream:
            stream.write(b'CPU runtime storage probe')
            stream.flush()
            os.fsync(stream.fileno())
            stream.seek(0)
            assert stream.read() == b'CPU runtime storage probe'
    assert not any(m in sys.modules for m in ('torch', 'numpy', 'cv2', 'diffusers'))
    return dict(status='PASS_CPU_PREPARED_ONLY_NOT_SMOKE_ACCEPTANCE', domain='dota', phase=phase, updates=spec['steps'], formal_endpoint=60000, output=spec['output_dir'], launcher_output=str(log), model_imported=False, GPU_used=False, published=False)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--phase', choices=['smoke', 'formal'], required=True)
    p.add_argument('--runtime', action='store_true')
    a = p.parse_args()
    print(json.dumps(validate(a.phase, a.runtime)))
