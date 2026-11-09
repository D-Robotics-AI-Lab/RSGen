"""Main-only fixed-K8L4 ours-mixedV3 DIOR detector train/eval from local-CPU-chain frozen contract."""
import argparse,contextlib,fcntl,hashlib,io,json,os,shlex,sys
from pathlib import Path
from datetime import datetime,timezone
ROOT=Path(__file__).resolve().parents[2];RUN='multibackbone_rot_no_ref_20261106_v1';MON=ROOT/'.train_monitor_multibackbone_rot_no_ref_20261106';REG=MON/'finite_detector_submissions'
CHAIN=ROOT/'experimental'/RUN/'downstream_dior_mixed_recovery_cpu_v3'
ROUTES={'jrwjn':('q-20260429225420-jrwjn','ml.gni3cl.45xlarge'),'rsrdr':('q-20260810131417-rsrdr','ml.gni3')}
PROFILE_SHA='db4d7c5e7c7d1ffd54af48a0275bfb1660b48392a6d5977171d2afba25f96ef1'
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
 return h.hexdigest()
def read(p):return json.loads(Path(p).read_text())
def now():return datetime.now(timezone.utc).isoformat()
def atomic(p,j,exclusive=False):
 if exclusive:
  with p.open('x') as f:json.dump(j,f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
 else:
  t=p.with_name(p.name+'.preparing')
  with t.open('x') as f:json.dump(j,f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
  os.replace(t,p)
def validate(s):
 assert s['run']==RUN and s['method']=='MIGC' and s['domain']=='dior' and s['condition']=='ours' and s['stage']=='migc_dior_ours_mixed_detector_train_a001'
 assert s['queue'] in ROUTES and s['finite'] and s['holding_time_seconds'] is None and s['K']==9264 and s['lr_steps']==[6176,8492] and s['seed']==42
 assert s['world']==8 and s['batch_per_device']==4 and s['native_eval_images']==3372
 assert s['resources']==dict(gpu_count=8,cpu=40,memory_gib=128,gpu_type='NVIDIA-L4',family=ROUTES[s['queue']][1],replicas=1)
 p=s['resource_plan'];assert sha(p['path'])==p['sha256'];r=read(p['path'])
 assert r['main_admitted'] and s['queue'] in r['admitted_queues'] and datetime.now(timezone.utc)<datetime.fromisoformat(r['valid_until'])
 prof=s['training_resource_profile'];assert sha(prof['path'])==prof['sha256']==PROFILE_SHA
 for k in ['target_freeze_acceptance','runtime_manifest','image_receipt']:
  x=s[k];assert sha(x['path'])==x['sha256'],(k,'drift')
 a=read(s['target_freeze_acceptance']['path'])
 assert a['status']=='MAIN_ACCEPTED_TARGET_FROZEN_DETECTOR_CONFIG_NOT_TRAINED'
 assert a['method']=='MIGC' and a['domain']=='dior' and a['condition']=='ours' and a['K']==9264 and a['seed']==42 and a['actual_GPU_training_verified'] is False
 for key in ['config','frozen_config','actual_cpu_gloo8','combine','GPU_script','R50']:
  assert sha(a[key]['path'])==a[key]['sha256'],('acceptance pin',key)
 pool=read(a['training_pool']['full_pool_v3']['path']);assert pool['schema']=='MIGC_DIOR_MIXED_FAILED4_RECOVERED4_FULL_POOL_V3'
 assert read(s['image_receipt']['path'])['image_url']==s['image']=='d-robotics-image-dev-cn-beijing.cr.volces.com/d-robotics-images/ccdiff:mmrotate_l4'
 m=read(s['runtime_manifest']['path']);assert m['files']
 for x in m['files']:assert sha(x['path'])==x['sha256'],('Sealed detector source drift',x['path'])
 assert any(x['path']==s['launcher'] for x in m['files'])
 # Frozen-config integrity + official GPU script equality rendered from the chain printer
 sys.path.insert(0,str(CHAIN));import commands
 rec=commands.frozen('MIGC','dior','ours')
 assert rec['config']['path']==a['config']['path'] and sha(rec['config']['path'])==a['config']['sha256']
 rendered=commands.command('train-eval','MIGC','dior','ours',None,None,1)
 assert Path(a['GPU_script']['path']).read_text()==rendered+'\n','GPU script differs from chain printer'
 assert 'cpu_entry.py' not in rendered and 'CUDA_VISIBLE_DEVICES=""' not in rendered
 for q in [Path(s['work_dir'])/'exactk_final.pth',Path(s['evidence'])/'train_stdout.log',Path(s['launcher_output'])]:assert not q.exists(),('New detector attempt required',str(q))
 return s
def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--spec',type=Path,required=True);p.add_argument('--submit',action='store_true');a=p.parse_args();s=validate(read(a.spec))
 if not a.submit:print(json.dumps(dict(status='DETECTOR_SPEC_VALIDATED_NOT_SUBMITTED',K=s['K'],resources=s['resources'])));return
 REG.mkdir(exist_ok=True)
 with (REG/'.dispatch.lock').open('a') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX);out=REG/(s['stage']+'.json');assert not out.exists(),'Prior intent: reconcile exactJob before retry'
  for q in REG.glob('*.json'):
   old=read(q)
   if (old.get('run'),old.get('method'),old.get('domain'),old.get('condition'))==(s['run'],s['method'],s['domain'],s['condition']):
    assert old.get('confirmed_terminal') is True,'Prior same run/method/domain/condition producer unresolved'
  code="import hashlib,json,pathlib,os,sys;h=lambda b:hashlib.sha256(b).hexdigest();b=pathlib.Path(sys.argv[1]).read_bytes();assert h(b)==sys.argv[2];m=json.loads(b);assert all(h(pathlib.Path(x['path']).read_bytes())==x['sha256'] for x in m['files']);c=pathlib.Path(sys.argv[3]).read_bytes();assert h(c)==sys.argv[4];os.execv('/bin/bash',['/bin/bash','-c',c.decode()])"
  command='exec '+shlex.join(['python','-B','-c',code,s['runtime_manifest']['path'],s['runtime_manifest']['sha256'],s['launcher'],sha(s['launcher'])])
  row=dict(run=RUN,method=s['method'],domain=s['domain'],condition=s['condition'],stage=s['stage'],job_name=s['job_name'],workload='FIXED_K_OURS_MIXED_DETECTOR_TRAIN_EVAL',spec=str(a.spec.resolve()),spec_sha256=sha(a.spec),queue_id=ROUTES[s['queue']][0],image=s['image'],resources=s['resources'],K=s['K'],seed=42,command_sha256=hashlib.sha256(command.encode()).hexdigest(),status='intent_recorded',created_at=now())
  atomic(out,row,True)
  try:
   sys.path.insert(0,str(ROOT));from dag_train_gen import create_ml_job,SUBMIT_KW
   kw=dict(SUBMIT_KW,resource_queue_id=ROUTES[s['queue']][0],image_url=s['image'],instance_type_id='',**s['resources']);kw.pop('holding_time_seconds',None)
   with contextlib.redirect_stdout(io.StringIO()),contextlib.redirect_stderr(io.StringIO()):response=create_ml_job(command=command,job_name=s['job_name'],**kw)
   jid=getattr(response,'id',None) or (response.get('id') if isinstance(response,dict) else None);row.update(id=jid,status='submitted' if jid else 'submit_result_uncertain',submitted_at=now())
  except Exception as e:
   row.update(status='submit_result_uncertain',error_type=type(e).__name__);atomic(out,row);raise RuntimeError('Create uncertain; reconcile exact job name before retry') from None
  atomic(out,row);assert row.get('id'),'No returned jobID; reconcile'
 print(json.dumps(row,indent=2))
if __name__=='__main__':main()
