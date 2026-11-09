"""Synthetic V3 input gates and unchanged numerical consumers; no execution."""
import copy,unittest
from pathlib import Path
import chunk_receipt_contract as c
import protocol as p

def fixture():
 names=[f'SYNTHETIC_{i}.jpg' for i in range(12340)]
 return dict(schema=c.SCHEMA,status=c.STATUS,verification_version=3,method='MIGC',domain='dior',mode='ours',split='train',metadata=dict(sha256='a'*64),count=12340,raw=dict(count=12340,file_sha256={n:'b'*64 for n in names}),files={n:{} for n in names},old_parent_job_status='Failed',old_failed_producer_relabelled_Completed=False,all_old_encoded_file_SHA_rechecked_against_audited_rank=True,all8_original_whole_streams_from0=True,actual_prepared_latent_CRC_SHA_equal=True,all_five_RNG_before_after_consumed_currentCUDA_values_equal=True,world=8,logical_world=8,seed=42,steps=50,CFG=5,MIGCsteps=25,FG_loaded=True,FFT=True,streams=[dict(logical_rank=i) for i in range(8)])
class Tests(unittest.TestCase):
 def test_explicit_V3_full_positive(self):self.assertEqual(c.validate_core(fixture(),'MIGC','dior','ours','a'*64,12340),c.SCHEMA)
 def test_old_V1_or_weak_encoded_schema_rejected(self):
  for key,value in [('schema','MIGC_DIOR_MIXED_FAILED4_RECOVERED4_FULL_POOL_V1'),('status','PASS_FULL_TYPED_DIOR_MIXED_WHOLE_STREAM_POOL_AND_NATIVE_PAIR'),('verification_version',1),('all_old_encoded_file_SHA_rechecked_against_audited_rank',False)]:
   r=fixture();r[key]=value
   with self.assertRaises(AssertionError):c.validate_core(r,'MIGC','dior','ours','a'*64,12340)
 def test_partial_or_Completed_legacy_parent_rejected(self):
  for key,value in [('count',6169),('old_parent_job_status','Completed'),('old_failed_producer_relabelled_Completed',True),('all8_original_whole_streams_from0',False)]:
   r=fixture();r[key]=value
   with self.assertRaises(AssertionError):c.validate_core(r,'MIGC','dior','ours','a'*64,12340)
 def test_wrong_arm_domain_metadata_rejected(self):
  for key,value in [('domain','hrsc'),('mode','native_no_fg'),('split','val'),('seed',43)]:
   r=fixture();r[key]=value
   with self.assertRaises(AssertionError):c.validate_core(r,'MIGC','dior','ours','a'*64,12340)
  with self.assertRaises(AssertionError):c.validate_core(fixture(),'MIGC','dior','ours','c'*64,12340)
 def test_old_Main_not_accepted(self):
  self.assertEqual(c.MAIN_NAME,'MIGC_DIOR_MIXED_RECOVERY_FULL_POOL_ACCEPTANCE_V3.json');self.assertEqual(c.OWNER_STATUS,'OWNER_ACCEPTED_FULL_TYPED_DIOR_MIXED_RECOVERY_POOL_V3')
 def test_numeric_guard_loader_freeze_unchanged(self):
  old=p.HERE.parent/'downstream_dior_mixed_recovery_cpu_v1'
  for name in ('actual_preflight.py','cpu_entry.py','cpu_bridge.py','exactk_guard.py','freeze.py','verify_result.py','commands.py','admit.py'):self.assertEqual((p.HERE/name).read_bytes(),(old/name).read_bytes())
 def test_original_K_LR_schedule_and_seed(self):
  d=p.DOMAINS['dior'];self.assertEqual(d['K'],9264);self.assertEqual(d['lr'],.04);self.assertEqual(d['steps'],[6176,8492]);self.assertEqual(d['batch'],4)
 def test_independent_EFS_namespaces_and_only_ours(self):
  self.assertEqual(tuple(p.DOMAINS),('dior',));self.assertEqual(p.CONDITIONS,('ours',));self.assertIn('migc_dior_mixed_recovery_pool_pair_v3',str(c.MIXED))
  for kind,path in p.paths('MIGC','dior','ours').items():self.assertIn('/downstream_dior_mixed_recovery_cpu_v3/',str(path));self.assertTrue(path.resolve().is_relative_to(Path('/data/efs/datasets/xianbao01.hou')))
if __name__=='__main__':unittest.main()
