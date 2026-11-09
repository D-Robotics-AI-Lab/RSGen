"""Explicit strengthened mixed DIOR V3 only; old V1/V2 receipts/Main rejected."""
import importlib.util
from pathlib import Path
HERE=Path(__file__).resolve().parent
MIXED=HERE.parent/'adapters/migc_dior_mixed_recovery_pool_pair_v3'
SCHEMA='MIGC_DIOR_MIXED_FAILED4_RECOVERED4_FULL_POOL_V3'
STATUS='PASS_FULL_TYPED_DIOR_MIXED_WHOLE_STREAM_POOL_AND_NATIVE_PAIR_V3'
OWNER_STATUS='OWNER_ACCEPTED_FULL_TYPED_DIOR_MIXED_RECOVERY_POOL_V3'
MAIN_NAME='MIGC_DIOR_MIXED_RECOVERY_FULL_POOL_ACCEPTANCE_V3.json'

def validate_core(receipt,method,domain,condition,metadata_sha256,count):
 assert method=='MIGC' and domain=='dior' and condition=='ours' and count==12340
 assert receipt['schema']==SCHEMA and receipt['status']==STATUS and receipt['verification_version']==3
 assert receipt['method']=='MIGC' and receipt['domain']=='dior' and receipt['mode']=='ours' and receipt['split']=='train'
 assert receipt['metadata']['sha256']==metadata_sha256 and receipt['count']==receipt['raw']['count']==len(receipt['files'])==len(receipt['raw']['file_sha256'])==12340
 assert set(receipt['files'])==set(receipt['raw']['file_sha256'])
 assert receipt['old_parent_job_status']=='Failed' and receipt['old_failed_producer_relabelled_Completed'] is False
 assert receipt['all_old_encoded_file_SHA_rechecked_against_audited_rank'] is True
 assert receipt['all8_original_whole_streams_from0'] is True and receipt['actual_prepared_latent_CRC_SHA_equal'] is True and receipt['all_five_RNG_before_after_consumed_currentCUDA_values_equal'] is True
 assert receipt['world']==receipt['logical_world']==8 and receipt['seed']==42 and receipt['steps']==50 and receipt['CFG']==5 and receipt['MIGCsteps']==25 and receipt['FG_loaded'] is True and receipt['FFT'] is True
 assert {s['logical_rank'] for s in receipt['streams']}==set(range(8)) and len(receipt['streams'])==8
 return SCHEMA

def validate_ready(receipt,method,domain,condition,metadata_sha256,count):
 validate_core(receipt,method,domain,condition,metadata_sha256,count)
 q=importlib.util.spec_from_file_location('dior_mixed_v3_contract',MIXED/'contract.py');c=importlib.util.module_from_spec(q);q.loader.exec_module(c)
 main=c.read(c.MON/MAIN_NAME)
 assert main['status']==OWNER_STATUS and main['owner']=='/root/exactk' and main['actual_execution_proxy']=='/root/execution_fresh'
 assert c.bound(main['generation'])==receipt and main['count']==12340
 assert receipt['collector']==c.pin(MIXED/'collect_pair.py') and receipt['contract']==c.pin(MIXED/'contract.py')
 plan=c.bound(receipt['collection_plan']);assert plan['reference']==c.pin(c.REF)
 assert len(receipt['producer_bindings'])==5
 for binding in receipt['producer_bindings']:c.bound(binding)
 assert receipt['producer_bindings'][0]==c.reference()['verified_old_complete_streams']
 new_ranks=set()
 for entry in receipt['producer_bindings'][1:]:
  accepted=c.bound(entry)
  assert accepted['schema']=='MIGC_DIOR_RECOVERY_STREAM_CPU_ACCEPTANCE_V3' and accepted['status']=='PASS_CPU_OWN_NEW_DIOR_RECOVERY_COMPLETE_STREAM_V3' and accepted['validation_version']==3
  assert accepted['verifier']==c.pin(MIXED/'accept_recovery_stream.py')
  assert accepted['prefix_manifest_reference_rows_exact'] is True and accepted['actual_H_path_original_edge_and_audit_same'] is True and accepted['actual_outer_runtime_and_command_descriptor_verified'] is True
  assert accepted['logical_rank'] in c.NEW and accepted['logical_rank'] not in new_ranks;new_ranks.add(accepted['logical_rank'])
  owner=c.bound(accepted['terminal_owner']);assert owner['job_state']=='Completed' and owner['all_instances_terminal'] is True and owner['all_worker_processes_absent'] is True
 assert new_ranks==c.NEW
 return SCHEMA
