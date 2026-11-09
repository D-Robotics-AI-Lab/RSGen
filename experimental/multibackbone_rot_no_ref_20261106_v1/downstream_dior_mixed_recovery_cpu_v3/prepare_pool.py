"""Accept one full producer and build its own immutable1:1 EFS pool.

HRSC scaling calls the original scale_hrsc_xml with mmrotate; run that phase
in the existing mmrotate image (CPU work). DIOR/DOTA use unchanged CCDiff GT
construction functions, without importing any CCDiff latent/noise contract.
"""
import argparse
import json
from pathlib import Path
import sys
import xml.etree.ElementTree as ET
from protocol import (ROOT,HERE,HRSC,DOMAINS,METHODS,CONDITIONS,paths,shared_protocol,
                      sha,bind,read_bound,write_new,load_old_numerics,OLD_ASYNC)


def acceptance(path,method,domain,condition):
    obj=json.loads(Path(path).read_text());data,train=shared_protocol(domain)
    from chunk_receipt_contract import validate_ready
    input_kind=validate_ready(obj,method,domain,condition,sha(data['train_metadata']),data['expected_active']['train'])
    assert obj['method'].upper()==method.upper()
    assert obj['domain']==domain and obj['split']=='train'
    expected_condition='native_noFG' if condition=='native_noFG' else 'scratchFG_rotation_rerender_no_reference_H_FFT_on'
    assert obj['condition']==expected_condition
    assert bool(obj['FG_loaded'])==(condition=='ours') and bool(obj['FFT'])==(condition=='ours')
    if condition=='ours':assert obj['actual_all_rank_pixel_and_latent_records_verified'] is True
    assert obj['raw']['count']==data['expected_active']['train']
    metadata_sources=[]
    for receipt in obj['producer_bindings']:
        assert sha(receipt['path'])==receipt['sha256'],'Producer source/receipt changed'
        if Path(receipt['path']).suffix=='.json':
            source=json.loads(Path(receipt['path']).read_text())
            if 'metadata_sha256' in source:metadata_sources.append(source['metadata_sha256'])
    metadata_sources.append(obj['metadata']['sha256'])
    assert sha(data['train_metadata']) in metadata_sources,'Producer metadata does not match audited downstream train split'
    raw=Path(obj['raw']['path']).resolve()
    assert raw.is_relative_to(Path('/data/efs/'))
    rows=[json.loads(line) for line in Path(data['train_metadata']).read_text().splitlines() if line.strip()]
    rows={Path(row['file_name']).stem:row for row in rows if any(p and any(b) for p,b in zip(row['caption'][1:],row['bndboxes']))}
    files={Path(name).stem:raw/name for name in obj['raw']['file_sha256']}
    assert len(files)==len(rows)==data['expected_active']['train']
    assert set(files)==set(rows) and {p.name for p in raw.iterdir() if p.suffix.lower() in ['.bmp','.png','.jpg','.jpeg']}==set(obj['raw']['file_sha256'])
    # Typed input has preexisting full own-pixel acceptance. Preserve old audit;
    # no second legacy decode/hash pass here. Verify exact symlink consumption.
    for file in files.values():
        original=obj['files'][file.name]['actual_saved_image_path']
        assert file.resolve()==Path(original).resolve() and file.is_file()
    return obj,data,train,rows,files


def hrsc(rows,files,output):
    from PIL import Image
    sys.path.insert(0,str(ROOT))
    from combine_train_gen import scale_hrsc_xml
    for name in ['AllImages','Annotations','ImageSets']:(output/name).mkdir()
    official=(HRSC/'ImageSets/trainval.txt').read_text().split()
    assert len(official)==617 and set(official)==set(rows)
    ids=[];converted=0
    for stem in official:
        real=HRSC/'FullDataSet/AllImages'/f'{stem}.bmp'
        xml=HRSC/'FullDataSet/Annotations'/f'{stem}.xml'
        assert real.is_file() and xml.is_file()
        objects=ET.parse(xml).getroot().findall('HRSC_Objects/HRSC_Object')
        consumed=[(p,b) for p,b in zip(rows[stem]['caption'][1:],rows[stem]['obboxes']) if p and any(b)]
        assert len(consumed)==len(objects),'Cannot copy fullXML onto condition-truncated synthetic scene'
        (output/'AllImages'/real.name).symlink_to(real.resolve())
        (output/'Annotations'/xml.name).symlink_to(xml.resolve())
        synthetic=stem+'_1';source=files[stem];destination=output/'AllImages'/f'{synthetic}.bmp'
        if source.suffix.lower()=='.bmp':destination.symlink_to(source.resolve())
        else:
            # Native HRSC loader hard-codes.bmp. Conversion must be lossless RGB.
            with Image.open(source) as image:rgb=image.convert('RGB');pixels=rgb.tobytes();rgb.save(destination)
            with Image.open(destination) as image:assert image.convert('RGB').tobytes()==pixels
            converted+=1
        scale_hrsc_xml(str(xml),str(output/'Annotations'/f'{synthetic}.xml'))
        tree=ET.parse(output/'Annotations'/f'{synthetic}.xml').getroot()
        assert tree.findtext('Img_SizeWidth')==tree.findtext('Img_SizeHeight')=='512'
        ids.extend([stem,synthetic])
    (output/'ImageSets/trainval.txt').write_text('\n'.join(ids)+'\n')
    assert len(ids)==1234
    return dict(real_images=617,synthetic_images=617,total_images=1234,lossless_BMP_conversions=converted,
                synthetic_XML_scaling='unchanged combine_train_gen.scale_hrsc_xml/scale_obb',
                scale_source=bind(ROOT/'combine_train_gen.py'),OBB_clipping=False)


def maybe_reuse_canonical(domain,output,data):
    if domain=='hrsc':return dict(reused=False,reason='Own scaledHRSC XML from original converter')
    old_state=Path('/data/efs/datasets/xianbao01.hou/evaluations/RSGen/fg_rotation_rerender_60k_2x2_20261104_v2/downstream_async_v1')/domain
    old=json.loads((old_state/'CANONICAL_DATA.json').read_text())
    assert old['metadata']['sha256']==sha(data['train_metadata']),'Current metadata differs from old canonicalGT contract'
    oracle=json.loads((old_state/'ANNOTATION_ORACLE.json').read_text())
    legacy=load_old_numerics();native=legacy.DOMAINS[domain]
    assert old['categories']==list(native['names']) and old['synthetic_size']==[512,512]
    from PIL import Image
    for record in old['real_images'].values():
        path=Path(record['path']);assert str(path.resolve())==record['realpath']
        with Image.open(path) as image:assert list(image.size)==[record['width'],record['height']]
    oldpath=Path(oracle['realpath']);own=output/('train.json' if domain=='dior' else 'labelTxt')
    expected=oracle['snapshot']['files']
    if domain=='dior':matches=sha(own)==sha(oldpath)==expected['train.json']
    else:matches={p.name:sha(p) for p in own.iterdir()}==expected=={p.name:sha(p) for p in oldpath.iterdir()}
    if not matches:return dict(reused=False,reason='New synthetic filename/ext/order/GT bytes differ; retain own annotations',old_annotation_receipt=bind(old_state/'ANNOTATION_ORACLE.json'))
    # Only remove this new unconsumed small duplicate after exact byte equality.
    if domain=='dior':own.unlink()
    else:
        for p in own.iterdir():p.unlink()
        own.rmdir()
    own.symlink_to(oldpath.resolve(),target_is_directory=domain=='dota')
    return dict(reused=True,path=str(oldpath),annotation_receipt=bind(old_state/'ANNOTATION_ORACLE.json'),
                metadata_identity=True,full_GT_byte_identity=True,old_noise_oracle_used=False)


def main():
    p=argparse.ArgumentParser();p.add_argument('--method',choices=METHODS,required=True);p.add_argument('--domain',choices=DOMAINS,required=True)
    p.add_argument('--condition',choices=CONDITIONS,required=True);p.add_argument('--generation',type=Path,required=True)
    p.add_argument('--producer-pairing',type=Path,help='Optional method-specific actual latent pairing acceptance; never a CCDiff oracle')
    a=p.parse_args();obj,data,train,rows,files=acceptance(a.generation,a.method,a.domain,a.condition)
    own=paths(a.method,a.domain,a.condition);output=own['dataset'];assert not output.exists(),'New attempt required; no overwrite'
    assert output.resolve().is_relative_to(Path('/data/efs/datasets/xianbao01.hou/datasets/RSGen'))
    output.mkdir(parents=True)
    import os,tempfile
    fd,probe=tempfile.mkstemp(prefix='downstream_write_fsync_read_',dir=output)
    try:
        with os.fdopen(fd,'wb') as stream:stream.write(b'RSGen_native_downstream_storage_probe');stream.flush();os.fsync(stream.fileno())
        assert Path(probe).read_bytes()==b'RSGen_native_downstream_storage_probe'
    finally:Path(probe).unlink(missing_ok=True)
    if a.domain=='hrsc':counts=hrsc(rows,files,output)
    else:
        legacy=load_old_numerics();old_rows=legacy.metadata(a.domain)
        assert old_rows==rows,'Current consumed metadata differs from old numericalGT implementation'
        counts=getattr(legacy,'combine_'+a.domain)(a.condition,Path(obj['raw']['path']),rows,output)
    gt=maybe_reuse_canonical(a.domain,output,data)
    result=dict(status='PASS_COMPLETE_1TO1_POOL',method=a.method,domain=a.domain,condition=a.condition,
                generation_receipt_status=obj['status'],generation_protocol=obj.get('generation_protocol','EXISTING_OURS_V2'),
                generation=bind(a.generation),producer_bindings=obj['producer_bindings']+[bind(ROOT/'.train_monitor_multibackbone_rot_no_ref_20261106/MIGC_DIOR_MIXED_RECOVERY_FULL_POOL_ACCEPTANCE_V3.json')],
                pairing=bind(a.producer_pairing) if a.producer_pairing else None,
                pairing_status='producer_receipt_supplied' if a.producer_pairing else 'separate_condition_ready_pair_comparison_pending',
                metadata=bind(data['train_metadata']),counts=counts,canonical_GT=gt,
                image_staging='EFS; real and compatible synthetic images symlinked; HRSC nonBMP losslessly converted',
                storage_probe=dict(status='PASS_WRITE_FSYNC_READ',realpath=str(output.resolve())),
                numerical_sources=[bind(ROOT/'combine_train_gen.py')] if a.domain=='hrsc' else
                    [bind(load_old_numerics().__file__),bind(load_old_numerics().DOMAINS[a.domain]['config'])],
                old_CCDiff_noise_oracle_used=False,actual_target_loader_verified=False)
    write_new(output/'COMBINE_COMPLETE.json',result)
    print('COMPLETE_1TO1_POOL',output)


if __name__=='__main__':main()
