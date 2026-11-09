"""Publish one immutable preflight source/input spec after its own pool is ready."""
import argparse
from protocol import (HERE,DOMAINS,METHODS,CONDITIONS,paths,admission_path,verify_pool,verify_spec,bind,write_new,source_pins)


def main():
    p=argparse.ArgumentParser();p.add_argument('--method',choices=METHODS,required=True);p.add_argument('--domain',choices=DOMAINS,required=True)
    p.add_argument('--condition',choices=CONDITIONS,required=True);a=p.parse_args()
    verify_pool(a.method,a.domain,a.condition);own=paths(a.method,a.domain,a.condition);cfg=DOMAINS[a.domain]
    file=admission_path(a.method,a.domain,a.condition)
    if file.exists():verify_spec(a.method,a.domain,a.condition);print(file);return
    record=dict(status='IMMUTABLE_TARGET_PREFLIGHT_SPEC',method=a.method,domain=a.domain,condition=a.condition,
                execution_profile='CPU_GLOO8_STRUCTURE_ONLY',GPU_count=0,backend='gloo',
                dataset=str(own['dataset'].resolve()),combine=bind(own['dataset']/'COMBINE_COMPLETE.json'),
                K=cfg['K'],lr_steps=cfg['steps'],seed=42,world=8,target_image=cfg['image'],
                source_pins=source_pins())
    write_new(file,record);print(file)


if __name__=='__main__':main()
