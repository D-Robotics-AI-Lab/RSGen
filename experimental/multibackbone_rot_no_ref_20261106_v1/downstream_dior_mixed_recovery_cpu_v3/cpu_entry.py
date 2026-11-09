"""CPU-only combine entry. Does not alter later freeze/train process contexts."""
import argparse
import sys


def main():
    parser=argparse.ArgumentParser();parser.add_argument('phase',choices=['combine','torchrun'])
    args,remaining=parser.parse_known_args()
    from cpu_bridge import install_no_cuda_guard
    guard=install_no_cuda_guard()
    import torch
    torch.set_num_threads(1)
    try:
        if args.phase=='combine':
            import prepare_pool
            sys.argv=[prepare_pool.__file__]+remaining
            prepare_pool.main()
        else:
            # Guard the torchrun agent too. Its children enter actual_preflight.py
            # and install their own execution-local guard before native imports.
            from torch.distributed import run
            sys.argv=['torch.distributed.run']+remaining
            run.main()
    finally:
        guard.check()
    print('CPU_EXECUTION_NO_CUDA_PASS',args.phase)


if __name__=='__main__':main()
