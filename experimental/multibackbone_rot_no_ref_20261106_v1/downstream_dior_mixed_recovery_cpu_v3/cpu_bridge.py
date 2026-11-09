"""Process-local CPU/Gloo guards and seed-device bridge; no import side effects.

Never import/install this bridge from detector configs or training commands.
The native sampler and its index/padding/shuffle code remain unchanged.
"""
import ast
import functools
import hashlib
import inspect
import os
import textwrap
from pathlib import Path


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class NoCUDA:
    def __init__(self):
        assert os.environ.get('CUDA_VISIBLE_DEVICES') == '', 'CPU profile requires an explicit empty device mask'
        import torch
        self.torch = torch
        self.attempts = []
        self.availability_queries = []
        self.originals = []
        assert not torch.cuda.is_initialized(), 'CUDA initialized before CPU guard'
        def blocked(name):
            def reject(*args, **kwargs):
                self.attempts.append(name)
                raise RuntimeError('CPU_GLOO8 forbids CUDA use: ' + name)
            return reject
        for owner, name in [(torch.cuda,'_lazy_init'),(torch.cuda,'init'),(torch.cuda,'set_device'),
                            (torch.cuda,'current_device'),(torch.cuda,'get_device_properties'),
                            (torch.Tensor,'cuda'),(torch.nn.Module,'cuda')]:
            if hasattr(owner,name):
                self.originals.append((owner,name,getattr(owner,name)))
                setattr(owner,name,blocked(name))
        if hasattr(torch._C,'_cuda_init'):
            self.originals.append((torch._C,'_cuda_init',torch._C._cuda_init))
            torch._C._cuda_init=blocked('_C._cuda_init')
        # Even CUDA availability queries can call cudaGetDeviceCount/cuInit.
        # This declared0GPU profile reports its known empty visibility without
        # invoking the driver. No data/sampler arithmetic uses these queries.
        def query(name,value):
            def cpu_answer(*args,**kwargs):
                self.availability_queries.append(name)
                return value
            return cpu_answer
        for owner,name,value in [(torch.cuda,'is_available',False),(torch.cuda,'device_count',0),
                                 (torch._C,'_cuda_getDeviceCount',0)]:
            if hasattr(owner,name):
                self.originals.append((owner,name,getattr(owner,name)))
                setattr(owner,name,query(name,value))
        self.check()

    def check(self):
        assert os.environ.get('CUDA_VISIBLE_DEVICES') == ''
        assert not self.attempts, self.attempts
        assert not self.torch.cuda.is_initialized()

    def receipt(self):
        self.check()
        return dict(visible_devices='',initialized=False,forbidden_attempts=[],
                    availability_query_policy='declared0GPU answersFalse/0 without CUDA driver query',
                    availability_queries=list(self.availability_queries),
                    guard_source=dict(path=str(Path(__file__).resolve()),sha256=_sha(__file__)),
                    scope='only this CPU execution process; no native file mutations')


def install_no_cuda_guard():
    """Explicit execution call only. Merely importing this module changes nothing."""
    return NoCUDA()


class SeedBridge:
    def __init__(self,build_dataloader):
        import torch.distributed as dist
        assert dist.is_initialized() and dist.get_backend()=='gloo' and dist.get_world_size()==8
        self.calls=[];self.sources=[];self.patches=[]
        namespace=build_dataloader.__globals__
        sampler=namespace['DistributedGroupSampler']
        assert sampler.__name__=='DistributedGroupSampler'
        self.sampler=sampler
        self.sampler_source=dict(path=inspect.getsourcefile(sampler),sha256=_sha(inspect.getsourcefile(sampler)))
        # The common old native sampler stores explicit42 directly. Some target
        # versions import sync_random_seed in its constructor instead. Inspect
        # actual function globals, not a host installation's path or version.
        scopes=[namespace,sampler.__init__.__globals__]
        seen=set()
        for scope in scopes:
            if id(scope) in seen:continue
            seen.add(id(scope))
            for key,original in list(scope.items()):
                if callable(original) and getattr(original,'__name__',None)=='sync_random_seed':
                    self.patch_scope(scope,key,original)

    def patch_scope(self,scope,key,original):
        signature=inspect.signature(original)
        assert 'seed' in signature.parameters and 'device' in signature.parameters, 'Unsupported native seed-sync API'
        path=inspect.getsourcefile(original)
        source=inspect.getsource(original)
        self.sources.append(dict(path=path,sha256=_sha(path),function=original.__name__,
                                 function_AST_sha256=hashlib.sha256(ast.dump(ast.parse(textwrap.dedent(source)),include_attributes=False).encode()).hexdigest(),
                                 native_default_device=str(signature.parameters['device'].default)))
        @functools.wraps(original)
        def cpu_seed(seed=None,device='cuda'):
            assert seed==42 and isinstance(seed,int),'Only explicit frozen seed42 is allowed'
            # Execute the actual original function, changing only its explicit
            # tensor device. Its int32 rank0 broadcast/return values are intact.
            result=original(seed=seed,device='cpu')
            assert result==42 and isinstance(result,int)
            self.calls.append(dict(input_seed=42,native_requested_device=str(device),
                                   actual_device='cpu',returned_seed=42))
            return result
        scope[key]=cpu_seed
        self.patches.append((scope,key,original))

    def receipt(self):
        assert all(v['returned_seed']==42 and v['actual_device']=='cpu' for v in self.calls)
        return dict(kind='EXPLICIT_CPU_DEVICE_FOR_NATIVE_SYNC_SEED42',
                    patched_aliases=len(self.patches),actual_native_sync_calls=len(self.calls),calls=self.calls,
                    native_sync_sources=self.sources,native_sampler_source=self.sampler_source,
                    source=dict(path=str(Path(__file__).resolve()),sha256=_sha(__file__)),
                    unchanged=['native sampler class','native seed42','native helper broadcast dtype/payload/value',
                               'group flags','group padding','epoch-dependent shuffle','samples_per_rank','world8'],
                    conditional_note='zero calls means selected native sampler stores explicit42 directly; no CUDA sync was invented',
                    mutation_scope='in-memory alias only, CPU worker lifetime; no file/installation changes')


def install_seed_bridge(build_dataloader):
    return SeedBridge(build_dataloader)
