"""Read-only sampling/LR guard and exact-K final checkpoint save after_run."""
import json
import math
from pathlib import Path

from mmcv.runner import HOOKS, Hook, get_dist_info


@HOOKS.register_module()
class FourCellExactKGuard(Hook):
    def __init__(self, K, loader_length, lr_steps, expected_lr, evidence):
        self.K, self.length = K, loader_length
        self.steps, self.lr, self.evidence = lr_steps, expected_lr, Path(evidence)
        self.epochs = set()

    def before_run(self, runner):
        assert get_dist_info()[1] == 8 and runner.iter == 0 and runner.max_iters == self.K
        self.base = [group['lr'] for group in runner.optimizer.param_groups]
        assert all(math.isclose(value, self.lr, rel_tol=1e-9) for value in self.base), self.base
        self.hook = next(h for h in runner._hooks if type(h).__name__ == 'StepLrUpdaterHook')
        assert not self.hook.by_epoch and list(self.hook.step) == self.steps and self.hook.warmup_iters == 500
        if get_dist_info()[0] == 0:
            self.evidence.mkdir(parents=True, exist_ok=True)
            assert not (self.evidence / 'runtime.jsonl').exists()

    def before_train_iter(self, runner):
        loader = runner.data_loader._dataloader
        sampler = loader.sampler
        assert type(sampler).__name__ == 'DistributedGroupSampler' and len(loader) == self.length
        epoch = runner.iter // self.length
        assert sampler.epoch == runner.data_loader.epoch == epoch
        factor = self.hook.gamma ** sum(runner.iter >= step for step in self.steps)
        if runner.iter < 500:
            assert self.hook.warmup == 'linear'
            factor *= self.hook.warmup_ratio + (1 - self.hook.warmup_ratio) * runner.iter / 500
        assert all(math.isclose(group['lr'], value * factor, rel_tol=1e-8)
                   for value, group in zip(self.base, runner.optimizer.param_groups))
        points = {0, 499, 500, self.K - 1, *self.steps, *(step - 1 for step in self.steps)}
        if get_dist_info()[0] == 0 and (epoch not in self.epochs or runner.iter in points):
            row = dict(iter=runner.iter, sampler=type(sampler).__name__, loader_length=self.length,
                       sampler_epoch=epoch, lr=[group['lr'] for group in runner.optimizer.param_groups])
            with (self.evidence / 'runtime.jsonl').open('a') as stream: stream.write(json.dumps(row) + '\n')
        self.epochs.add(epoch)

    def after_run(self, runner):
        assert runner.iter == self.K
        if get_dist_info()[0] == 0:
            # Stock after_train_iter saves meta.iter K-1; save after_run at K.
            runner.save_checkpoint(runner.work_dir, filename_tmpl='exactk_final.pth')
            with (self.evidence / 'train_complete.json').open('x') as stream:
                json.dump(dict(iter=runner.iter, expected_iters=self.K,
                               sampler_epochs=sorted(self.epochs)), stream, indent=2)
