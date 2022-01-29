from torch.optim.lr_scheduler import _LRScheduler

class LinearWarmupLR(_LRScheduler):
    def __init__(self, optimizer, warmup_steps, total_steps, lr, end_lr):
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.lr = lr
        self.end_lr = end_lr
        super(LinearWarmupLR, self).__init__(optimizer)

    def get_lr(self):
        if self._step_count <= self.warmup_steps:
            lr = self.lr * self._step_count / self.warmup_steps
        else:
            ratio = 1 - (self._step_count - self.warmup_steps) \
                      / (self.total_steps - self.warmup_steps)
            lr = (self.lr - self.end_lr) * ratio + self.end_lr
        return [lr for group in self.optimizer.param_groups]
