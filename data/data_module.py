from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader
from .collator import collate_fn
from functools import partial

class myDataModule(LightningDataModule):
    def __init__(
        self,
        batch_size: int = 1024,
        num_workers: int = 0,
        seed: int = 42,
        max_node: int = 128,
        tr_set=None,
        val_set=None,
        tt_set=None,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.batch_size     = batch_size
        self.num_workers    = num_workers
        self.max_node       = max_node
        self.tr_set         = tr_set
        self.val_set        = val_set
        self.tt_set         = tt_set
    
    def train_dataloader(self):
        return DataLoader(
                self.tr_set,
                self.batch_size,
                shuffle=True,
                num_workers=self.num_workers,
                pin_memory=True,
                collate_fn=partial(collate_fn, max_node=self.max_node),
            )
    
    def val_dataloader(self):
        return DataLoader(
                self.val_set,
                self.batch_size,
                shuffle=False,
                num_workers=self.num_workers,
                pin_memory=False,
                collate_fn=partial(collate_fn, max_node=self.max_node),
            )
    
    def test_dataloader(self):
        return DataLoader(
                self.tt_set,
                self.batch_size,
                shuffle=False,
                num_workers=self.num_workers,
                pin_memory=False,
                collate_fn=partial(collate_fn, max_node=self.max_node),
            )
