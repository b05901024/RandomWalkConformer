from functools import partial
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader
from .collator import collate_fn

class myDataModule(LightningDataModule):
    def __init__(
        self,
        batch_size: int = 1024,
        num_workers: int = 0,
        seed: int = 42,
        fp_path: str = "",
        drop_last: bool = False,
        tr_set=None,
        val_set=None,
        tt_set=None,
        fp=None,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.batch_size     = batch_size
        self.num_workers    = num_workers
        self.drop_last      = drop_last
        self.tr_set         = tr_set
        self.val_set        = val_set
        self.tt_set         = tt_set
        self.fp             = fp
    
    def train_dataloader(self):
        return DataLoader(
                self.tr_set,
                self.batch_size,
                shuffle=True,
                num_workers=self.num_workers,
                pin_memory=True,
                drop_last=self.drop_last,
                collate_fn=partial(collate_fn, fp=self.fp)
            )
    
    def val_dataloader(self):
        return DataLoader(
                self.val_set,
                self.batch_size,
                shuffle=False,
                num_workers=self.num_workers,
                pin_memory=False,
                drop_last=self.drop_last,
                collate_fn=partial(collate_fn, fp=self.fp)
            )
    
    def test_dataloader(self):
        return DataLoader(
                self.tt_set,
                self.batch_size,
                shuffle=False,
                num_workers=self.num_workers,
                pin_memory=False,
                drop_last=self.drop_last,
                collate_fn=partial(collate_fn, fp=self.fp)
            )
