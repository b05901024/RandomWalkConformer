from ogb.graphproppred import PygGraphPropPredDataset
from ogb.lsc.pcqm4m_pyg import PygPCQM4MDataset
from ogb.lsc import PygPCQM4Mv2Dataset
from .preprocess import preprocess_item

# obgb
class myGraphPropPredDataset(PygGraphPropPredDataset):
    def download(self):
        super(myGraphPropPredDataset, self).download()

    def process(self):
        super(myGraphPropPredDataset, self).process()

    def __getitem__(self, idx):
        if isinstance(idx, int):
            item = self.get(self.indices()[idx])
            item.idx = idx
            return preprocess_item(item)
        else:
            return self.index_select(idx)

# PCQM4M
class myPygPCQM4MDataset(PygPCQM4MDataset):
    def download(self):
        super(myPygPCQM4MDataset, self).download()

    def process(self):
        super(myPygPCQM4MDataset, self).process()

    def __getitem__(self, idx):
        if isinstance(idx, int):
            item = self.get(self.indices()[idx])
            item.idx = idx
            return preprocess_item(item)
        else:
            return self.index_select(idx)

# PCQM4Mv2
class myPygPCQM4Mv2Dataset(PygPCQM4Mv2Dataset):
    def download(self):
        super(myPygPCQM4Mv2Dataset, self).download()

    def process(self):
        super(myPygPCQM4Mv2Dataset, self).process()

    def __getitem__(self, idx):
        if isinstance(idx, int):
            item = self.get(self.indices()[idx])
            item.idx = idx
            return preprocess_item(item)
        else:
            return self.index_select(idx)
            