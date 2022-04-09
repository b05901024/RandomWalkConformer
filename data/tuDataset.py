import torch_geometric.datasets
from .preprocess import preprocess_item

class myTUDataset(torch_geometric.datasets.TUDataset):
    def download(self):
        super(myTUDataset, self).download()

    def process(self):
        super(myTUDataset, self).process()

    def __getitem__(self, idx):
        if isinstance(idx, int):
            item = self.get(self.indices()[idx])
            item.idx = idx
            return preprocess_item(item, discrete=False, edge_feat=False)
        else:
            return self.index_select(idx)
            