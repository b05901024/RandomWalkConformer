import torch_geometric.datasets
from .preprocess import preprocess_item

class myZINCDataset(torch_geometric.datasets.ZINC):
    def download(self):
        super(myZINCDataset, self).download()

    def process(self):
        super(myZINCDataset, self).process()

    def __getitem__(self, idx):
        if isinstance(idx, int):
            item = self.get(self.indices()[idx])
            item.idx = idx
            return preprocess_item(item)
        else:
            return self.index_select(idx)
            