import torch_geometric.datasets
from .preprocess import preprocess_item

class myGNNBenchmarkDataset(torch_geometric.datasets.GNNBenchmarkDataset):
    def download(self):
        super(myGNNBenchmarkDataset, self).download()

    def process(self):
        super(myGNNBenchmarkDataset, self).process()

    def __getitem__(self, idx):
        if isinstance(idx, int):
            item = self.get(self.indices()[idx])
            item.idx = idx
            return preprocess_item(item, False, True)
        else:
            return self.index_select(idx)
            