from .cifar10 import get_data, create_loaders
from .cover_dataset import CoverDataset


def get(root, download, batch_size, num_workers, box=8):
    dataset, testset = get_data(root, download)
    dataset = CoverDataset(dataset, box)
    testset = CoverDataset(testset, box)
    return create_loaders(batch_size, num_workers, dataset, testset)
