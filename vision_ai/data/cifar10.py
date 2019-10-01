import torch.utils.data
import torchvision


def get(root, download, batch_size, num_workers):
    dataset, testset = get_data(root, download)
    return create_loaders(batch_size, num_workers, dataset, testset)


def create_loaders(batch_size, num_workers, dataset, testset):
    data = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers
    )
    test = torch.utils.data.DataLoader(
        testset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers
    )
    return data, test


def get_data(root, download):
    dataset = torchvision.datasets.CIFAR10(
        root=root,
        train=True,
        download=download,
        transform=torchvision.transforms.Compose([
            torchvision.transforms.RandomHorizontalFlip(p=0.5),
            torchvision.transforms.Pad(4, padding_mode="symmetric"),
            torchvision.transforms.RandomCrop(32),
            torchvision.transforms.ToTensor()
        ])
    )
    testset = torchvision.datasets.CIFAR10(
        root=root,
        train=False,
        download=download,
        transform=torchvision.transforms.ToTensor()
    )
    return dataset, testset
