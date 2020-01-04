import torch.utils.data
import torchvision


def get(root, download, batch_size, num_workers):
    dataset, testset, unlabeled = get_data(root, download)
    return create_loaders(
        batch_size,
        num_workers,
        dataset,
        testset,
        unlabeled,
        shuffle_test=False
    )


def create_loaders(
    batch_size,
    num_workers,
    dataset,
    testset,
    unlabeled,
    shuffle_test
):
    data = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers
    )
    test = torch.utils.data.DataLoader(
        testset,
        batch_size=batch_size,
        shuffle=shuffle_test,
        num_workers=num_workers
    )
    unlabel = torch.utils.data.DataLoader(
        unlabeled,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers
    )
    return data, test, unlabel


def get_data(root, download):

    transform = torchvision.transforms.Compose([
        torchvision.transforms.RandomHorizontalFlip(p=0.5),
        torchvision.transforms.Pad(4, padding_mode="symmetric"),
        torchvision.transforms.RandomCrop(96),
        torchvision.transforms.ToTensor()
    ])

    dataset = torchvision.datasets.STL10(
        root=root,
        split="train",
        transform=transform,
        download=download
    )

    testset = torchvision.datasets.STL10(
        root=root,
        split="test",
        transform=torchvision.transforms.ToTensor(),
        download=download
    )

    unlabeled = torchvision.datasets.STL10(
        root=root,
        split="train+unlabeled",
        transform=transform,
        download=download
    )
    return dataset, testset, unlabeled
