import torch.utils.data
import torchvision


def get(root, download, batch_size, num_workers):
    dataset = torchvision.datasets.MNIST(
        root=root,
        train=True,
        download=download,
        transform=torchvision.transforms.Compose([
            torchvision.transforms.RandomCrop(28, padding=2),
            torchvision.transforms.ToTensor()
        ])
    )

    data = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers
    )

    testset = torchvision.datasets.MNIST(
        root=root,
        train=False,
        download=download,
        transform=torchvision.transforms.ToTensor()
    )

    test = torch.utils.data.DataLoader(
        testset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers
    )

    return data, test
