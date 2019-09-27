import torchvision


def get(root, download):
    dataset = torchvision.datasets.MNIST(
        root=root,
        train=True,
        download=download,
        transform=torchvision.transforms.Compose([
            torchvision.transforms.RandomCrop(32, padding=4),
            torchvision.transforms.RandomRotation(
                45,
                resample="bilinear"
            ),
            torchvision.transforms.ToTensor()
        ])
    )

    testset = torchvision.datasets.MNIST(
        root=root,
        train=False,
        download=download,
        transform=torchvision.transform.ToTensor()
    )

    return dataset, testset
