from torchvision import datasets, transforms
from torch.utils.data import DataLoader

def get_dataloaders(batch_size: int):

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize( (0.49139968, 0.48215827, 0.44653124), (0.24703233, 0.24348505, 0.26158768) )
    ])

    # Download train and test data sets with transforms
    train_data = datasets.CIFAR10(root="./data", train=True, transform=transform, download=True)
    test_data = datasets.CIFAR10(root="./data", train=False, transform=transform, download=True)

    # make data loaders with batch size
    train_loader = DataLoader(train_data, batch_size, shuffle=True)
    test_loader = DataLoader(test_data, batch_size, shuffle=False)

    return train_loader, test_loader

# if __name__ == "__main__": 
#     train_loader, test_loader = get_dataloaders(64).   # 782
#     print(len(train_loader), len(test_loader))         # 157

#     images, labels = next(iter(train_loader))
#     print(images.shape)     # torch.Size([64, 3, 32, 32])       32 by 32 images with color channel 3
#     print(labels.shape)     # torch.Size([64])