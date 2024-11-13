import torch
from torchvision import datasets
from torchvision import transforms
from torch.utils.data import DataLoader


def get_mean_std(data_loader):
    ch_sum, ch_sqr_sum, num_samples = 0, 0, 0

    for data, _ in data_loader:
        batch_size, channels, height, width = data.shape
        num_pixels = batch_size * height * width

        ch_sum += torch.sum(data, dim=[0, 2, 3])
        ch_sqr_sum += torch.sum(data**2, dim=[0, 2, 3])
        num_samples += num_pixels

    mean = ch_sum / num_samples
    std = ((ch_sqr_sum / num_samples) - mean**2).sqrt()

    return mean, std


cifar_train_path = "/Volumes/Jeannie/AI/DATASET/CIFAR10/train"
cifar_test_path = "/Volumes/Jeannie/AI/DATASET/CIFAR10/test"
train_data = datasets.CIFAR10(
    root=cifar_train_path, download=False, transform=transforms.ToTensor()
)
trainloader = DataLoader(train_data, batch_size=256)
print("Train data")
mean, std = get_mean_std(trainloader)
print(mean)  # tensor([0.4914, 0.4822, 0.4465])
print(std)  # tensor([0.2470, 0.2435, 0.2616])

print("Test data")
test_data = datasets.CIFAR10(
    root=cifar_test_path, download=False, transform=transforms.ToTensor()
)
testloader = DataLoader(test_data, batch_size=256)
mean, std = get_mean_std(testloader)
print(mean)  # tensor([0.4914, 0.4822, 0.4465])
print(std)  # tensor([0.2470, 0.2435, 0.2616])


MNIST_train_path = "/Volumes/Jeannie/AI/DATASET/MNIST/train"
MNIST_test_path = "/Volumes/Jeannie/AI/DATASET/MNIST/test"

train_data = datasets.MNIST(
    root=MNIST_train_path, download=False, transform=transforms.ToTensor()
)
trainloader = DataLoader(train_data, batch_size=256)
print("Train data")
mean, std = get_mean_std(trainloader)
print(mean)  # tensor([0.1307])
print(std)  # tensor([0.3081])

print("Test data")
test_data = datasets.MNIST(
    root=MNIST_test_path, download=False, transform=transforms.ToTensor()
)
testloader = DataLoader(test_data, batch_size=256)
mean, std = get_mean_std(testloader)
print(mean)  # tensor([0.1307])
print(std)  # tensor([0.3081])


FashionMNIST_train_path = "/Volumes/Jeannie/AI/DATASET/FashionMNIST/train"
FashionMNIST_test_path = "/Volumes/Jeannie/AI/DATASET/FashionMNIST/test"

train_data = datasets.FashionMNIST(
    root=FashionMNIST_train_path, download=False, transform=transforms.ToTensor()
)
trainloader = DataLoader(train_data, batch_size=256)
print("Train data")
mean, std = get_mean_std(trainloader)
print(mean)  # tensor([0.2860])
print(std)  # tensor([0.3530])

print("Test data")
test_data = datasets.FashionMNIST(
    root=FashionMNIST_test_path, download=False, transform=transforms.ToTensor()
)
testloader = DataLoader(test_data, batch_size=256)
mean, std = get_mean_std(testloader)
print(mean)  # tensor([0.2860])
print(std)  # tensor([0.3530])
