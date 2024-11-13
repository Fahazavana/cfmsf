import torch
from torch.utils.data import DataLoader, Subset, random_split
from torchvision import datasets, transforms


class DataFactory:
    def __init__(
        self,
        dataset_name,
        data_dir="./data",
        transform=None,
        download=True,
        train_subset_size=None,
        split_train=True,  # Add split_train argument
        validation_split=0.2,  # Default validation split
    ):
        self.dataset_name = dataset_name
        self.data_dir = data_dir
        self.transform = transform
        self.download = download
        self.train_subset_size = train_subset_size
        self.split_train = split_train  # Store split_train value
        self.validation_split = validation_split  # Store validation_split value

        # Load and split the dataset
        self._load_and_split()

    def _load_and_split(self):
        """Loads and splits the specified dataset."""

        # Define default transform if none provided
        if self.transform is None:
            self.transform = transforms.ToTensor()

        if self.dataset_name == "CIFAR10":
            self.train_data = datasets.CIFAR10(
                root=self.data_dir, train=True, download=self.download, transform=self.transform
            )
            self.test_data = datasets.CIFAR10(
                root=self.data_dir, train=False, download=self.download, transform=self.transform
            )
        elif self.dataset_name == "MNIST":
            self.train_data = datasets.MNIST(
                root=self.data_dir, train=True, download=self.download, transform=self.transform
            )
            self.test_data = datasets.MNIST(
                root=self.data_dir, train=False, download=self.download, transform=self.transform
            )
        elif self.dataset_name == "FashionMNIST":
            self.train_data = datasets.FashionMNIST(
                root=self.data_dir, train=True, download=self.download, transform=self.transform
            )
            self.test_data = datasets.FashionMNIST(
                root=self.data_dir, train=False, download=self.download, transform=self.transform
            )
        elif self.dataset_name == "CelebA":
            self.train_data = datasets.CelebA(
                root=self.data_dir,
                split="train",
                download=self.download,
                transform=self.transform,
            )
            self.test_data = datasets.CelebA(
                root=self.data_dir, split="test", download=self.download, transform=self.transform
            )
        else:
            raise ValueError(f"Unsupported dataset: {self.dataset_name}")

        # Handle train subset size
        if self.train_subset_size is not None:
            # Use a random subset of the training data
            indices = torch.randperm(len(self.train_data))[: self.train_subset_size]
            self.train_data = Subset(self.train_data, indices)

        # Split the training data into training and validation sets (if requested)
        if self.split_train:
            if self.dataset_name != "CelebA":
                train_size = int((1 - self.validation_split) * len(self.train_data))
                valid_size = len(self.train_data) - train_size
                self.train_data, self.valid_data = random_split(
                    self.train_data, [train_size, valid_size]
                )
            else:
                self.valid_data = datasets.CelebA(
                    root=self.data_dir,
                    split="valid",
                    download=self.download,
                    transform=self.transform,
                )
                indices = torch.randperm(len(self.valid_data))[: 10000]
                self.valid_data = Subset(self.valid_data, indices)
        else:
            # If not splitting, use the entire dataset as training data
            self.valid_data = None

    def get_loader(self, batch_size, num_workers, data_type):
        if data_type == "train":
            return DataLoader(
                self.train_data,
                batch_size=batch_size,
                shuffle=True,
                num_workers=num_workers,
            )
        if data_type == "valid":
            if self.valid_data is not None:  # Check if validation data exists
                return DataLoader(
                    self.valid_data,
                    batch_size=batch_size,
                    shuffle=False,
                    num_workers=num_workers,
                )

            else:
                return None
        if data_type == "test":
            return DataLoader(
                self.test_data,
                batch_size=batch_size,
                shuffle=False,
                num_workers=num_workers,
            )
        raise ValueError(f"Invalid loader type: {loader_type}")
