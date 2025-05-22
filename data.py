import torch
import numpy as np
from torch.utils.data import DataLoader, random_split, Subset
from torchvision import datasets
from torchvision.transforms import v2
from cifar5m import CIFAR5M


def compute_subset_mean_min_max(subset):
    # Access the underlying dataset and indices
    dataset = subset.dataset
    indices = subset.indices

    # Retrieve the data corresponding to the subset indices
    subset_data = torch.stack([dataset[i][0] for i in indices])

    # Compute the mean and min-max of the retrieved data
    data_mean_norm = subset_data.flatten(1).norm(dim=1).mean()
    data_min = subset_data.min()
    data_max = subset_data.max()

    return data_mean_norm, data_min, data_max



def normalization_params(dataset_name):
    if dataset_name in ['MNIST', 'MNIST_odd_even']:
        return (0.1307,), (1.0,)
    elif dataset_name in ["CIFAR10", "CIFAR10_v_a", "CIFAR100", "CIFAR5M", "CIFAR5M_v_a"]:
        return (0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261)
    # elif dataset_name == "CIFAR10_v_a":
    #     return (0.50707516,  0.48654887,  0.44091784), (0.26733429,  0.25643846,  0.27615047)
    elif dataset_name == "celebA":
        return (0.5093, 0.4231, 0.3775), (0.3050, 0.2827, 0.2812) # computed using 20k samples
    else:
        raise ValueError("Unsupported Dataset")


def get_dataset(dataset_name, max_classes=-1, max_samples=-1, train=True, size=None, dtype=torch.float64, normalize=True,normalize_to_sphere=False):
    '''
    :param max_samples: Samples are chosen randomly! If you want the same samples each time, set a seed.
    '''
    transform_list = [v2.ToImage(), v2.ToDtype(dtype, scale=True)]
    if normalize:
        transform_list += [
            v2.Normalize(*normalization_params(dataset_name))
        ] 

    if size:
        transform_list += [v2.Resize(size)]
    if dataset_name == 'MNIST':
        transform = v2.Compose(transform_list)
        dataset = datasets.MNIST(root='./data', train=train, download=True, transform=transform)
        num_classes = 10
    elif dataset_name == 'MNIST_odd_even':
        transform = v2.Compose(transform_list)
        dataset = datasets.MNIST(root='./data', train=train, download=True, transform=transform)
        def create_labels(y0):
            labels_dict = {0: 0, 1: 1, 2: 0, 3: 1, 4: 0, 5: 1, 6: 0, 7: 1, 8: 0, 9: 1}
            y0 = torch.stack([torch.tensor(labels_dict[int(cur_y)]) for cur_y in y0])
            return y0
        dataset.og_targets = dataset.targets
        dataset.targets = create_labels(dataset.targets)
        num_classes = 10
    elif dataset_name == 'CIFAR10':
        transform = v2.Compose(transform_list)
        dataset = datasets.CIFAR10(root='./data', train=train, download=True, transform=transform)
        num_classes = 10
    elif dataset_name == 'CIFAR5M':
        num_classes = 10
        transform = v2.Compose(transform_list)
        if train:
            dataset = CIFAR5M(transform=transform)
        else:
            dataset = datasets.CIFAR10(root='./data', train=train, download=True, transform=transform)
        num_classes = 10
    elif dataset_name == 'CIFAR100':
        transform = v2.Compose(transform_list)
        dataset = datasets.CIFAR100(root='./data', train=train, download=True, transform=transform)
        num_classes = 100
    elif dataset_name == 'CIFAR10_v_a':
        transform = v2.Compose(transform_list)
        dataset = datasets.CIFAR10(root='./data', train=train, download=True, transform=transform)
        num_classes = 10
        def create_labels(y0):
            labels_dict = {0: 0, 1: 0, 2: 1, 3: 1, 4: 1, 5: 1, 6: 1, 7: 1, 8: 0, 9: 0}
            y0 = torch.stack([torch.tensor(labels_dict[int(cur_y)]) for cur_y in y0])
            return y0
        dataset.og_targets = dataset.targets
        dataset.targets = create_labels(dataset.targets)
    elif dataset_name == 'CIFAR5M_v_a':
        transform = v2.Compose(transform_list)
        if train:
            dataset = CIFAR5M(transform=transform)
        else:
            dataset = datasets.CIFAR10(root='./data', train=train, download=True, transform=transform)
        num_classes = 10

        def create_labels(y0):
            labels_dict = {0: 0, 1: 0, 2: 1, 3: 1, 4: 1, 5: 1, 6: 1, 7: 1, 8: 0, 9: 0}
            y0 = torch.stack([torch.tensor(labels_dict[int(cur_y)]) for cur_y in y0])
            return y0

        dataset.og_targets = dataset.targets
        dataset.targets = create_labels(dataset.targets)
    elif dataset_name == 'celebA':
        crop_size = 178 if not size else size
        transform_list += [v2.CenterCrop(crop_size)] # can play with this
        # if not normalize:
        #     transform_list.append(v2.Normalize((-0.5, -0.5, -0.5), (2, 2, 2))) # celebA images are in [-1, 1] by default. Unnormalized images should be in [0, 1] for the VAE code
        transform = v2.Compose(transform_list)
        split = "train" if train else "test"
        dataset = datasets.CelebA(root="./data", split=split, download=True, transform=transform)
        # dataset.attr = dataset.attr[:,dataset.attr_names.index('Male')].unsqueeze(1)
        dataset.attr = ((dataset.attr) - 0.5) * 2 #make attributes -1 or 1
        # dataset.targets = dataset.attr.squeeze()
        num_classes = 40
    else:
        raise ValueError("Unsupported dataset")

    indices = torch.arange(len(dataset))
    if dataset_name == "celebA":
        if max_classes != -1 and max_classes < num_classes:
            dataset.attr = dataset.attr[:, max_classes]
            num_classes = max_classes
        if max_samples != -1 and max_samples < len(indices):
            indices = np.random.choice(len(indices), max_samples, replace=False)
    else:
        if max_classes != -1 and max_classes < num_classes:
            indices = torch.nonzero(torch.Tensor(dataset.targets) < max_classes).squeeze(1)
            num_classes = max_classes

        if max_samples != -1 and max_samples < len(indices):
            num_per_class = max_samples // num_classes
            print(f"Selecting {num_per_class} samples per class")
            indices = []
            for i in range(num_classes):
                indices.extend(np.random.choice(np.where(np.array(dataset.targets) == i)[0], num_per_class, replace=False)) # balanced sampling

    if max_classes != -1 or max_samples != -1:
        dataset = Subset(dataset, indices)

    data_mean_norm, min_value, max_value = compute_subset_mean_min_max(dataset) # updating to include min and max values
    if normalize_to_sphere:
        data_mean_norm = 1.0
    return dataset, num_classes, data_mean_norm, min_value, max_value


class LightweightDataLoader:
    def __init__(self, dataset, batch_size=1):
        """
        Replacement for pytorch's DataLoader, which seems to be too slow. Loads all the data at initialization
        """
        self.batch_size = batch_size
        self.indices = list(range(len(dataset)))
        self.num_samples = len(dataset)
        dataloader = DataLoader(dataset, batch_size=self.num_samples, shuffle=False)
        self.data, self.labels = next(iter(dataloader))


    def __iter__(self):
        """
        Returns an iterator for the DataLoader.

        Shuffles the dataset if specified and creates batches.
        """
        self.batches = [
            self.indices[i:i + self.batch_size]
            for i in range(0, self.num_samples, self.batch_size)
        ]
        self.current_batch = 0
        return self

    def __next__(self):
        """
        Fetches the next batch of data.

        Raises:
            StopIteration: When no more data is available.
        """
        if self.current_batch >= len(self.batches):
            raise StopIteration

        batch_indices = self.batches[self.current_batch]
        self.current_batch += 1
        
        return self.data[batch_indices], self.labels[batch_indices]

    def __len__(self):
        """Returns the total number of batches."""
        return (self.num_samples + self.batch_size - 1) // self.batch_size