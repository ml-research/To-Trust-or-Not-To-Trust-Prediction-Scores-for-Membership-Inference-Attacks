import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import random_split, Subset
import torchvision.transforms as T
import torchvision
import os


def get_normalization(dataset):
    if dataset == 'cifar10':
        normalization = T.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2470, 0.2435, 0.2616])
    elif dataset == 'stanford_dogs':
        normalization = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    else:
        raise Exception('Dataset not found')

    return normalization


def get_inverse_normalization(dataset):
    mean, std = None, None
    if dataset == 'cifar10':
        mean = [0.4914, 0.4822, 0.4465]
        std = [0.2470, 0.2435, 0.2616]
    elif dataset == 'stanford_dogs':
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
    else:
        raise Exception('Dataset not found')

    mean = torch.tensor(mean)
    std = torch.tensor(std)

    return T.Normalize(mean=-mean / std, std=1 / std)


def get_train_val_split(data, train_size, seed=0, stratify=False, targets=None):
    if train_size > 0 and train_size < 1:
        training_set_length = int(train_size * len(data))
    elif train_size > 1:
        training_set_length = int(train_size)
    else:
        raise RuntimeError('Invalid argument for `size` given.')
    validation_set_length = len(data) - training_set_length
    if stratify:
        indices = list(range(len(data)))
        train_indices, validation_indices = train_test_split(indices, train_size=training_set_length,
                                                             test_size=validation_set_length, random_state=seed,
                                                             stratify=targets)
        train_set = Subset(data, train_indices)
        validation_set = Subset(data, validation_indices)
        return train_set, validation_set

    torch.manual_seed(seed)
    training_set, validation_set = random_split(data, [training_set_length, validation_set_length])

    return training_set, validation_set


def get_subsampled_dataset(dataset, dataset_size=None, proportion=None, seed=0, stratify=False, targets=None):
    if dataset_size > len(dataset):
        raise ValueError('Dataset size is smaller than specified subsample size')
    if dataset_size is None:
        if proportion is None:
            raise ValueError('Neither dataset_size nor proportion specified')
        else:
            dataset_size = int(proportion * len(dataset))
    if stratify:
        indices = list(range(len(dataset)))
        if targets is None:
            targets = dataset.targets
        subsample_indices, _ = train_test_split(indices, train_size=dataset_size, random_state=seed, stratify=targets)
        subsample = Subset(dataset, subsample_indices)
        return subsample

    torch.manual_seed(seed)
    subsample, _ = random_split(dataset, [dataset_size, len(dataset) - dataset_size])
    return subsample


def filter_fake_data(parent_folder, model, limit_samples=None):
    """
    Remove false classified data and optionally limit the number of samples per class.
    """
    transform = T.Compose([T.ToTensor(), T.Normalize([0.4914, 0.4822, 0.4465], [0.2470, 0.2435, 0.2616])])
    model.eval()
    dataset = torchvision.datasets.ImageFolder(parent_folder, transform=transform)
    image_paths = dataset.imgs
    total_images_removed = 0
    with torch.no_grad():
        for i in range(len(dataset)):
            image, label = dataset[i]
            output = model(image.unsqueeze(0))
            pred = torch.argmax(output, dim=1)[0].cpu().item()
            if pred != label:
                os.remove(image_paths[i][0])
                total_images_removed += 1
    if limit_samples:
        for folder in [f.path for f in os.scandir(parent_folder) if f.is_dir()]:
            paths = os.listdir(folder)
            files = [f for f in paths if os.path.isfile(os.path.join(folder, f))]
            files = [os.path.join(folder, f) for f in files]
            if len(files) > limit_samples:
                for file in files[limit_samples:]:
                    os.remove(file)
                    total_images_removed += 1

    print(f'Finish data filtering. Removed {total_images_removed} images.')


def get_mean_and_stddev(dataset: torch.utils.data.Dataset):
    dataloader = torch.utils.data.DataLoader(dataset)

    r_values, g_values, b_values = [], [], []
    for img, _ in dataloader:
        r_values.append(img[0][0, :, :])
        g_values.append(img[0][1, :, :])
        b_values.append(img[0][2:, :, ])
    r_values = torch.stack(r_values)
    g_values = torch.stack(g_values)
    b_values = torch.stack(b_values)

    mean = r_values.mean(), g_values.mean(), b_values.mean()
    std = r_values.std(), g_values.std(), b_values.std()

    return mean, std


def permute_pixels(input_image: torch.Tensor):
    """
    Takes multiple images and permutes the pixels.
    :param input_image: Multiple images in a tensor with shape (3, X, Y)
    :return: Returns a tensor with the same shape but with each image having permuted pixels
    """
    pixel_permutation_indices = torch.randperm(input_image.shape[-2] * input_image.shape[-1]).to(input_image.device)
    # flatten the tensor such that the pixels of each image are flattened resulting in a tensor with shape (N, 3, -1)
    flattened_pixels = input_image.view(input_image.shape[0], -1)
    permuted_images = flattened_pixels.index_select(dim=1, index=pixel_permutation_indices).view(input_image.shape)

    return permuted_images


def create_permuted_dataset(dataset: torch.utils.data.dataset):
    """
    Takes a dataset and permutes the pixels of each image. The labels of each of the images is set to 0.
    """
    permuted_images = []
    for img, _ in dataset:
        permuted_images.append(permute_pixels(img))
    permuted_images = torch.stack(permuted_images)
    return torch.utils.data.TensorDataset(permuted_images, torch.tensor([0] * len(permuted_images)))


def create_scaled_dataset(dataset: torch.utils.data.dataset):
    """
    Takes a dataset and scales the pixels of each image by 255.
    """
    scaled_images = []
    for img, _ in dataset:
        scaled_images.append(img * 255)
    scaled_images = torch.stack(scaled_images)
    return torch.utils.data.TensorDataset(scaled_images, torch.tensor([0] * len(scaled_images)))


def create_un_normalized_dataset(dataset: torch.utils.data.dataset, dataset_name: str):
    """
    Takes a dataset and the name of the dataset and reverses the normalization of the dataset.
    """
    un_normalized_images = []
    un_normalize = get_inverse_normalization(dataset_name)
    for img, _ in dataset:
        un_normalized_images.append(un_normalize(img))
    un_normalized_images = torch.stack(un_normalized_images)
    return torch.utils.data.TensorDataset(un_normalized_images, torch.tensor([0] * len(un_normalized_images)))