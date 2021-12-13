import PIL.Image
from torchvision.datasets import ImageFolder
from torchvision.datasets.utils import check_integrity, download_url
from typing import Callable, Optional, List
import hashlib
import os
import numpy as np
from sklearn.model_selection import train_test_split
import pickle
import torch
from tqdm import tqdm
import shutil

import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'stylegan2-ada-pytorch'))

os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'

cifar10_classes = {
    0: 'plane', 1: 'car', 2: 'bird', 3: 'cat', 4: 'deer', 5: 'dog', 6: 'frog', 7: 'horse', 8: 'ship', 9: 'truck'
}


class FakeAFHQDogs(ImageFolder):
    """Dataset that was generated using the `StyleGAN2-ADA<https://github.com/NVlabs/stylegan2-ada-pytorch>`.
    This class is based on the AHFQ-Class from torchvision.
    """
    base_folder = 'fake_afhq_dogs'
    pretrained_net_url = "https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/afhqdog.pkl"
    pretrained_net_filename = "pretrained_net_afhqdog.pkl"
    pretrained_net_md5 = '8fa2162d23fe76012bec12b1910b4e65'
    image_folder_md5_for_5k_imgs = '7bed6525fb3ef93bc7a34a694ef97fc6'

    def __init__(
        self,
        root: str,
        train: bool = True,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        train_test_split_factor: float = 0.5,
        seed: int = 42,
        num_generated_samples: int = 5000
    ) -> None:
        self.seed = seed
        torch.manual_seed(self.seed)
        self.base_folder = os.path.join(root, self.base_folder)
        self.train = train
        self.num_generated_samples = num_generated_samples
        self.train_test_split_factor = train_test_split_factor

        self.generate_samples()

        if not self._check_integrity():
            raise RuntimeError('Dataset not found or corrupted.')

        # load the images as ImageFolder
        super(FakeAFHQDogs, self).__init__(
            os.path.join(self.base_folder, "Images"), transform=transform, target_transform=target_transform
        )

        # if the number of samples does not match, re-generate the images
        if self.num_generated_samples != len(self.targets):
            print(
                f'Number of images is different. Found {len(self.targets)} images but there should ' +
                f'be {self.num_generated_samples}. Re-generating images...'
            )
            shutil.rmtree(os.path.join(self.base_folder, "Images"))
            self.generate_samples()
            if not self._check_integrity():
                raise RuntimeError('Dataset not found or corrupted.')

        # get the indices of the samples for each split
        train_split_indices, test_split_indices = train_test_split(
            [i for i in range(len(self.targets))],
            test_size=self.train_test_split_factor,
            random_state=self.seed,
            shuffle=True,
            stratify=self.targets
        )
        sample_indices = train_split_indices if self.train else test_split_indices

        # filter the imgs, the samples and the targets
        self.imgs = np.array(self.imgs)[sample_indices].tolist()
        self.samples = np.array(self.samples)[sample_indices].tolist()
        self.targets = np.array(self.targets)[sample_indices].tolist()

    def __getitem__(self, index: int):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        path, _ = self.samples[index]
        target = self.targets[index]
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return sample, target

    def _check_integrity(self):
        # check the integrity of the pre-trained network
        if not check_integrity(os.path.join(self.base_folder, self.pretrained_net_filename), self.pretrained_net_md5):
            return False

        # check the integrity of the image folder but only if there are supposed to be 5k samples
        if self.num_generated_samples == 5000 and not self._check_folder_integrity(
            os.path.join(self.base_folder, "Images"), self.image_folder_md5_for_5k_imgs
        ):
            return False

        return True

    def _check_folder_integrity(self, folder_name, md5):
        """
        Inspired by https://stackoverflow.com/a/24937710. Special thanks to
        Andy <https://stackoverflow.com/users/189134/andy>.
        """
        md5sum = hashlib.md5()
        if not os.path.exists(folder_name):
            return False

        for root, dirs, files in os.walk(folder_name):
            dirs.sort()
            for fnames in sorted(files):
                fpath = os.path.join(root, fnames)
                try:
                    f = open(fpath, 'rb')
                except:
                    # if the file cannot be opened just continue
                    f.close()
                    continue

                for chunk in iter(lambda: f.read(4096), b''):
                    md5sum.update(hashlib.md5(chunk).digest())
                f.close()
        return md5sum.hexdigest() == md5

    def generate_samples(self) -> None:
        if self._check_integrity():
            print('Files already downloaded and verified')
            return
        # download the pre-trained network
        download_url(
            self.pretrained_net_url, self.base_folder, self.pretrained_net_filename, md5=self.pretrained_net_md5
        )

        # load the pre-trained network
        with open(os.path.join(self.base_folder, self.pretrained_net_filename), 'rb') as f:
            generator = pickle.load(f)['G_ema']

        # generate latent codes
        latent_codes = torch.randn([self.num_generated_samples, generator.z_dim])
        # divide the latent codes into 10 chunks for each class
        latent_codes = latent_codes.chunk(10)

        # try to move the geneartor to the gpu
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if torch.cuda.is_available():
            try:
                generator = generator.to(device)
            except:
                print('Failed to push the generator on to the GPU. Generating samples on the CPU instead.')
                device = 'cpu'

        for i, latent_code_chunk in tqdm(enumerate(latent_codes), desc='Generating images', total=len(latent_codes)):
            imgs = []
            for code in latent_code_chunk:
                code = code.to(device)
                # generate the images
                img = generator(code.unsqueeze(0), None, truncation_psi=0.7)[0].cpu()
                imgs.append(img)
            self._save_images(imgs, os.path.join(self.base_folder, "Images", f'GoodBoys{i}'))

    def _save_images(self, images: List[torch.tensor], out_dir: str):
        # if the directory does already exist delete it
        if os.path.exists(out_dir) and os.path.isdir(out_dir):
            raise RuntimeError(
                f'There is already a directory present at {out_dir}. If you want to re-generate images please delete the Image directory first.'
            )

        os.makedirs(out_dir, exist_ok=False)

        for idx, img in enumerate(images):
            img = (img.permute(1, 2, 0) * 127.5 + 128).clamp(0, 255).to(torch.uint8)
            PIL.Image.fromarray(img.numpy(), 'RGB').save(os.path.join(out_dir, f'{idx:05d}.png'))
