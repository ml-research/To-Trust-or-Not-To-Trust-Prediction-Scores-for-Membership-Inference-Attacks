from torchvision.datasets import ImageFolder
from torchvision.datasets.utils import check_integrity, download_and_extract_archive
from typing import Callable, Optional
import hashlib
import os


class AFHQ(ImageFolder):
    """`Animal Faces-HQ Dataset <https://github.com/clovaai/stargan-v2#animal-faces-hq-dataset-afhq>`_ Dataset.
    This class is based on the CIFAR10-Class from torchvision
    """
    url = "https://www.dropbox.com/s/raw/t9l9o3vsx2jai3z/afhq.zip"
    filename = "afhq.zip"
    zip_md5 = '6909887208505a6e70d732fa20629995'
    image_folder_md5 = '7d4838ee32104e015c8efd68f3069319'

    def __init__(
        self,
        root: str,
        train: bool = True,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        download: bool = False
    ) -> None:
        self.root = root  # the root will be changed when the train or val images are loaded
        self.train = train

        if download:
            self.download()
        if not self._check_integrity():
            raise RuntimeError('Dataset not found or corrupted. You can use download=True to download it')

        # load the images as ImageFolder
        image_folder_path = os.path.join(root, 'afhq', 'train' if self.train else 'val')
        super(AFHQ, self).__init__(image_folder_path, transform=transform, target_transform=target_transform)

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
        # check the integrity of the zip
        if not check_integrity(os.path.join(self.root, self.filename)):
            return False

        # check the integrity of the extracted image folder
        if not self._check_folder_integrity(os.path.join(self.root, "afhq"), self.image_folder_md5):
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

    def download(self) -> None:
        if self._check_integrity():
            print('Files already downloaded and verified')
            return
        download_and_extract_archive(self.url, self.root, filename=self.filename, md5=self.zip_md5)
