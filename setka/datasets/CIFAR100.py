import torchvision.datasets
import setka
import setka.base

class CIFAR100(setka.base.DataSet):
    def __init__(self,
                 root='~/datasets'):

        train_transforms = torchvision.transforms.Compose([
            torchvision.transforms.RandomCrop(32, padding=4),
            torchvision.transforms.RandomHorizontalFlip(),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

        test_transforms = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

        self.train_data = torchvision.datasets.CIFAR100(
            '~/datasets', train=True, download=True,
            transform=train_transforms)
        self.test_data = torchvision.datasets.CIFAR100(
            '~/datasets', train=False, download=True,
            transform=test_transforms)

        self.n_valid = int(0.05 * len(self.train_data))

        self.subsets = ['train', 'valid', 'test']

    def getlen(self, subset):
        if subset == 'train':
            return len(self.train_data) - self.n_valid
        elif subset == 'valid':
            return self.n_valid
        elif subset == 'test':
            return len(self.test_data)

    def getitem(self, subset, index):
        if subset == 'train':
            image, label = self.train_data[self.n_valid + index]
            return {'image': image, 'label': label}
        elif subset == 'valid':
            image, label = self.train_data[index]
            return {'image': image, 'label': label}
        elif subset == 'test':
            image, label = self.test_data[index]
            return {'image': image, 'label': label}