import torchvision.datasets
import setka
import setka.base


class CIFAR(setka.base.DataSet):
    def __init__(self, n_classes=10, root='~/datasets'):
        super(CIFAR, self).__init__()
        self.n_classes = n_classes
        self.root = root

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

        base_ds = getattr(torchvision.datasets, 'CIFAR'+str(self.n_classes))
        self.train_data = base_ds(self.root, train=True, download=True, transform=train_transforms)
        self.test_data = base_ds(self.root, train=False, download=True, transform=test_transforms)

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
