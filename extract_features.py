import torch
import h5py
import torchvision.transforms as transforms

from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from utils.rotation_net import RotationNet
from datasets import RotationDataset

if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Define the transforms to apply to the images
    train_transform = transforms.Compose([
        transforms.RandAugment(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    # Create the data loaders for training and validation
    full_dataset = RotationDataset('mnt/qb/work/akata/jstrueber72/ZSTTT/data/CUB/', transform=train_transform)

    train_size = int(0.8 * len(full_dataset))
    test_size = len(full_dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(full_dataset, [train_size, test_size])
    test_dataset.transform = test_transform

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)
    val_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=4)

    # Load the pre-trained ResNet101 model from timm with additional rotation head
    model = RotationNet(num_classes=train_dataset.num_classes, architecture='resnet101')

    # Print the modified model architecture
    print(model)

    # Define the loss functions for the classification and rotation heads
    classification_criterion = torch.nn.CrossEntropyLoss()
    rotation_criterion = torch.nn.CrossEntropyLoss()

    # Define the optimizer to use for training the model
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    print("Train:", len(train_loader))
    print("Val: ", len(val_loader))

    # Finetune the model
    num_epochs=3
    for epoch in tqdm(range(num_epochs), desc='Current epoch'):
        for idx, data in tqdm(enumerate(train_loader), total=len(train_loader), desc='Training'):
            print(data)

    # Extract the features

    #

