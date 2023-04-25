import torch
import h5py
import torchvision.transforms as transforms

from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from utils.rotation_net import RotationNet
from datasets import RotationDataset

if __name__ == '__main__':
    # Hyperparameters:
    num_epochs = 3
    batch_size = 32
    learning_rate = 0.001

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Define the transforms to apply to the images
    train_transform = transforms.Compose([
        transforms.RandAugment(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    valid_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    # Create the data loaders for training and validation
    full_dataset = RotationDataset('mnt/qb/work/akata/jstrueber72/ZSTTT/data/CUB/', transform=train_transform)

    train_size = int(0.8 * len(full_dataset))
    valid_size = len(full_dataset) - train_size
    train_dataset, valid_dataset = torch.utils.data.random_split(full_dataset, [train_size, valid_size])
    valid_dataset.transform = valid_transform

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    # Load the pre-trained ResNet101 model from timm with additional rotation head
    model = RotationNet(num_classes=train_dataset.num_classes, architecture='resnet101')
    model.to(device)

    # Print the modified model architecture
    print(model)

    # Define the loss functions for the classification and rotation heads
    classification_criterion = torch.nn.CrossEntropyLoss()
    rotation_criterion = torch.nn.CrossEntropyLoss()

    # Define the optimizer to use for training the model
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    print("Train:", len(train_loader))
    print("Validation: ", len(valid_loader))

    # Finetune the model
    for epoch in range(num_epochs):
        model.train()

        # Initialize variables to keep track of the loss and accuracy
        running_loss = 0.0
        correct_clf_predictions = 0
        correct_rot_predictions = 0

        # Initialize the progress bar
        train_pbar = tqdm(train_loader, total=len(train_loader), desc=f"Epoch {epoch + 1}/{num_epochs}", unit="batch")

        for data in train_pbar:
            _, rot_images, clf_labels, rot_labels = data
            # Move the data to the device
            images = rot_images.to(device)
            clf_labels = clf_labels.to(device)
            rot_labels = rot_labels.to(device)

            # Zero the gradients
            optimizer.zero_grad()

            # Forward pass
            clf_outputs, rot_outputs = model(images)
            classification_loss = classification_criterion(clf_outputs, clf_labels)
            rotation_loss = rotation_criterion(rot_outputs, rot_labels)

            loss = classification_loss + rotation_loss

            # Backward pass
            loss.backward()
            optimizer.step()

            # Update the loss and accuracy variables
            running_loss += loss.item() * images.size(0)
            predicted_clf_labels = torch.argmax(clf_outputs, dim=1)
            predicted_rot_labels = torch.argmax(rot_outputs, dim=1)
            correct_clf_predictions += (predicted_clf_labels == clf_labels).sum().item()
            correct_rot_predictions += (predicted_rot_labels == rot_labels).sum().item()

        # Calculate the epoch loss and accuracy
        train_loss = running_loss / len(train_dataset)
        train_clf_accuracy = correct_clf_predictions / len(train_dataset)
        train_rot_accuracy = correct_rot_predictions / len(train_dataset)


        # Update the progress bar
        train_pbar.set_postfix(loss=train_loss.item())

        # Validation mode
        model.eval()

        # Initialize progress bar
        valid_pbar = tqdm(valid_loader)
        valid_loss = 0.0
        valid_clf_correct = 0
        valid_rot_correct = 0

        # Iterate over batches
        for data in valid_pbar:
            _, rot_images, clf_labels, rot_labels = data
            # Move the data to the device
            images = rot_images.to(device)
            clf_labels = clf_labels.to(device)
            rot_labels = rot_labels.to(device)

            # Forward pass
            outputs = model(images)
            clf_outputs, rot_outputs = outputs

            # Calculate loss
            class_loss = classification_criterion(clf_outputs, clf_labels)
            rot_loss = rotation_criterion(rot_outputs, rot_labels)
            loss = class_loss + rot_loss

            # Update the loss and accuracy variables
            valid_loss += loss.item() * images.size(0)
            predicted_clf_labels = torch.argmax(clf_outputs, dim=1)
            predicted_rot_labels = torch.argmax(rot_outputs, dim=1)
            valid_clf_correct += (predicted_clf_labels == clf_labels).sum().item()
            valid_rot_correct += (predicted_rot_labels == rot_labels).sum().item()

            # Update progress bar
            valid_clf_accuracy = valid_clf_correct / len(valid_dataset)
            valid_rot_accuracy = valid_rot_correct / len(valid_dataset)
            valid_pbar.set_postfix(loss=valid_loss / (len(valid_dataset) / batch_size), accuracy=valid_clf_accuracy)

        # Print epoch results
        print(f'Epoch {epoch + 1}: train loss={train_loss / len(train_dataset):5.3f}, '
              f'train clf accuracy={train_clf_accuracy:5.3f}, '
              f'train rot accuracy={train_rot_accuracy:5.3f}, '
              f'valid loss={valid_loss / len(valid_dataset)}, '
              f'train clf accuracy={valid_clf_accuracy:5.3f}, '
              f'train rot accuracy={valid_rot_accuracy:5.3f}')

    # Extract the features

    #

