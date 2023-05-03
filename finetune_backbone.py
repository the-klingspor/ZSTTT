import torch
import torchvision.transforms as transforms
import torch.optim.lr_scheduler as lr_scheduler
import argparse

from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

import logger

from utils.rotation_net import RotationNet
from datasets import RotationDataset

parser = argparse.ArgumentParser()

#todo: write help statements

# Dataset and paths
parser.add_argument('--dataset', default='CUB', help='CUB')

# Hyper parameters
parser.add_argument('--architecture', type=str, default="resnet50")
parser.add_argument('--epochs', type=int, default=90)
parser.add_argument('--batch_size', type=int, default=16)
parser.add_argument('--learning_rate', type=float, default=5e-4)
parser.add_argument('--momentum', type=float, default=0.9)
parser.add_argument('--weight_decay', type=float, default=1e-4)
parser.add_argument('--label_smoothing', type=float, default=0.1)
parser.add_argument('--loss_weight', type=float, default=0.1)
parser.add_argument('--image_size', type=int, default=336)

## wandb
parser.add_argument('--log_online', action='store_true',
                    help='Flag. If set, run metrics are stored online in addition to offline logging. Should generally '
                         'be set.')
parser.add_argument('--wandb_key', default='65954b19f28cc0f35372188d50be8f11cdb79321', type=str, help='API key for W&B.')
parser.add_argument('--project', default='Sample_Project', type=str,
                    help='Name of the project - relates to W&B project names. In --savename default setting part of '
                         'the savename.')
parser.add_argument('--group', default='', type=str, help='Name of the group - relates to W&B group names - all runs '
                                                          'with same setup but different seeds are logged into one '
                                                          'group. In --savename default setting part of the savename. '
                                                          'Name is created as model_dataset_group')
parser.add_argument('--savename', default='group_plus_seed', type=str, help='Run savename - if default, the savename'
                                                                            ' will comprise the project and group name '
                                                                            '(see wandb_parameters()).')
parser.add_argument('--name_seed', type=str, default=0, help="Randomly generated code as name for the run.")
parser.add_argument('--outname', help='folder to output data and model checkpoints')


if __name__ == '__main__':
    opt = parser.parse_args()
    if opt.log_online:
        logger.setup_logger(opt)

    # Hyperparameters
    image_size_unc  = int(opt.image_size // 0.95)
    step_size = int(opt.epochs // 3)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Define the transforms to apply to the images
    # ToDo: automatically choose transformation used by timm
    train_transform = transforms.Compose([
        transforms.RandAugment(),
        transforms.Resize(size=image_size_unc, interpolation=transforms.InterpolationMode.BICUBIC, max_size=None, antialias=None),
        transforms.CenterCrop(size=opt.image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    valid_transform = transforms.Compose([
        transforms.Resize(size=image_size_unc, interpolation=transforms.InterpolationMode.BICUBIC, max_size=None, antialias=None),
        transforms.CenterCrop(size=opt.image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    # Create the data loaders for training and validation
    full_dataset = RotationDataset('/mnt/qb/akata/jstrueber72/datasets/CUB/', transform=train_transform)

    train_size = int(0.8 * len(full_dataset))
    valid_size = len(full_dataset) - train_size
    train_dataset, valid_dataset = torch.utils.data.random_split(full_dataset, [train_size, valid_size])
    valid_dataset.transform = valid_transform

    train_loader = DataLoader(train_dataset, batch_size=opt.batch_size, shuffle=True, num_workers=4)
    valid_loader = DataLoader(valid_dataset, batch_size=opt.batch_size, shuffle=False, num_workers=4)

    # Load the pre-trained ResNet101 model from timm with additional rotation head
    model = RotationNet(num_classes=full_dataset.num_classes, architecture=opt.architecture)
    model.to(device)

    # Print the modified model architecture
    print(model)

    # Define the loss functions for the classification and rotation heads
    classification_criterion = torch.nn.CrossEntropyLoss(label_smoothing=opt.label_smoothing)
    rotation_criterion = torch.nn.CrossEntropyLoss()

    # Define the optimizer to use for training the model
    optimizer = torch.optim.SGD(model.parameters(), lr=opt.learning_rate, momentum=opt.momentum,
                                weight_decay=opt.weight_decay)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=0.1)

    print("Number of training samples: ", len(train_dataset))
    print("Number of validation samples: ", len(valid_dataset))

    # Finetune the model
    for epoch in range(opt.epochs):
        model.train()

        # Initialize variables to keep track of the loss and accuracy
        train_loss = 0.0
        correct_clf_predictions = 0
        correct_rot_predictions = 0

        # Initialize the progress bar
        train_pbar = tqdm(enumerate(train_loader), total=len(train_loader), desc=f"Epoch {epoch + 1}/{opt.epochs}", unit="batch")

        for idx, data in train_pbar:
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
            loss = classification_loss + opt.loss_weight * rotation_loss

            # Backward pass
            loss.backward()
            optimizer.step()

            # Update the loss and accuracy variables
            train_loss += loss.item()
            predicted_clf_labels = torch.argmax(clf_outputs.detach(), dim=1)
            predicted_rot_labels = torch.argmax(rot_outputs.detach(), dim=1)
            correct_clf_predictions += (predicted_clf_labels == clf_labels).sum().item()
            correct_rot_predictions += (predicted_rot_labels == rot_labels).sum().item()

            # Update the progress bar
            train_pbar.set_postfix(loss=train_loss / (idx + 1))

        # Step of LR schedule
        scheduler.step()


        # Calculate the epoch loss and accuracy
        train_clf_accuracy = correct_clf_predictions / len(train_dataset)
        train_rot_accuracy = correct_rot_predictions / len(train_dataset)

        # Validation mode
        model.eval()

        # Initialize progress bar
        valid_pbar = tqdm(enumerate(valid_loader))
        valid_loss = 0.0
        valid_clf_correct = 0
        valid_rot_correct = 0
        valid_clf_accuracy = 0
        valid_rot_accuracy = 0

        # Iterate over batches
        print("\n")
        for idx, data in valid_pbar:
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
            loss = class_loss + opt.loss_weight * rot_loss

            # Update the loss and accuracy variables
            valid_loss += loss.item()
            predicted_clf_labels = torch.argmax(clf_outputs, dim=1)
            predicted_rot_labels = torch.argmax(rot_outputs, dim=1)
            valid_clf_correct += (predicted_clf_labels == clf_labels).sum().item()
            valid_rot_correct += (predicted_rot_labels == rot_labels).sum().item()

            # Update progress bar
            valid_pbar.set_postfix(loss=valid_loss / (idx + 1),
                                   accuracy=valid_clf_correct / (opt.batch_size * (idx + 1)))

        # Calculate validation epoch accuracies
        valid_clf_accuracy = valid_clf_correct / len(valid_dataset)
        valid_rot_accuracy = valid_rot_correct / len(valid_dataset)

        dict_to_log = {
            'valid_loss': valid_loss * opt.batch_size / len(valid_dataset),
            'valid_rotation_accuracy': valid_rot_accuracy,
            'valid_classification_accuracy': valid_clf_accuracy
        }
        logger.log(dict_to_log)

        # Print epoch results
        print(f'\nEpoch {epoch + 1}: train loss={train_loss * opt.batch_size / len(train_dataset):5.3f}, '
              f'train clf acc={train_clf_accuracy:5.3f}, '
              f'train rot acc={train_rot_accuracy:5.3f}, '
              f'valid loss={valid_loss * opt.batch_size / len(valid_dataset):5.3f}, '
              f'valid clf acc={valid_clf_accuracy:5.3f}, '
              f'valid rot acc={valid_rot_accuracy:5.3f}')

    # Save the model
    torch.save(model.state_dict(), "/mnt/qb/work/akata/jstrueber72/ZSTTT/data/CUB/resnet50_cub.pth")
