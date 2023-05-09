import torch
import torchvision.transforms as transforms
import random
import numpy as np
import copy


def rotation_ttt_loop(model, image, opt):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = copy.deepcopy(model)
    model.to(device)

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=opt.ttt_learning_rate, momentum=opt.ttt_momentum,
                                weight_decay=opt.ttt_weight_decay)

    # Training loop using the rotation self-supervision
    losses = []
    for i in range(opt.ttt_n_loops):
        model.train()
        rotated_images, rotation_labels = generate_rotation_batch(image, opt)
        rotated_images = rotated_images.to(device)
        rotation_labels = rotation_labels.to(device)

        # Zero the gradients
        optimizer.zero_grad()

        # Forward pass
        clf_outputs, rot_outputs = model(rotated_images)
        loss = criterion(rot_outputs, rotation_labels)

        # Backward pass
        loss.backward()
        optimizer.step()

        losses.append(loss.cpu().detach())

    # Single forward pass on original image to extract feature
    model.eval()
    image_size_unc  = int(opt.image_size // 0.95)
    final_transform = transforms.Compose([
        transforms.Resize(size=image_size_unc, interpolation=transforms.InterpolationMode.BICUBIC, max_size=None, antialias=None),
        transforms.CenterCrop(opt.image_size),
        transforms.ToTensor(),
        transforms.RandomHorizontalFlip(p=0.5)
    ])

    transformed_image = torch.unsqueeze(final_transform(image), dim=0).to(device)

    feature = model.forward_features(transformed_image).cpu().detach()
    losses = torch.stack(losses)

    return feature, losses


def generate_rotation_batch(image, opt):
    # Initialize the list of augmented images and labels
    augmented_images = []
    rotation_labels = []

    # Create a transformation pipeline for the random crop and flip
    image_size_unc  = int(opt.image_size // 0.95)
    transform = transforms.Compose([
        transforms.Resize(size=image_size_unc, interpolation=transforms.InterpolationMode.BICUBIC, max_size=None, antialias=None),
        transforms.RandomCrop(opt.image_size),
        transforms.ToTensor(),
        transforms.RandomHorizontalFlip(p=0.5)
    ])

    # Augment the original image and label batch_size times
    for i in range(opt.ttt_batch_size):
        # Apply the transformation pipeline
        transformed_image = transform(image)

        # Randomly rotate the transformed image and create a new label
        rotated_image, rotation_label = random_rotate(transformed_image)

        # Add the rotated image and label to the list
        augmented_images.append(rotated_image)
        rotation_labels.append(rotation_label)

    # Convert the list of augmented images and labels to tensors
    augmented_images = torch.stack(augmented_images)
    rotation_labels = torch.stack(rotation_labels)

    return augmented_images, rotation_labels


def random_rotate(image):
    # Randomly rotate the image by 0, 90, 180, or 270 degrees
    angles = [0, 90, 180, 270]
    angle = random.choice(angles)
    rotated_image = transforms.functional.rotate(image, angle)
    rotated_label = torch.tensor(angle // 90)  # Create a new label based on the chosen angle

    return rotated_image, rotated_label