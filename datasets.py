import os
import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset
from PIL import Image


class RotationDataset(Dataset):
    def __init__(self, root_dir, classes='trainvalclasses.txt', transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.images = []
        self.labels = []
        self.class_names = []
        self.class_map = {}
        self.num_classes = 0
        self.rotation_angles = [0, 90, 180, 270]  # Discrete rotation angles

        # Read class names and assign unique IDs to each class
        with open(os.path.join(root_dir, classes), 'r') as f:
            lines = f.readlines()
            for line in lines:
                full_class_name = line.strip()
                self.class_names.append(full_class_name)
                parts = line.strip().split('.')
                class_idx = parts[0]
                class_name = parts[1]
                self.class_map[class_name] = class_idx
                self.num_classes += 1

        # Read image filenames and labels
        for class_name in self.class_names:
            class_directory = os.path.join(root_dir, 'images', class_name)
            class_images = [os.path.join(class_directory, f) for f in os.listdir(class_directory)
                            if os.path.isfile(os.path.join(class_directory, f))]
            self.images += class_images
            short_class_name = class_name.split('.')[1]
            class_labels = len(class_images) * self.class_map[short_class_name]
            self.label += class_labels

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image_path = os.path.join(self.root_dir, 'images', self.images[idx])
        image = Image.open(image_path).convert('RGB')
        label = self.labels[idx]

        rotation_label = torch.randint(4)
        rotation_angle = self.rotation_angles[rotation_label]

        # Apply the rotation to the image
        rotated_image = transforms.functional.rotate(image, rotation_angle)

        if self.transform is not None:
            image = self.transform(image)
            rotated_image = self.transform(rotated_image)

        return image, rotated_image, label, rotation_angle