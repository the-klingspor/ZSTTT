import torch
import torchvision.transforms as transforms
import argparse
import numpy as np
import os

from torch.utils.data import DataLoader
from tqdm import tqdm

from utils.rotation_net import RotationNet
from datasets import RotationDataset


parser = argparse.ArgumentParser()
#todo: write help statements

# Dataset and paths
parser.add_argument('--dataset', type=str, default='CUB', help='CUB')
parser.add_argument('--backbone_path', type=str, default='/mnt/qb/work/akata/jstrueber72/ZSTTT/data/CUB/resnet50_cub.pth')
parser.add_argument('--data_path', type=str, default='/mnt/qb/akata/jstrueber72/datasets/CUB/')
parser.add_argument('--splitdir', type=str, default='/mnt/qb/work/akata/jstrueber72/ZSTTT/data/CUB/')
parser.add_argument('--class_txt', type=str, default='trainvalclasses.txt')
parser.add_argument('--attribute_path', type=str, default='/mnt/qb/akata/jstrueber72/datasets/CUB/attributes/class_attribute_labels_continuous.txt')
parser.add_argument('--include_txt', type=str, default=None)

# Hyper parameters
parser.add_argument('--image_size', type=int, default=336)
parser.add_argument('--architecture', type=str, default="resnet50")
parser.add_argument('--batch_size', type=int, default=64, help='input batch size')

def extract_features(opt, model=None):
    # Hyperparameters
    image_size_unc  = int(opt.image_size // 0.95)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Define the transforms to apply to the images
    # ToDo: automatically choose transformation used by timm

    transform = transforms.Compose([
        transforms.Resize(size=image_size_unc, interpolation=transforms.InterpolationMode.BICUBIC, max_size=None, antialias=None),
        transforms.CenterCrop(size=opt.image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    # Create the data loaders for training and validation
    dataset = RotationDataset(opt.data_path, opt.splitdir, opt.class_txt, transform=transform, include_txt=opt.include_txt)
    loader = DataLoader(dataset, batch_size=opt.batch_size, shuffle=False, num_workers=4)

    # Load the pre-trained ResNet101 model from timm with additional rotation head
    if model == None:
        model = RotationNet(num_classes=dataset.num_classes, architecture=opt.architecture)
        model.load_state_dict(torch.load(opt.backbone_path))

    model.to(device)
    model.eval()

    # Initialize the progress bar
    pbar = tqdm(enumerate(loader), total=len(loader), unit="batch")

    features = np.zeros((len(dataset), 2048))
    labels = np.zeros(len(dataset))

    # Extract features for the each batch of images
    start_idx = 0
    for bach_idx, data in pbar:
        images, _, batch_labels, _ = data

        # Move the data to the device
        images = images.to(device)

        # Forward pass
        with torch.no_grad():
            batch_features = model.forward_features(images).cpu().numpy()

        # Stack the features and labels into the output arrays
        end_idx = start_idx + len(images)
        features[start_idx:end_idx, :] = batch_features
        labels[start_idx:end_idx] = batch_labels.numpy()

        # Increment the starting index for the next batch
        start_idx = end_idx

    return features, labels, dataset.images


def get_zsl_data_collection(opt):
    print("Start data collection.")

    # Initialize all data entries
    data = {}
    data['images'] = np.array([])
    data['features'] = np.array([])
    data['labels'] = np.array([])
    data['trainval_loc'] = np.array([])
    data['train_loc'] = np.array([])
    data['val_loc'] = np.array([])
    data['test_seen_loc'] = np.array([])
    data['test_unseen_loc'] = np.array([])

    # Construct class to attribute mapping
   # with open(opt.attribute_path, 'r') as f:
   #     raw_attributes = f.readlines()

    #attributes = [np.fromstring(attr, dtype=float, sep=' ') for attr in raw_attributes]
    #data['attributes'] = np.stack(attributes)

    # Classes used for training
    print("Collecting training data")
    opt.class_txt = 'trainclasses1.txt'
    if opt.include_txt:
        opt.include_txt = os.path.join(opt.data_path, 'unseen_train.txt')
    train_features, train_labels, train_images = extract_features(opt)
    data['images'] = train_images
    data['features'] = train_features
    data['labels'] = train_labels
    end_train_loc = len(train_features)
    train_idx = np.arange(end_train_loc)
    data['trainval_loc'] = train_idx
    data['train_loc'] = train_idx
    data['test_seen_loc'] = train_idx
    print("Done")

    # Classes used for validation
    print("Collecting validation data")
    opt.class_txt = 'valclasses1.txt'
    valid_features, valid_labels, valid_images = extract_features(opt)
    data['images'] = np.concatenate([data['images'], valid_images])
    data['features'] = np.concatenate([data['features'], valid_features])
    data['labels'] = np.concatenate([data['labels'], valid_labels])
    start_val_loc = end_train_loc
    end_val_loc    = start_val_loc + len(valid_features)
    val_idx = np.arange(start_val_loc, end_val_loc)
    data['trainval_loc'] = np.concatenate([data['trainval_loc'], val_idx])
    data['val_loc'] = val_idx
    data['test_seen_loc'] = np.concatenate([data['test_seen_loc'], val_idx])
    print("Done")

    # Classes used for testing
    print("Collecting test data")
    opt.class_txt = 'testclasses.txt'
    opt.include_txt = None
    test_features, test_labels, test_images = extract_features(opt)
    data['images'] = np.concatenate([data['images'], test_images])
    data['features'] = np.concatenate([data['features'], test_features])
    data['labels'] = np.concatenate([data['labels'], test_labels])
    start_test_loc = end_val_loc
    end_test_loc    = start_test_loc + len(test_features)
    test_idx = np.arange(start_test_loc, end_test_loc)
    data['test_unseen_loc'] = test_idx
    print("Done")

    return data


if __name__ == '__main__':
    opt = parser.parse_args()

    data = get_zsl_data_collection(opt)

    for key, value in data.items():
        print(f"Entry: {key} with shape {len(value)}")