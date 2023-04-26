import torch
import torchvision.transforms as transforms
import argparse
import numpy as np

from torch.utils.data import DataLoader
from tqdm import tqdm

from utils.rotation_net import RotationNet
from datasets import RotationDataset


parser = argparse.ArgumentParser()
#todo: write help statements

# Dataset and paths
parser.add_argument('--dataset', type=str, default='CUB', help='CUB')
parser.add_argument('--backbone_path', type=str, default='/mnt/qb/work/akata/jstrueber72/ZSTTT/data/CUB/resnet50_cub')
parser.add_argument('--data_path', type=str, default='/mnt/qb/akata/jstrueber72/datasets/CUB/')
parser.add_argument('--splitdir', type=str, default='/mnt/qb/work/akata/jstrueber72/ZSTTT/data/CUB/')
parser.add_argument('--class_txt', type=str, default='trainvalclasses.txt')
parser.add_argument('--attribute_path', type=str, default='/mnt/qb/akata/jstrueber72/datasets/CUB/attributes/class_attribute_labels_continuous.txt')

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
    dataset = RotationDataset(opt.data_path, opt.splitdir, opt.class_txt, transform=transform)
    loader = DataLoader(dataset, batch_size=opt.batch_size, shuffle=True, num_workers=4)

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

    return features, labels


def get_zsl_data_collection(opt):
    print("Start data collection.")

    # Construct class to attribute mapping
    attribute_mapping = {}
    with open(opt.attribute_path, 'r') as f:
        attributes = f.readlines()
        for idx, attr_raw in enumerate(attributes):
            attribute = np.fromstring(attr_raw, dtype=float, sep=' ')
            attribute_mapping[float(idx)] = attribute

    # Initialize all data entries
    data = {}
    #data['images'] = np.array([])
    data['features'] = np.array([])
    data['labels'] = np.array([])
    data['trainval_loc'] = np.array([])
    data['train_loc'] = np.array([])
    data['val_loc'] = np.array([])
    data['test_seen_loc'] = np.array([])
    data['test_unseen_loc'] = np.array([])
    data['attributes'] = np.array([])

    # Classes used for training
    print("Collecting training data")
    opt.class_txt = 'trainclasses1.txt'
    train_features, train_labels = extract_features(opt)
    #data['images'] = train_images
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
    valid_features, valid_labels = extract_features(opt)
    #data['images'] = np.concatenate([data['images'], valid_images])
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
    test_features, test_labels = extract_features(opt)
    #data['images'] = np.concatenate([data['images'], test_images])
    data['features'] = np.concatenate([data['features'], test_features])
    data['labels'] = np.concatenate([data['labels'], test_labels])
    start_test_loc = end_val_loc
    end_test_loc    = start_test_loc + len(test_features)
    test_idx = np.arange(start_test_loc, end_test_loc)
    data['test_unseen_loc'] = test_idx
    print("Done")

    # Collect the attribute vector for each sample
    #data['attributes'] = np.take(list(attribute_mapping.values()), data['labels'].astype(int), axis=0)
    data['attributes'] = np.array([attribute_mapping[label] for label in data['labels']])

    return data


if __name__ == '__main__':
    opt = parser.parse_args()

    data = get_zsl_data_collection(opt)

    for key, value in data.items():
        print(f"Entry: {key} with shape {value.shape}")