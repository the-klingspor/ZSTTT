import torch
import os

from PIL import Image
from tqdm import tqdm


def ttt_epoch(ttt_method, model, data, opt):
    # do TTT and extract features
    ttt_train_features = []
    ttt_losses = []
    # Apply rotation TTT to each image
    pbar = tqdm(data.images, total=len(data.images), desc="TTT Run")

    for img_name in pbar:
        image_path = os.path.join(opt.data_path, 'images', img_name)
        image = Image.open(image_path).convert('RGB')
        feature, losses = ttt_method(model, image, opt)
        ttt_train_features.append(feature)
        ttt_losses.append(losses)

        pbar.set_postfix(loss=losses.mean())

    ttt_train_features = torch.stack(ttt_train_features).squeeze()
    ttt_losses = torch.stack(ttt_losses)

    return ttt_train_features, ttt_losses
