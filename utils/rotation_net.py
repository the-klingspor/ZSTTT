import timm
import torch


class RotationNet(torch.nn.Module):
    def __init__(self, num_classes, num_rotation_classes=4, architecture='resnet101'):
        super().__init__()

        # Load the pre-trained ResNet101 model from timm
        self.model = timm.create_model(architecture, pretrained=True, num_classes=0)

        # Modify the model to add a second output head for rotation prediction
        in_features = self.model.feature_info[-1]["num_chs"]
        self.fc_classification = torch.nn.Linear(in_features, num_classes)
        self.fc_rotation = torch.nn.Linear(in_features, num_rotation_classes)

    def forward(self, inputs):
        x = self.model(inputs)

        classification_output = self.fc_classification(x)
        rotation_output = self.fc_rotation(x)

        return classification_output, rotation_output

    def forward_features(self, inputs):
        x = self.model(inputs)

        return x

# test code:
if __name__ == '__main__':
    model = RotationNet(5)

    # Define the batch size and number of channels
    batch_size = 8
    num_channels = 3

    # Define the input dimensions
    input_height = 224
    input_width = 224

    # Generate a random input tensor with the specified dimensions
    inputs = torch.randn((batch_size, num_channels, input_height, input_width))

    cls, rot = model(inputs)
    features = model.forward_features(inputs)

    print(torch.argmax(cls, dim=-1))
    print(torch.argmax(rot, dim=-1))
    print(features.shape)