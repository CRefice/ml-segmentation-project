import torch
import torch.nn as nn
import torch.nn.functional as F


def _double_conv(in_channels: int, out_channels: int):
    """
    Creates a layer consisting of two sequential 3x3 2D-convolutions,
    each followed by a ReLU layer.

    Arguments:
    in_channels -- the amount of channels of the layer's input tensor
    out_channels -- the desired amount of channels of the layer's output tensor
    """
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
        nn.ReLU(inplace=True),
    )


def _channel_conversion_pairs(in_channels: int, depth: int):
    """
    Creates a list of pairs of channel numbers, where each pair (in, out) at index i
    represents the number of input and output channels of the i'th layer of the U-Net.

    Arguments:
    in_channels -- the amount of channels of the input given to the first layer.
    depth -- the desired amount of convolution layers
    """
    out_channels = 64
    pairs = []
    for i in range(depth):
        pairs.append((in_channels, out_channels))
        in_channels = out_channels
        out_channels *= 2
    return pairs


class UNet(nn.Module):
    """
    A Pytorch U-Net module.
    """

    def __init__(self, in_channels: int = 1, num_classes: int = 2, depth: int = 5):
        """
        Create a new U-Net model with the given parameters.

        Arguments:
        in_channels -- The amount of channels of the input images that will be fed to the model.
                       Would be 1 for grayscale images, and 3 for RGB images. (Default: 1)
        num_classes -- The amount of classes the U-Net should predict in its segmentation.
                       Default is 2 (binary segmentation).
        depth       -- The amount of convolutional layers of the U-Net.
                       Default is 5, which is the amount used by the original authors.
        """
        super(UNet, self).__init__()

        self.in_channels = in_channels
        self.out_channels = num_classes
        if num_classes == 2:
            self.out_channels = 1
        self.depth = depth

        down_channels = _channel_conversion_pairs(in_channels, depth)
        # Last 64->out_channels convolution is done separately by the "out" layer
        up_channels = list(reversed(down_channels[1:]))

        self.max_pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.down_conv = nn.ModuleList(
            [_double_conv(up, down) for (up, down) in down_channels]
        )
        self.up_conv = nn.ModuleList(
            [_double_conv(down, up) for (up, down) in up_channels]
        )
        self.up_trans = nn.ModuleList(
            [
                nn.ConvTranspose2d(down, up, kernel_size=2, stride=2)
                for (up, down) in up_channels
            ]
        )
        self.out = nn.Conv2d(64, self.out_channels, kernel_size=1)

    def forward(self, image):
        """
        Feeds the input image through the U-Net, returning the result.

        Arguments:
        image -- A tensor of size [..., N, H, W], where N == self.num_classes

        Returns:
        A tensor of size [..., N, H, W], where N == 1 if self.num_classes == 2, or self.num_classes otherwise.
        """
        # Downward, "encoding" path
        x = self.down_conv[0](image)
        # Collect results of double convolutions in array
        # for use in the upward path
        encoded = []
        for down_conv in self.down_conv[1:]:
            encoded.append(x)
            x = self.max_pool(x)
            x = down_conv(x)

        # Upward, "decoding" path
        encoded = reversed(encoded)
        for (up_conv, up_trans, y) in zip(self.up_conv, self.up_trans, encoded):
            x = up_trans(x)
            x = torch.cat([y, x], 1)
            x = up_conv(x)
        return self.out(x)

    def predict(self, image, threshold=0.2):
        """
        A utility method to predict class labels from an input image,
        by passing the output of the network to a softmax or sigmoid function.

        Arguments:
        image -- A tensor of size [..., N, H, W], where N == self.num_classes
        threshold -- The minimum confidence level at which images are considered to be "foreground".
                     Only used if self.num_classes == 2. (Default: 0.2)

        Returns:
        A tensor of size [..., N, H, W], where N == 1 if self.num_classes == 2, or self.num_classes otherwise.
        """
        with torch.no_grad():
            outputs = self.forward(image)
            if self.out_channels > 1:
                pred = F.softmax(outputs, dim=1)
            else:
                confidence = torch.sigmoid(outputs)
                pred = confidence > threshold
            return pred.cpu().squeeze()
