import torch
import torch.nn as nn


def double_conv(in_c, out_c):
    conv = nn.Sequential(
        nn.Conv2d(in_c, out_c, kernel_size=3, padding=1),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_c, out_c, kernel_size=3, padding=1),
        nn.ReLU(inplace=True),
    )
    return conv


def crop_img(tensor, target_tensor):
    target_size = target_tensor.size()[2]
    tensor_size = tensor.size()[2]
    delta = tensor_size - target_size
    delta = delta // 2
    return tensor[:, :, delta : target_size + delta, delta : target_size + delta]


def channel_conversion_pairs(in_channels: int, depth: int):
    out_channels = 64
    pairs = []
    for i in range(depth):
        pairs.append((in_channels, out_channels))
        in_channels = out_channels
        out_channels *= 2
    return pairs


class UNet(nn.Module):
    def __init__(self, in_channels: int = 1, out_channels: int = 1, depth: int = 5):
        super(UNet, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.depth = depth

        down_channels = channel_conversion_pairs(in_channels, depth)
        # Last 64->out_channels convolution is done separately by the "out" layer
        up_channels = list(reversed(down_channels[1:]))

        self.max_pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.down_conv = nn.ModuleList(
            [double_conv(up, down) for (up, down) in down_channels]
        )
        self.up_conv = nn.ModuleList(
            [double_conv(down, up) for (up, down) in up_channels]
        )
        self.up_trans = nn.ModuleList(
            [
                nn.ConvTranspose2d(down, up, kernel_size=2, stride=2)
                for (up, down) in up_channels
            ]
        )
        self.out = nn.Conv2d(64, self.out_channels, kernel_size=1) 

    def forward(self, image):
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
            # Not needed when using "same" padding
            # y = crop_img(y, x)
            x = torch.cat([y, x], 1)
            x = up_conv(x)
        return self.out(x)

    def predict(self, image, threshold=0.2):
        confidence = self.forward(image)
        return confidence > threshold
