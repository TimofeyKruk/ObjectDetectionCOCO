import torch
import torchvision
import torch.nn as nn


def conv_layer(in_dim, out_dim, filter_size, stride, padding, max_pool=False, bias=False, leaky_parameter=0.1,
               inplace=True):
    if max_pool is True:
        return nn.Sequential(nn.Conv2d(in_dim, out_dim, filter_size, stride, padding, bias),
                             nn.BatchNorm2d(out_dim),
                             nn.LeakyReLU(leaky_parameter, inplace),
                             nn.MaxPool2d(2, 2))
    else:
        return nn.Sequential(nn.Conv2d(in_dim, out_dim, filter_size, stride, padding, bias),
                             nn.BatchNorm2d(out_dim),
                             nn.LeakyReLU(leaky_parameter, inplace))


class modelYOLO(nn.Module):
    def __init__(self, num_classes, anchors=[(1.3221, 1.73145),
                                             (3.19275, 4.00944),
                                             (5.05587, 8.09892),
                                             (9.47112, 4.84053),
                                             (11.2364, 10.0071)]) -> None:
        super(modelYOLO, self).__init__()

        self.num_classes = num_classes
        self.anchors = anchors

        # Part before residual:
        self.conv1 = conv_layer(3, 32, 3, 1, 1, max_pool=True)
        self.conv2 = conv_layer(32, 64, 3, 1, 1, max_pool=True)
        self.conv3 = conv_layer(64, 128, 3, 1, 1)
        self.conv4 = conv_layer(128, 64, 1, 1, 0)
        self.conv5 = conv_layer(64, 128, 3, 1, 1, max_pool=True)
        self.conv6 = conv_layer(128, 256, 3, 1, 1)
        self.conv7 = conv_layer(256, 128, 1, 1, 0)
        self.conv8 = conv_layer(128, 256, 3, 1, 1, max_pool=True)
        self.conv9 = conv_layer(256, 512, 3, 1, 1)
        self.conv10 = conv_layer(512, 256, 1, 1, 0)
        self.conv11 = conv_layer(256, 512, 3, 1, 1)
        self.conv12 = conv_layer(512, 256, 1, 1, 0)
        self.conv13 = conv_layer(256, 512, 3, 1, 1)

        # Saving here high-resolution features for future
        self.p2maxpool = nn.MaxPool2d(2, 2)
        self.p2conv1 = conv_layer(512, 1024, 3, 1, 1)
        self.p2conv2 = conv_layer(1024, 512, 1, 1, 0)
        self.p2conv3 = conv_layer(512, 1024, 3, 1, 1)
        self.p2conv4 = conv_layer(1024, 512, 1, 1, 0)
        self.p2conv5 = conv_layer(512, 1024, 3, 1, 1)
        self.p2conv6 = conv_layer(1024, 1024, 3, 1, 1)
        self.p2conv7 = conv_layer(1024, 1024, 3, 1, 1)

        # Convolution for residual
        self.residual_conv1 = conv_layer(512, 64, 1, 1, 0)

        # Summing residual and previous layer output
        self.p3conv1 = conv_layer(256 + 1024, 1024, 3, 1, 1)

        self.p3conv2 = nn.Conv2d(1024, len(self.anchors) * (5 + self.num_classes), 1, 1, 0, bias=False)

    def forward(self, x):
        # First part before residual saving (feature extraction)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.conv6(x)
        x = self.conv7(x)
        x = self.conv8(x)
        x = self.conv9(x)
        x = self.conv10(x)
        x = self.conv11(x)
        x = self.conv12(x)
        x = self.conv13(x)

        # Saving residual
        residual = x

        # Part 2
        output = self.p2maxpool(x)
        output = self.p2conv1(output)
        output = self.p2conv2(output)
        output = self.p2conv3(output)
        output = self.p2conv4(output)
        output = self.p2conv5(output)
        output = self.p2conv6(output)
        output = self.p2conv7(output)

        # "Shrinking" residual
        output2 = self.residual_conv1(residual)
        # Reshaping the results
        batch_size, channels, h, w = output2.data.size()
        output2 = output2.view(batch_size, int(channels / 4), h, 2, w, 2).contiguous()
        output2 = output2.permute(0, 3, 5, 1, 2, 4).contiguous()
        output2 = output2.view(batch_size, -1, int(h / 2), int(w / 2))

        # Part 3. Uniting residual and output
        output = torch.cat((output, output2), dim=1)
        output = self.p3conv1(output)
        output = self.p3conv2(output)
        return output


def train_model(model_yolo, train, test, PATH, tensorboard, epoch=10, cuda=True, save=True) -> modelYOLO:
    '''
    :param model_yolo: object of class modelYOLO
    :param train: train_loader <- data to train the model
    :param test:test_loader <- data to test on
    :param PATH: string <- directory to save model
    :param epoch: int <- number of epoch for training
    :param cuda: bool <- whether to train on GPU or not
    :param save:bool <- variable for saving model weights or not
    :return: model_yolo: modelYOLO object <- trained model
    '''

    if cuda is True and torch.cuda.is_available() is True:
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")
    print("Device: ", device)

    return model_yolo


if __name__ == '__main__':
    print(conv_layer(32, 64, 3, 1, 0, True))
