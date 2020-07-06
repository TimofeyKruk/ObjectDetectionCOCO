import torch
import torch.nn as nn
import yolo_loss


def conv_layer(in_dim, out_dim, filter_size, stride, padding, max_pool=False, bias=False, leaky_parameter=0.1,
               inplace=True):
    if max_pool is True:
        return nn.Sequential(nn.Conv2d(in_dim, out_dim, filter_size, stride, padding, bias=bias),
                             nn.BatchNorm2d(out_dim),
                             nn.LeakyReLU(leaky_parameter, inplace),
                             nn.MaxPool2d(2, 2))
    else:
        return nn.Sequential(nn.Conv2d(in_dim, out_dim, filter_size, stride, padding, bias=bias),
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


# TODO: implement tensorboard and learning rate scheduler and save model
def train_model(model, train, test, num_classes, saveName, tensorboard, epochs=10, cuda=True, save=True) -> modelYOLO:
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
    model.to(device)
    print("Device: ", device)

    criterion = yolo_loss.yoloLoss(num_classes, device=device,cuda=cuda)
    criterion=criterion.to(device)

    optimizer = torch.optim.SGD(model.parameters(), lr=0.00005, momentum=0.9, weight_decay=0.00005)

    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[7, 12, 15], gamma=0.1)

    for name, param in model.named_parameters():
        if param.device.type != 'cuda':
            print('param {}, not on GPU'.format(name),param.device.type)

    for epoch in range(epochs):
        running_loss = 0.0
        print("Epoch: ", epoch)

        for i, data in enumerate(train):
            if torch.cuda.is_available() and cuda:
                images = data[0].to(device)
                targets = [label.to(device) for label in data[1]]
                #print("___Data sent to device CUDA")
            else:
                images, targets = data[0], data[1]

            optimizer.zero_grad()

            outputs = model(images)
            #print("Variable {} is on device: {}".format("outputs",outputs.device.type))

            loss_total, loss_coordinates, loss_confidence, loss_classes = criterion(outputs, targets)

            loss_total.backward()

            optimizer.step()

            running_loss += loss_total.item()

            if (i + 1)% 25 == 0:
                tensorboard.add_scalar("train loss", running_loss / 25, epoch * 2587 + i // 25)
                print("epoch: ", epoch, "batch: ", i, "loss_total: ", running_loss / 25)
                running_loss = 0.0

    print("Last used LR: ", scheduler.get_last_lr())
    scheduler.step()

    if save is True:
        torch.save(model.state_dict(), saveName)
        print("Model was saved at file:", saveName)

    return model


if __name__ == '__main__':
    myNet = modelYOLO(80)

    print(myNet)
