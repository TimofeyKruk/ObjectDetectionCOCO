"""
Created on Thu Oct 26 11:06:51 2017
@author: Utku Ozbulak - github.com/utkuozbulak
"""
from PIL import Image
import numpy as np
import torch
import small.small.yolo as yolo
from small.small import data_preparation
import matplotlib.pyplot as plt

from small.small.misc_functions import get_example_params, save_class_activation_images


class CamExtractor():
    """
        Extracts cam features from the model
    """

    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None

    def save_gradient(self, grad):
        self.gradients = grad

    def forward_pass(self, x):
        # First part before residual saving (feature extraction)
        x = self.model.conv1(x)
        x = self.model.conv2(x)
        x = self.model.conv3(x)
        x = self.model.conv4(x)
        x = self.model.conv5(x)
        x = self.model.conv6(x)
        x = self.model.conv7(x)
        x = self.model.conv8(x)
        x = self.model.conv9(x)
        x = self.model.conv10(x)
        x = self.model.conv11(x)
        x = self.model.conv12(x)
        x = self.model.conv13(x)

        # HOOK Registration
        saved_convolution = x
        x.register_hook(self.save_gradient)

        # Saving residual
        residual = x

        # Part 2
        output = self.model.p2maxpool(x)
        output = self.model.p2conv1(output)
        output = self.model.p2conv2(output)
        output = self.model.p2conv3(output)
        output = self.model.p2conv4(output)
        output = self.model.p2conv5(output)
        output = self.model.p2conv6(output)
        output = self.model.p2conv7(output)

        # "Shrinking" residual
        output2 = self.model.residual_conv1(residual)
        # Reshaping the results
        batch_size, channels, h, w = output2.data.size()
        output2 = output2.view(batch_size, int(channels / 4), h, 2, w, 2).contiguous()
        output2 = output2.permute(0, 3, 5, 1, 2, 4).contiguous()
        output2 = output2.view(batch_size, -1, int(h / 2), int(w / 2))

        # Part 3. Uniting residual and output
        output = torch.cat((output, output2), dim=1)
        output = self.model.p3conv1(output)

        output = self.model.p3conv2(output)

        return saved_convolution, output


class GradCam():
    """
        Produces class activation map
    """

    def __init__(self, model, target_layer):
        self.model = model
        self.model.eval()
        # Define extractor
        self.extractor = CamExtractor(self.model, target_layer)

    def generate_cam(self, input_image, target_class="person"):
        # Full forward pass

        conv_output, model_output = self.extractor.forward_pass(input_image)

        my_dict = {"person": 0,
                   "car": 1,
                   "bird": 2,
                   "cat": 3,
                   "dog": 4}
        target_class = my_dict[target_class]
        # Target for backprop
        one_hot_output = torch.FloatTensor(model_output.size()).zero_()
        for anchor in range(5):
            one_hot_output[0, anchor * 10 + 5 + target_class, :, :] = 1

        # Zero grads
        self.model.zero_grad()

        # Backward pass with specified target
        model_output.backward(gradient=one_hot_output, retain_graph=True)

        # Get hooked gradients
        guided_gradients = self.extractor.gradients.data.numpy()[0]

        # Get convolution outputs
        target = conv_output.data.numpy()[0]

        # Get weights from gradients
        weights = np.mean(guided_gradients, axis=(1, 2))

        # Create empty numpy array for cam
        cam = np.ones(target.shape[1:], dtype=np.float32)

        # Multiply each weight with its conv output and then, sum
        for i, w in enumerate(weights):
            cam += w * target[i, :, :]
        # ReLU
        cam = np.maximum(cam, 0)
        # Normalize
        cam = (cam - np.min(cam)) / (np.max(cam) - np.min(cam))
        cam = np.uint8(cam * 255)  # Scale between 0-255 to visualize
        cam = np.uint8(Image.fromarray(cam).resize((input_image.shape[2],
                                                    input_image.shape[3]), Image.ANTIALIAS)) / 255
        return cam


def load_model(PATH, class_number=95):
    model = yolo.modelYOLO(num_classes=class_number)
    model.load_state_dict(torch.load(PATH))
    print("Model was loaded from file: ", PATH)
    return model


if __name__ == '__main__':
    PATH = "F:\WORK_Oxagile\INTERN\ImageSegmentation\small\SMALL_SavedModelWeights6_after15_after20_after30_after35_after44_after60_after70"
    num_classes = 5
    batch_size = 1
    img_size_transform = 32 * 13

    # Loading model from memory
    model = load_model(PATH, num_classes)

    dataset = data_preparation.loadCOCO("F:\WORK_Oxagile\INTERN\Datasets\COCO\\", img_size_transform,
                                        train_bool=False,
                                        shuffle_test=False,
                                        batch_size=batch_size)

    for b, data in enumerate(dataset):
        images, targets = data
        file_name_to_export = "Heatmap" + str(b) + ".jpg"
        # Grad cam
        grad_cam = GradCam(model,
                           target_layer=13)

        # Generate cam mask
        cam = grad_cam.generate_cam(images,
                                    target_class="dog")
        plt.imshow(cam)
        plt.show()

        # Save mask
        # save_class_activation_images(images[0], cam, file_name_to_export)
        print('Grad cam calculated for one image!')

        break
