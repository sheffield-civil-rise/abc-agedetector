import torch
import torchvision
import warnings


class CustomModule(torch.nn.Module):
    """ This is a superclass for custom network classes """
    def __init__(self):
        super(CustomModule, self).__init__()


class CombinePredictions(CustomModule):
    """ Custom module to combine predictions of patches from batch patchnet"""
    def __init__(self):
        super(CombinePredictions, self).__init__()

    def forward(self, x):
        ''' maybe correct this if mean is not appropriate

            essentially averages class predictions for specific patches to
            create single guess
        '''
        return torch.mean(x, axis=-2)


class BaseConvNet(CustomModule):
    """ Basic convolutional network for image classification """
    def __init__(self, numclasses=7, **kwargs):
        super(BaseConvNet, self).__init__()
        self._numclasses = numclasses
        self._generate_layers()

    def _generate_layers(self):
        ''' create layers '''
        self._convlayers()
        self._fclayers()

    def _convlayers(self):
        _conv1 = torch.nn.Conv2d(3, 6, 5)
        _pool = torch.nn.MaxPool2d(2, 2)
        _conv2 = torch.nn.Conv2d(6, 16, 5)

        self.conv = torch.nn.Sequential(
            _conv1,
            torch.nn.ReLU(),
            _pool,
            _conv2,
            torch.nn.ReLU(),
            _pool)

    def _fclayers(self):
        _fc1 = torch.nn.Linear(16 * 5 * 5, 120)
        _fc2 = torch.nn.Linear(120, 84)
        _fc3 = torch.nn.Linear(84, self._numclasses)
        self.fc = torch.nn.Sequential(
            _fc1,
            torch.nn.ReLU(),
            _fc2,
            torch.nn.ReLU(),
            _fc3)

    def forward(self, x):
        ''' forward pass '''
        x = self.conv(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


class AgeDetector(torch.nn.Module):
    def __init__(
        self,
        numclasses=7,
        backbone=torchvision.models.resnet18,
        pretrained=False,
        fixbackbone=False):
        """
        this uses a segmentation-based backbone to calculate age of patches
        """
        super(AgeDetector, self).__init__()
        self._numclasses = numclasses
        self._pretrained = pretrained
        self._fixbackbone = fixbackbone
        self._generate_backbone(backbone)
        self._generate_outlayer()
        self._generate_prediction_layer()

    def _generate_backbone(self, backbone):
        ''' creates backbone network '''
        if self._fixbackbone and not self._pretrained:
            warnings.warning(
                'fix_backbone requested so enabling pretrained weights by default')
        self._backbone = backbone(
            pretrained=(self._pretrained or self._fixbackbone))
        if not isinstance(self._backbone, CustomModule):
            self._backbone_out_features = self._backbone.fc.in_features
        if self._fixbackbone:
            for param in self._backbone.parameters():
                param.requires_grad = False

    def _generate_outlayer(self):
        ''' creates output layer for classification '''
        if not isinstance(self._backbone, CustomModule):
            self._backbone.fc = torch.nn.Linear(
                self._backbone_out_features, self._numclasses)

    def _generate_prediction_layer(self):
        ''' create softmax prediction layer '''
        self._predict = CombinePredictions()

    def forward(self, x):
        ''' forward transform '''
        in_shape = x.shape
        nb_batch, nb_patch = in_shape[0], in_shape[1]

        x = torch.flatten(x, start_dim=0, end_dim=1)
        x = self._backbone(x)
        x = torch.reshape(x, (nb_batch, nb_patch, -1))
        x = self._predict(x)
        return x
