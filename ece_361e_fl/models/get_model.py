from models.conv5 import Conv5, Conv5_small
from models.simplefc import SimpleFC
from models.simplecnn import SimpleCNN
from models.vgg11 import VGG11
from models.vgg16 import VGG16
from models.mobilenet import MobileNetv1


def get_model(model_name, loss_type='fedavg'):
    if model_name == "conv5":
        return Conv5(loss_type=loss_type)
    elif model_name == "conv5small":
        return Conv5_small(loss_type=loss_type)
    elif model_name == "simplefc":
        return SimpleFC(loss_type=loss_type)
    elif model_name == "simplecnn":
        return SimpleCNN(loss_type=loss_type)
    elif model_name == "vgg11":
        return VGG11(loss_type=loss_type)
    elif model_name == "vgg16":
        return VGG16(loss_type=loss_type)
    elif model_name == "mobilenet":
        return MobileNetv1(loss_type=loss_type)
    else:
        raise NotImplementedError(f'[!] ERROR: Model {model_name} not implemented yet')
