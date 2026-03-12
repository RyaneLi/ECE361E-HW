import argparse
import numpy as np
import torch
import random
import torchvision.datasets as dsets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset

from onnxruntime.quantization import (
    CalibrationDataReader,
    QuantFormat,
    QuantType,
    quantize_static,
)
import onnxruntime as ort

parser = argparse.ArgumentParser(description='ECE361E HW4 Quantization')
# TODO add argument for ONNX model
parser.add_argument('--onnx_model', type=str, required=True, help='Path to floating-point ONNX model')
parser.add_argument('--output_model', type=str, default='MobilenetV1_int8.onnx',
                    help='Path to save quantized INT8 ONNX model')
parser.add_argument('--calib_batch_size', type=int, default=128, help='Batch size for calibration dataloader')
args = parser.parse_args()

# Each experiment you will do will have slightly different results due to the randomness
# of 1. the initialization value for the weights of the model, 2. sampling batches of training data
# 3. numerical algorithms for computation (in CUDA.) In order to have reproducible results,
# we have fixed a random seed to a specific value such that we "control" the randomness.
random_seed = 1
torch.manual_seed(random_seed)
random.seed(random_seed)
np.random.seed(random_seed)
torch.cuda.manual_seed(random_seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
g = torch.Generator()
g.manual_seed(random_seed) # for data loader shuffling

# TODO add CIFAR10 train dataset

train_dataset = dsets.CIFAR10(
    root='data',
    train=True,
    transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.4914, 0.4822, 0.4465),
                             std=(0.2023, 0.1994, 0.2010)),
    ]),
    download=True,
)

num_calib_images = 1000
indices = list(range(num_calib_images))
calib_dataset = Subset(train_dataset, indices)
calib_loader = DataLoader(
    calib_dataset,
    batch_size=args.calib_batch_size,
    shuffle=False,
    generator=g,
)

# TODO add CIFAR10 Calibration Data Reader

class CIFAR10CalibrationDataReader(CalibrationDataReader):
    def __init__(self, dataloader, input_name, max_samples=None):
        self.dataloader = dataloader
        self.input_name = input_name
        self.max_samples = max_samples
        self._enumerator = None

    def get_next(self):
        if self._enumerator is None:
            self._enumerator = self._data_generator()
        return next(self._enumerator, None)

    def _data_generator(self):
        sample_count = 0
        for images, _ in self.dataloader:
            batch_size = images.size(0)
            for i in range(batch_size):
                if self.max_samples is not None and sample_count >= self.max_samples:
                    return
                # Feed one image at a time so the batch dimension is 1,
                # matching the exported ONNX model's input shape (1, 3, 32, 32).
                single_image = images[i:i + 1]  # shape: (1, 3, 32, 32)
                yield {self.input_name: single_image.numpy()}
                sample_count += 1

# TODO Preprocess model for quantization

session = ort.InferenceSession(args.onnx_model, providers=['CPUExecutionProvider'])
input_name = session.get_inputs()[0].name

# TODO Use 1,000 images from the CIFAR10 Calibration Data Reader

calib_data_reader = CIFAR10CalibrationDataReader(
    calib_loader,
    input_name=input_name,
    max_samples=num_calib_images,
)

# TODO Perform static quantization

quantize_static(
    model_input=args.onnx_model,
    model_output=args.output_model,
    calibration_data_reader=calib_data_reader,
    quant_format=QuantFormat.QDQ,
    per_channel=True,
    weight_type=QuantType.QInt8,
    activation_type=QuantType.QInt8,
)

print(f'Successfully quantized model saved to {args.output_model}')

