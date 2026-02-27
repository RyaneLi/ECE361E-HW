import pandas as pd
import matplotlib.pyplot as plt

# Load per‑epoch test accuracy
vgg11 = pd.read_csv('p1_q2_vgg11.csv')  # columns: epoch,test_accuracy
vgg16 = pd.read_csv('p1_q2_vgg16.csv')

plt.figure(figsize=(8, 5))

plt.plot(vgg11['epoch'], vgg11['test_accuracy'],
         label='VGG11', marker='o', linewidth=2)
plt.plot(vgg16['epoch'], vgg16['test_accuracy'],
         label='VGG16', marker='s', linewidth=2)

plt.title('VGG11 vs VGG16 Test Accuracy on CIFAR10')
plt.xlabel('Epoch')
plt.ylabel('Test accuracy [%]')
plt.grid(True)
plt.legend()

plt.tight_layout()
plt.savefig('p1_q3_vgg11_vgg16.png', dpi=300)
plt.show()