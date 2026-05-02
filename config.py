import torch

BASE_DIR  = '/kaggle/input/datasets/masoudnickparvar/brain-tumor-mri-dataset/'
TRAIN_DIR = BASE_DIR + 'Training/'
TEST_DIR  = BASE_DIR + 'Testing/'
CHECKPOINT_PATH = 'checkpoints/final_weights.pth'

CLASS_NAMES = ['glioma', 'meningioma', 'notumor', 'pituitary']
num_classes = 4

resize_x = 224
resize_y = 224
input_channels = 3

batch_size = 32
epochs = 25
learning_rate = 1e-4
dropout_rate = 0.4
hidden_units = 256
val_split = 0.15
random_seed = 42
scheduler_patience = 3
scheduler_factor = 0.5

imagenet_mean = [0.485, 0.456, 0.406]
imagenet_std  = [0.229, 0.224, 0.225]

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
