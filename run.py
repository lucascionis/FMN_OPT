import pickle

import torch
import torchvision
from robustbench.utils import load_model as rb_load_model

from src.fmn_opt import FMNOpt


def splitting_pkl_name(filename):
    splits = filename.split('_')
    splits.remove('cifar10')
    optimizer, scheduler, loss = splits[-3:]
    loss = loss.split('.')[0]

    model = '_'.join(splits[:-3])
    return model, optimizer, scheduler, loss

# Load optimizer, scheduler configurations
fmn_config_path = 'configs/Wang2023Better_WRN-70-16_cifar10_SGD_CosineAnnealingLR_LL.pkl'
pkl_filename = fmn_config_path.split('/')[-1]
model_name, optimizer, scheduler, loss = splitting_pkl_name(pkl_filename)

model = rb_load_model(
    model_dir="./Models/pretrained",
    model_name=model_name,
    dataset='cifar10',
    norm='Linf'
    )
dataset = torchvision.datasets.CIFAR10('./Models/data',
                                       train=False,
                                       download=True,
                                       transform=torchvision.transforms.ToTensor())

steps = 50

# load fmn pkl config file
try:
    with open(fmn_config_path, 'rb') as file:
        fmn_config = pickle.load(file)
except Exception as e:
    print("Cannot load the configuration:")
    print(fmn_config_path)
    exit(1)

optimizer_config = fmn_config['best_config']['opt_s']
scheduler_config = fmn_config['best_config']['sch_s']

if scheduler == 'MultiStepLR':
    milestones = len(scheduler_config['milestones'])
    scheduler_config['milestones'] = np.linspace(0, steps, milestones)

if scheduler == 'CosineAnnealingLR':
    scheduler_config['T_max'] = steps

if scheduler == 'CosineAnnealingWarmRestarts':
    scheduler_config['T_0'] = steps//2


# Instantiate the attack
torch.cuda.empty_cache()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


fmn_opt = FMNOpt(
    model=model.eval().to(device),
    dataset=dataset,
    norm = 'inf',
    steps = steps,
    batch_size=40,
    batch_number=40,
    optimizer=optimizer,
    scheduler=scheduler,
    optimizer_config=optimizer_config,
    scheduler_config=scheduler_config,
    device=device
    )


fmn_opt.run()