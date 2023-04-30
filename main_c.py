import utils
import consts
import models
import torch
import torchvision.transforms as transforms
import random
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme()

torch.manual_seed(consts.SEED)
random.seed(consts.SEED)
np.random.seed(consts.SEED)

# GPU available?
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# load model and dataset
model = utils.load_pretrained_cnn(1).to(device)
model.eval()
dataset = utils.TMLDataset(transform=transforms.ToTensor())
data_loader = torch.utils.data.DataLoader(dataset, batch_size=consts.BATCH_SIZE)

# model accuracy
acc_orig = utils.compute_accuracy(model, data_loader, device)
print(f'Model accuracy before flipping: {acc_orig:0.4f}')

# layers whose weights will be flipped
layers = {'conv1': model.conv1,
          'conv2': model.conv2,
          'fc1':   model.fc1,
          'fc2':   model.fc2,
          'fc3':   model.fc3}

# flip bits at random and measure impact on accuracy (via RAD)
RADs_bf_idx = dict([(bf_idx, []) for bf_idx in range(32)]) # will contain a list of RADs for each index of bit flipped
RADs_all = []  # will eventually contain all consts.BF_PER_LAYER*len(layers) RADs
for layer_name in layers:
    layer = layers[layer_name]
    with torch.no_grad():
        W = layer.weight
        W.requires_grad = False

        if len(W.shape) == 4:
            n, m, p, c = W.shape
            indexes = [(i, j, k, l) for i in range(n) for j in range(m) for k in range(p) for l in range(c)]
        elif len(W.shape) == 2:
            n, m = W.shape
            indexes = [(i, j) for i in range(n) for j in range(m)]
        else:
            raise NotImplementedError

        sample_indexes = random.sample(indexes, min(len(indexes), consts.BF_PER_LAYER))

        # for _ in range(consts.BF_PER_LAYER):
        for weight_index in sample_indexes:
            # FILL ME: flip a random bit in a randomly picked weight, measure RAD, and restore weight
            original_weight = W[weight_index].detach().clone()

            flipped_weight, bit_index = utils.random_bit_flip(W[weight_index])
            W[weight_index] = flipped_weight

            acc_after_flip = utils.compute_accuracy(model, data_loader, device)

            rad = (acc_orig - acc_after_flip) / acc_orig

            W[weight_index] = original_weight

            RADs_bf_idx[bit_index].append(rad)
            RADs_all.append(rad)

# Max and % RAD>10%
RADs_all = np.array(RADs_all)
print(f'Total # weights flipped: {len(RADs_all)}')
print(f'Max RAD: {np.max(RADs_all):0.4f}')
print(f'RAD>10%: {np.sum(RADs_all>0.1)/RADs_all.size:0.4f}')
            
# boxplots: bit-flip index vs. RAD
plt.figure()
plt.boxplot(list(RADs_bf_idx.values()))
plt.title('values of all bit flips')
plt.xlabel('bit index')
plt.ylabel('RAD')
plt.savefig('bf_idx-vs-RAD.jpg')
