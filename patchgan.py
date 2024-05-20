import torch
import math
from torch.utils.data import TensorDataset, Dataset, DataLoader
from torchvision import datasets, transforms
from torch import nn
import pennylane as qml
import matplotlib.pyplot as plt
import pandas as pd
import argparse
import json
import os

class MNISTDataset(Dataset):
    def __init__(self, image_size,digit_class):
        super().__init__()
        transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor()
        ])
        self.train_dataset = datasets.MNIST(
            root='./data',
            train=True,
            transform=transform,
            download=True
        )
        self.test_dataset = datasets.MNIST(
            root='./data',
            train=False,
            transform=transform,
            download=True
        )
        # Keep only samples with label 0
        self.train_dataset = [(img, label) for img, label in self.train_dataset if label == digit_class]
        self.test_dataset = [(img, label) for img, label in self.test_dataset if label == digit_class]

    def __len__(self):
        return len(self.train_dataset)

    def __getitem__(self, index):
        return self.train_dataset[index]

def quantum_circuit(noise, weights, n_qubits , q_depth):
    dev = qml.device("lightning.qubit", wires=n_qubits)
    @qml.qnode(dev, diff_method="parameter-shift")
    def circuit(noise, weights):
        weights = weights.reshape(q_depth, n_qubits)
        for i in range(n_qubits):
            qml.RY(noise[i], wires=i)
        for i in range(q_depth):
            for y in range(n_qubits):
                qml.RY(weights[i][y], wires=y)
            for y in range(n_qubits - 1):
                qml.CNOT(wires=[y, y + 1])
        return qml.probs(wires=list(range(n_qubits)))
    return circuit(noise, weights)

def partial_measure(noise, weights,n_qubits,n_ancillary_qubits,q_depth):
    probs = torch.tensor(quantum_circuit(noise, weights,n_qubits,q_depth)) 
    probsgiven0 = probs[:(2 ** (n_qubits- n_ancillary_qubits))]
    probsgiven0 /= torch.sum(probs)
    probsgiven = probsgiven0 / torch.max(probsgiven0)  
    return probsgiven 

class PatchQuantumGenerator(nn.Module):
    """Quantum generator class for the patch method"""

    def __init__(self, n_generators, n_qubits, q_depth, q_delta=1):
        super().__init__()
        self.q_params = nn.ParameterList(
            [
                nn.Parameter(q_delta * torch.rand(q_depth * n_qubits), requires_grad=True)
                for _ in range(n_generators)
            ]
        )
        self.n_generators = n_generators

    def forward(self, x,patch_size,n_qubits,n_ancillary_qubits,q_depth):
        images = torch.Tensor(x.size(0), 0)
        for params in self.q_params:
            patches = torch.Tensor()
            for elem in x:
                q_out = partial_measure(elem, params, n_qubits, n_ancillary_qubits, q_depth).float().unsqueeze(0)
                q_out = q_out[:, :patch_size]
                patches = torch.cat((patches, q_out))
            images = torch.cat((images, patches), 1)
        return images


class Discriminator(nn.Module):
    """Fully connected classical discriminator"""

    def __init__(self, input_size):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_size, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.model(x)


def main(args):
    with open(args.config_file) as f:
        config = json.load(f)
    
    batch_size = config[args.config_name]["batch_size"]
    image_size = config[args.config_name]["image_size"]
    n_qubits = config[args.config_name]["n_qubits"]
    patch_size = config[args.config_name]["patch_size"]
    n_a_qubits = config[args.config_name]["n_ancillary_qubits"]
    n_generators = config[args.config_name]["n_generators"]
    q_depth = config[args.config_name]["q_depth"]
    lrD = config[args.config_name]["lr_discriminator"]
    lrG = config[args.config_name]["lr_generator"]
    num_iter = config[args.config_name]["num_iterations"] 
    digit_class = args.digit_class

    data_obj = MNISTDataset(image_size=image_size, digit_class=digit_class)
    dataloader = DataLoader(data_obj.train_dataset, batch_size=batch_size, shuffle=True)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    discriminator = Discriminator(image_size * image_size).to(device)
    generator = PatchQuantumGenerator(n_generators, n_qubits,q_depth).to(device)

    criterion = nn.BCELoss()
    optD = torch.optim.SGD(discriminator.parameters(), lr=lrD)
    optG = torch.optim.SGD(generator.parameters(), lr=lrG)

    real_labels = torch.full((batch_size,), 1.0, dtype=torch.float, device=device)
    fake_labels = torch.full((batch_size,), 0.0, dtype=torch.float, device=device)
    scale_factor = 0.05
    # fixed_noise = torch.rand(batch_size, n_qubits, device=device) * math.pi / 2
    fixed_noise = torch.rand(batch_size, n_qubits, device=device) * math.pi / 2 * 0.05
    # fixed_noise = torch.randn(batch_size, n_qubits, device=device)* scale_factor


    counter = 0
    results = []
    losses = []

    # Create the parent folder
    parent_folder = f"output_qb{n_qubits}_bs{batch_size}_lrG{lrG}_lrD{lrD}"
    if not os.path.exists(parent_folder):
        os.makedirs(parent_folder)

    # Create the "model" folder within the parent folder
    model_folder = os.path.join(parent_folder, "model")
    if not os.path.exists(model_folder):
        os.makedirs(model_folder)

    # Create the "images" folder within the parent folder
    images_folder = os.path.join(parent_folder, "images")
    if not os.path.exists(images_folder):
        os.makedirs(images_folder)

    while True:
        for i, (data, _) in enumerate(dataloader):
            data = data.reshape(-1, image_size * image_size)
            real_data = data.to(device)
            noise = torch.rand(batch_size, n_qubits, device=device) * math.pi / 2
            fake_data = generator(noise, patch_size, n_qubits, n_a_qubits,q_depth)

            discriminator.zero_grad()
            outD_real = discriminator(real_data).view(-1)
            outD_fake = discriminator(fake_data.detach()).view(-1)
            errD_real = criterion(outD_real, real_labels)
            errD_fake = criterion(outD_fake, fake_labels)
            errD_real.backward()
            errD_fake.backward()
            errD = errD_real + errD_fake
            optD.step()

            generator.zero_grad()
            outD_fake = discriminator(fake_data).view(-1)
            errG = criterion(outD_fake, real_labels)
            errG.backward()
            optG.step()

            losses.append({'Iteration': counter, 'Discriminator Loss': errD.item(), 'Generator Loss': errG.item()})
            counter += 1
            print(f'Iteration: {counter}, Discriminator Loss: {errD:0.3f}, Generator Loss: {errG:0.3f}')

            if counter % 50 == 0:
                test_images = generator(fixed_noise, patch_size, n_qubits, n_a_qubits,q_depth).view(batch_size, 1, image_size, image_size).cpu().detach()
                results.append(test_images)
                generator_path = os.path.join(model_folder, f'generator_{n_qubits}qb_{counter}.pth')
                torch.save(generator.state_dict(), generator_path)
                # Save test images
                for j in range(batch_size):
                    image_path = os.path.join(images_folder, f'image_{counter*batch_size + j}.png')
                    plt.imsave(image_path, test_images[j].squeeze().numpy(), cmap='gray')

            if counter == num_iter:
                break
        if counter == num_iter:
            break

    losses_df = pd.DataFrame(losses)
    losses_df.to_csv(os.path.join(parent_folder, f'training_losses_{n_qubits}qb_{digit_class}.csv'), index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Quantum GAN Configuration')
    parser.add_argument('--config_file', type=str, default='config.json', help='Configuration file path')
    parser.add_argument('--config_name', type=str, default='default', help='Configuration name')
    parser.add_argument('--digit_class', type=int, default=0, help='Digit class to train on')
    args = parser.parse_args()
    main(args)
