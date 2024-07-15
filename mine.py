import gym
import minerl
import logging
from pretrainVAE import VAE
import torch
from torch import optim
from torch import nn
import numpy as np
import matplotlib.pyplot as plt
from torchvision.utils import make_grid
from PIL import Image
# logging.basicConfig(level=logging.DEBUG)
import torch

import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Function to display images
def show_images(obsdata, recon_images, iteration):
    
    # Ensure no gradient is computed
    with torch.no_grad():
        # Concatenate the original and reconstructed images
        comparison = torch.cat([obsdata[0], recon_images[0]], dim=1)
        # Convert to grid
        comparison_grid = make_grid(comparison, nrow=obsdata.size(0))
        # Convert to PIL image
        pil_img = Image.fromarray((255*comparison_grid.cpu()).numpy().astype(np.uint8).transpose(1, 2, 0))

        
        pil_img.save('vae/{}.png'.format(iteration))


def loss_function(x_hat, x, mu, log_var):
    # reconstruction loss.
    start_index = (x_hat.shape[2] - x.shape[2]) // 2
    end_index = start_index + x.shape[2]
    x_hat = x_hat[:, :, start_index:end_index, :]

    BCE = F.binary_cross_entropy(x_hat, x, reduction='sum')

    # KL-divergence
    KLD = -0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp())

    loss = BCE + KLD
    return loss, BCE, KLD

env = gym.make('MineRLBasaltFindCave-v0')

obs = env.reset()

model = VAE(3, 2048).to(device) 

optimizer = optim.Adam(model.parameters(), lr=0.0005)  
scheduler = optim.lr_scheduler.StepLR(optimizer, 25, 0.95)

# Set the model to train mode
model.train()
    
loss_values = []
up = True
batch_size = 4
batch = torch.zeros((batch_size, 3, 360, 640)).to(device)

for trial in range(100000):
    if trial % 3600 == 0:
        obs = env.reset()
        
    if trial == 0:
        # Take a random action
        action = env.action_space.sample()
        # In BASALT environments, sending ESC action will end the episode
        # Lets not do that
        action["ESC"] = 0
        action['attack'] = 0
        action['hotbar.1'] = 0
        action['hotbar.2'] = 0
        action['hotbar.3'] = 0
        action['hotbar.4'] = 0
        action['hotbar.5'] = 0
        action['hotbar.6'] = 0
        action['hotbar.7'] = 0
        action['hotbar.8'] = 0
        action['hotbar.9'] = 0
        action['inventory'] = 0
        action['pickItem'] = 0
        action['sneak'] = 0
        action['sprint'] = 0
        action['swapHands'] = 0
        action['use'] = 0
        action['camera'] = [5 if up else -5, action['camera'][1]/10]
        up = not up
    if trial % batch_size == 0:
        action = env.action_space.sample()
        # In BASALT environments, sending ESC action will end the episode
        # Lets not do that
        action["ESC"] = 0
        action['attack'] = 0
        action['hotbar.1'] = 0
        action['hotbar.2'] = 0
        action['hotbar.3'] = 0
        action['hotbar.4'] = 0
        action['hotbar.5'] = 0
        action['hotbar.6'] = 0
        action['hotbar.7'] = 0
        action['hotbar.8'] = 0
        action['hotbar.9'] = 0
        action['inventory'] = 0
        action['pickItem'] = 0
        action['sneak'] = 0
        action['sprint'] = 0
        action['swapHands'] = 0
        action['use'] = 0
        action['camera'] = [10 if up else -10, action['camera'][1]/10]
        up = not up
        
    obs, reward, done, _ = env.step(action)
    
    obsdata = torch.tensor(obs['pov'].copy().astype(np.float32)).squeeze().to(device)
    obsdata = obsdata.permute(-1, *range(obsdata.dim() - 1))
    obsdata = obsdata / 255.0
    batch[trial % batch_size] = obsdata
    
    if trial % batch_size != 0:
        continue
    if trial == 0:
        continue
    
    # Initialize the running loss
    running_loss = 0.0

    # Zero the gradients
    optimizer.zero_grad()

    # Forward pass
    recon_images, mu, log_var = model(batch)
    loss, BCE, KLD = loss_function(recon_images, batch, mu, log_var)  
    loss_values.append(loss.to('cpu').item()) 

    # Backward pass
    optimizer.zero_grad()  
    loss.backward()
    optimizer.step() 
    
    every_n = 500
    if trial % every_n == 0:  # Display every n iterations
        show_images(obsdata.to('cpu'), recon_images.to('cpu'), trial)

    # Update the weights
    optimizer.step()
    env.render()
    
model.to('cpu')
torch.save(model.state_dict(), 'vae.model')

# After training loop
# Step 4: (Optional) Plot the loss
plt.plot(loss_values)
plt.title('Loss over Time')
plt.xlabel('Iteration')
plt.ylabel('Loss')
plt.show()