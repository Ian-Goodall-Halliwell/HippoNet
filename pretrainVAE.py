import torch
import torch.nn as nn
import torch.nn.functional as F

class VAE(nn.Module):
    def __init__(self, in_channels: int, latent_dim: int, hidden_dims=None) -> None:
        super(VAE, self).__init__()

        self.latent_dim = latent_dim

        if hidden_dims is None:
            hidden_dims = [32, 64, 128, 256, 512]
            # hidden_dims = [32, 64, 128, 256]

        # Encoder
        modules = []
        for h_dim in hidden_dims:
            modules.append(
                nn.Sequential(
                    nn.Conv2d(in_channels, out_channels=h_dim,
                              kernel_size=3, stride=2, padding=1),
                    nn.BatchNorm2d(h_dim),
                    nn.LeakyReLU()))
            in_channels = h_dim

        self.encoder = nn.Sequential(*modules)

        # Adjust the size of the linear layers to match the output of your encoder
        self.fc_mu = nn.Linear(hidden_dims[-1]*240, latent_dim)  # Adjusted for the new flattened size
        self.fc_var = nn.Linear(hidden_dims[-1]*240, latent_dim)  # Adjusted for the new flattened size

        # Decoder
        modules = []
        self.decoder_input = nn.Sequential(
            nn.Linear(latent_dim, hidden_dims[-1]*240),
            nn.LeakyReLU())

        hidden_dims.reverse()

        for i in range(len(hidden_dims) - 1):
            modules.append(
                nn.Sequential(
                    nn.ConvTranspose2d(hidden_dims[i],
                                       hidden_dims[i + 1],
                                       kernel_size=3,
                                       stride=2,
                                       padding=1,
                                       output_padding=1),
                    nn.BatchNorm2d(hidden_dims[i + 1]),
                    nn.LeakyReLU()))

        self.decoder = nn.Sequential(*modules)

        self.final_layer = nn.Sequential(
                            nn.ConvTranspose2d(hidden_dims[-1],
                                               hidden_dims[-1],
                                               kernel_size=3,
                                               stride=2,
                                               padding=1,
                                               output_padding=1),
                            nn.BatchNorm2d(hidden_dims[-1]),
                            nn.LeakyReLU(),
                            nn.Conv2d(hidden_dims[-1], out_channels=3,
                                      kernel_size=3, padding=1),
                            nn.Sigmoid())

    def encode(self, input):
        result = self.encoder(input)
        result = torch.flatten(result, start_dim=1)
        mu = self.fc_mu(result)
        log_var = self.fc_var(result)
        return mu, log_var

    def decode(self, z):
        result = self.decoder_input(z)
        result = result.view(-1, 512, 12, 20)  # -1 is used to automatically infer the batch size

        result = self.decoder(result)
        result = self.final_layer(result)
        return result

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu

    def forward(self, input):
        mu, log_var = self.encode(input)
        z = self.reparameterize(mu, log_var)
        return self.decode(z), mu, log_var

    def reconstruct(self, x):
        '''
        Reconstruct from input images (b, 3, 224, 224)
        '''
        return self.forward(x)[0]
    
    def get_z(self, x):
        '''
        Return the latent embedding of input images
        '''
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)
        return z
    
    def generate_from_z(self, z):
        '''
        Generate images from latent embedding z
        '''
        return self.decode(z)