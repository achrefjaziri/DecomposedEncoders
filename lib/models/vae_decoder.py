from collections import OrderedDict
import torch
import torch.nn as nn



class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)



class UnFlatten(nn.Module):
    def forward(self, input, size=1024):
        return input.view(input.size(0), size, 1, 1)


class VAE_decoder(nn.Module):
    def __init__(self, image_channels=3, input_feature=50176,h_dim=1024, z_dim=32):
        super(VAE_decoder, self).__init__()
        '''
        self.encoder = nn.Sequential(
            nn.Conv2d(image_channels, 32, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=4, stride=2),
            nn.ReLU(),
            Flatten()
        )
        '''


        self.fc1 = nn.Linear(input_feature, z_dim)
        self.fc2 = nn.Linear(input_feature, z_dim)
        self.fc3 = nn.Linear(z_dim, h_dim)

        self.decoder = nn.Sequential(
            UnFlatten(),
            nn.ConvTranspose2d(h_dim, 128, kernel_size=5, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=5, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=6, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(32, image_channels, kernel_size=6, stride=2),
            nn.Sigmoid(),
        )

    def reparameterize(self, mu, logvar):
        std = logvar.mul(0.5).exp_()
        # return torch.normal(mu, std)
        esp = torch.randn(*mu.size()).to(std.device)
        z = mu + std * esp
        return z

    def decode(self, z):

        z = self.fc3(z)
        z = self.decoder(z)
        return z

    def forward(self, h):
        mu, logvar = self.fc1(h), self.fc2(h)
        z = self.reparameterize(mu, logvar)

        z = self.decode(z)
        return z, mu, logvar


class VAE(nn.Module):
    def __init__(self, image_channels=3, h_dim=1024, z_dim=32):
        super(VAE, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(image_channels, 32, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=4, stride=2),
            nn.ReLU(),
            Flatten()
        )

        self.fc1 = nn.Linear(h_dim, z_dim)
        self.fc2 = nn.Linear(h_dim, z_dim)
        self.fc3 = nn.Linear(z_dim, h_dim)

        self.decoder = nn.Sequential(
            UnFlatten(),
            nn.ConvTranspose2d(h_dim, 128, kernel_size=5, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=5, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=6, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(32, image_channels, kernel_size=6, stride=2),
            nn.Sigmoid(),
        )

    def reparameterize(self, mu, logvar):
        std = logvar.mul(0.5).exp_()
        # return torch.normal(mu, std)
        esp = torch.randn(*mu.size()).to(std.device)
        z = mu + std * esp
        return z

    def bottleneck(self, h):
        mu, logvar = self.fc1(h), self.fc2(h)
        z = self.reparameterize(mu, logvar)
        return z, mu, logvar

    def encode(self, x):
        h = self.encoder(x)
        z, mu, logvar = self.bottleneck(h)
        return z, mu, logvar

    def decode(self, z):
        z = self.fc3(z)
        z = self.decoder(z)
        return z

    def forward(self, x):
        z, mu, logvar = self.encode(x)
        z = self.decode(z)
        return z, mu, logvar





if __name__=="__main__":

    h = torch.randn(size=(4, 50176))
    im = torch.randn(size=(4,3,64,64))

    vae_Dec = VAE_decoder(image_channels=3,input_feature=50176, h_dim=1024, z_dim=64)  # args.linear_classifier

    z, mu, logvar= vae_Dec(h)
    print('final outputs',z.shape,mu.shape,logvar.shape)





    '''
        def encode(self, x):
        x = self.encoder(x)
        x = x.view(x.size(0), -1)
        z_mean = self.latent_mu(x)
        z_std = self.latent_std(x)
        return z_mean, z_std

    def reparameterize(self, mu, std):
        eps = std.data.new(std.size()).normal_()
        return eps.mul(std).add(mu)

    def decode(self, z):
        z = self.latent_decoder(z)
        z = z.view(z.size(0), self.enc_channels, self.enc_spatial_dim_x, self.enc_spatial_dim_y)
        x = self.decoder(z)
        return x

    def generate(self):
        z = torch.randn(self.batch_size, self.latent_dim).to(self.device)
        x = self.decode(z)
        x = torch.sigmoid(x)
        return x

    def forward(self, x):
        z_mean, z_std = self.encode(x)
        output_samples = torch.zeros(self.num_samples, x.size(0), self.out_channels, self.patch_size,
                                     self.patch_size).to(self.device)
        classification_samples = torch.zeros(self.num_samples, x.size(0), self.num_classes).to(self.device)
        for i in range(self.num_samples):
            z = self.reparameterize(z_mean, z_std)
            output_samples[i] = self.decode(z)
            classification_samples[i] = self.classifier(z)
        return classification_samples, output_samples, z_mean, z_std
    '''
