#create encoder
#create decoder
#create discriminator

import torch
from torch.autograd import Variable
import torch.nn as nn

class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(28*28, 400),
            nn.ReLU(),
            nn.Linear(400, 40))

    def reparameterize(self, mu, logvar):
        var = logvar.exp()
        std = var.sqrt()
        eps = Variable(torch.FloatTensor(std.size()).normal_())
        return eps.mul(std).add(mu)

    def forward(self, x):
        h = self.model(x)
        mu = h[:, :20]
        logvar = h[:,20:]
        z = self.reparameterize(mu, logvar)
        return z

class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(20, 400),
            nn.ReLU(),
            nn.Linear(400, 28*28),
            nn.Sigmoid())
    def forward(self, x):
        out = self.model(x)
        return out

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(20, 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )
    def forward(self, z):
        validity = self.model(z)
        return validity



#creating the dataset
#sample the information from any other distribution as noise and produce and train the whole pipeline

"""    def generation_with_interpolation(self, x_one, x_two, alpha):
        hidden_one = self.encoder(x_one)
        hidden_two = self.encoder(x_two)
        mu_one = hidden_one[:,:20]
        logvar_one = hidden_one[:, 20:]
        mu_two = hidden_two[:,:20]
        logvar_two = hidden_two[:,20:]
        mu = (1-alpha)*mu_one + alpha*mu_two
        logvar = (1-alpha)*logvar_one+alpha*logvar_two
        z = self.reparameterize(mu, logvar)
        generated_image = self.decoder(z)
        return generated_image
"""