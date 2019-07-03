import torch
from torch.autograd import Variable
import torch.nn as nn
class C_VariationalAutoEncoder(nn.Module):
    def __init__(self):
        super(C_VariationalAutoEncoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(28*28+10, 400),
            nn.ReLU(),
            nn.Linear(400, 40))
        self.decoder = nn.Sequential(
            nn.Linear(20+10, 400),
            nn.ReLU(),
            nn.Linear(400, 28*28),
            nn.Sigmoid())
    def reparameterize(self, mu, logvar):
        var = logvar.exp()
        std = var.sqrt()
        eps = Variable(torch.FloatTensor(std.size()).normal_())
        return eps.mul(std).add(mu)

    def forward(self, x, c):
        enc_input = torch.cat([x, c], 1)
        h         = self.encoder(enc_input)
        mu        = h[:, :20]
        logvar    = h[:, 20:]
        z         = self.reparameterize(mu, logvar)
        dec_input = torch.cat([z, c], 1)
        x_hat     = self.decoder(dec_input)
        return x_hat, mu, logvar

    def generate_image(self, c):
        z = Variable(torch.FloatTensor(c.size()[0], 20).normal_())
        dec_input = torch.cat([z, c], 1)
        generated_image = self.decoder(dec_input)
        return generated_image