import matplotlib
import matplotlib.pyplot as plt
import argparse
import random
import numpy as np
import os
import torch
from torchvision import datasets, transforms
from torch.utils.data.sampler import SubsetRandomSampler
import torch.utils.data
import torch
from torch import nn
from torch.nn import functional as F
from scipy.io import loadmat

# Sample from the Gumbel-Softmax distribution and optionally discretize.
# The Gumbel-Softmax distribution is a continuous relaxation of the categorical distribution
class GumbelSoftmax(nn.Module):

  def __init__(self, f_dim, c_dim):
    super(GumbelSoftmax, self).__init__()
    self.logits = nn.Linear(f_dim, c_dim)
    self.f_dim = f_dim
    self.c_dim = c_dim
    """
    f_dim := feature dimension
    c_dim := number of categories
    logits := takes input of size f_dim and outputs c_dim dimensional logits
    """

  def sample_gumbel(self, shape, is_cuda=False, eps=1e-20):
    U = torch.rand(shape)
    if is_cuda:
      U = U.cuda()
    return -torch.log(-torch.log(U + eps) + eps)
  # campiona da una distribuzione di Gumbel

  def gumbel_softmax_sample(self, logits, temperature):
    y = logits + self.sample_gumbel(logits.size(), logits.is_cuda)
    return F.softmax(y / temperature, dim=-1)

  def gumbel_softmax(self, logits, temperature, hard=False):
    """
    ST-gumple-softmax
    input: [*, n_class]
    return: flatten --> [*, n_class] an one-hot vector
    """
    #categorical_dim = 10
    y = self.gumbel_softmax_sample(logits, temperature)

    if not hard:
        return y

    shape = y.size()
    _, ind = y.max(dim=-1)
    y_hard = torch.zeros_like(y).view(-1, shape[-1])
    y_hard.scatter_(1, ind.view(-1, 1), 1)
    y_hard = y_hard.view(*shape)
    # Set gradients w.r.t. y_hard gradients w.r.t. y
    y_hard = (y_hard - y).detach() + y
    return y_hard

  def forward(self, x, temperature=1.0, hard=False):
    logits = self.logits(x).view(-1, self.c_dim)
    prob = F.softmax(logits, dim=-1)
    y = self.gumbel_softmax(logits, temperature, hard)
    return logits, prob, y
  
  # Sample from a Gaussian distribution
class Gaussian(nn.Module):
  def __init__(self, in_dim, z_dim):
    super(Gaussian, self).__init__()
    self.mu = nn.Linear(in_dim, z_dim)
    self.var = nn.Linear(in_dim, z_dim)

  def reparameterize(self, mu, var):
    std = torch.sqrt(var + 1e-10)
    noise = torch.randn_like(std)
    z = mu + noise * std
    return z

  def forward(self, x):
    mu = self.mu(x)
    var = F.softplus(self.var(x))
    z = self.reparameterize(mu, var)
    return mu, var, z

# Inference Network
class InferenceNet(nn.Module):
  def __init__(self, x_dim, z_dim, y_dim):
    super(InferenceNet, self).__init__()
    # ci sono due reti neurali: una per q(y|x) e una per q(z|y,x)
    # q(y|x)
    print("x_dim", x_dim, y_dim,z_dim)
    self.inference_qyx = torch.nn.ModuleList([
        nn.Linear(x_dim, 512),
        nn.ReLU(),
        nn.Linear(512, 512),
        nn.ReLU(),
        GumbelSoftmax(512, y_dim)
    ])

    # q(z|y,x)
    self.inference_qzyx = torch.nn.ModuleList([
        nn.Linear(x_dim + y_dim, 512),
        nn.ReLU(),
        nn.Linear(512, 512),
        nn.ReLU(),
        Gaussian(512, z_dim)
    ])

  # q(y|x)
  def qyx(self, x, temperature, hard):
    print("Entra in qyx")
    num_layers = len(self.inference_qyx)
    for i, layer in enumerate(self.inference_qyx):
      if i == num_layers - 1:
        print("entra in if")
        #last layer is gumbel softmax
        x = layer(x, temperature, hard)
      else:
        print("entra in else")
        # print("x:", x)
        # print("x dimension", x.shape)
        print("layer:", layer)
        x=layer(x) # dimension of x: torch.Size([1, 746496])
                    # layer is a torch linear object
    print("Esce da qyx")
    return x
  # funzione per calcolare q(y|x)

  # q(z|x,y)
  def qzxy(self, x, y):
    concat = torch.cat((x, y), dim=1) # combina l'input di x e y
    for layer in self.inference_qzyx:
      concat = layer(concat)
    return concat

  def forward(self, x, temperature=1.0, hard=0):
    #x = Flatten(x)
    print("Entra in forward infNet")
    # q(y|x)
    logits, prob, y = self.qyx(x, temperature, hard)

    # q(z|x,y)
    mu, var, z = self.qzxy(x, y)

    output = {'mean': mu, 'var': var, 'gaussian': z,
              'logits': logits, 'prob_cat': prob, 'categorical': y}
    print("Esce da forward infNet")
    return output
# in input prende un immagine x
# la rete usa il metodo qyx  inferire la variabile latente discreta y data l'immagine di input x. Questo viene fatto approssimando la distribuzione categoriale con Gumbel-Softmax.
# La rete usa il metodo qzxy per inferire la variabile latente continua z data l'immagine x e la variabile latente discreta y.
# in output restituisce la media mu, la varianza var e il campione z della variabile latente continua z, i logit, la probabilità e il campione y della variabile latente discreta y.

# Generative Network
class GenerativeNet(nn.Module):
  def __init__(self, x_dim, z_dim, y_dim):
    super(GenerativeNet, self).__init__()

    # p(z|y)
    self.y_mu = nn.Linear(z_dim,z_dim*y_dim)#y_dim, z_dim
    self.y_var = nn.Linear(z_dim,z_dim*y_dim)#

    # p(x|z) genera x dato z
    self.generative_pxz = torch.nn.ModuleList([
        nn.Linear(z_dim, 512),
        nn.ReLU(),
        nn.Linear(512, 512),
        nn.ReLU(),
        nn.Linear(512, x_dim),
        torch.nn.Sigmoid() # garantisce che l'output sia compreso tra 0 e 1
    ])

  # p(z|y)
  def pzy(self, y):
    print("y",y.size())
    y_mu = self.y_mu(y)
    y_var = F.softplus(self.y_var(y)) # garantisce che la varianza sia sempre positiva
    return y_mu, y_var

  # p(x|z)
  def pxz(self, z):
    for layer in self.generative_pxz:
      z = layer(z)
    return z

  def forward(self, z, y):
    # p(z|y)
    y_mu, y_var = self.pzy(y)

    # p(x|z)
    x_rec = self.pxz(z)

    output = {'y_mean': y_mu, 'y_var': y_var, 'x_rec': x_rec}
    return output
  
  # in input la classe prende una variabile latente z e una variabile categorica y
  # la rete usa il metodo pzy per calcolare la media e la varianza della distribuzione gaussiana di z data y
  # e il metodo pxz per generare l'immagine x dato il campione z
  # in output la rete restituisce la media e la varianza delle variabili latenti y e l'immagine generata x