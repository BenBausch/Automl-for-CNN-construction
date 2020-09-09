import torch
import numpy as np

class AvgrageMeter(object):

  def __init__(self):
    self.reset()

  def reset(self):
    self.avg = 0
    self.sum = 0
    self.cnt = 0

  def update(self, val, n=1):
    self.sum += val * n
    self.cnt += n
    self.avg = self.sum / self.cnt


def accuracy(output, target, topk=(1,)):
  maxk = max(topk)
  batch_size = target.size(0)

  _, pred = output.topk(maxk, 1, True, True)
  pred = pred.t()
  correct = pred.eq(target.view(1, -1).expand_as(pred))


  res = []
  for k in topk:
    correct_k = correct[:k].view(-1).float().sum(0)
    res.append(correct_k.mul_(100.0/batch_size))
  return res


def get_borders(name_of_parameter):
  borders = {
    'optimizer': (float, [0,1]),
    'criterion': (float, [0,1]),
    'n_conv_layers': (int, [1,3]),
    'n_channels_conv_0': (int, [64,2048]),
    'n_channels_conv_1': (int, [64,2048]),
    'n_channels_conv_2': (int, [64,2048]),
    'kernel_size': (int, [1,5]),
    'global_avg_pooling': (bool, [True, False]),
    'use_BN': (bool, [True, False]),
    'n_fc_layers': (int, [1,3]),
    'n_channels_fc_0': (int, [1, 300]),
    'n_channels_fc_1': (int, [1, 300]),
    'n_channels_fc_2': (int, [1, 300])}
  if name_of_parameter in borders.keys():
    return borders[name_of_parameter]
  return None


def within_borders(value, borders):
  """
  Cuts the value to be within the borders if necessary.
  """
  if borders[0] > value:
    return borders[0]
  elif borders[1] < value:
    return borders[1]
  else:
    return value


def get_optim(optim_value):
  "handels the encoding of the opitmizers"
  if optim_value < 0 or optim_value > 1:
    optim_value = np.random.uniform(1,0)
  if optim_value <= 0.25:
    return torch.optim.Adam
  elif optim_value <= 0.5:
    return torch.optim.AdamW
  elif optim_value <= 0.75:
    return torch.optim.Adadelta
  else:
    return torch.optim.SGD

def get_crit(crit_value):
  "handels the encoding of the criterions"
  if crit_value < 0 or crit_value > 1:
    crit_value = np.random.uniform(1,0)
  if crit_value < 0.5:
    return torch.nn.CrossEntropyLoss
  else:
    return torch.nn.MSELoss
