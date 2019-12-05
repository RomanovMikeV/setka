import torch

def tensor_loss(output, input):
    return torch.nn.functional.cross_entropy(output, input[1])

def const(output, input):
    return 1.0, 1.0

def tensor_acc(output, input):
    n_correct = (output.argmax(dim=1) == input[1]).float().sum()
    return n_correct, input[1].numel()

def list_loss(output, input):
    return tensor_loss(output[0], input)

def list_acc(output, input):
    return tensor_acc(output[0], input)

def dict_loss(output, input):
    return tensor_loss(output['res'], input)

def dict_acc(output, input):
    return tensor_acc(output['res'], input)