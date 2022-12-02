import torch
from torch import nn


def deepfool(
  model_fn,
  x,
  y,
  device,
  steps=50,
  overshoot=0.02,
  return_target_y=False
):
    x = x.clone().detach().to(device)
    y = y.clone().detach().to(device)

    batch_size = len(x)
    correct = torch.tensor([True] * batch_size)
    target_y = y.clone().detach().to(device)
    curr_steps = 0

    adv_x = []
    for idx in range(batch_size):
        x_temp = x[idx:idx + 1].clone().detach()
        adv_x.append(x_temp)

    while (True in correct) and (curr_steps < steps):
        for idx in range(batch_size):
            if not correct[idx]: continue
            early_stop, pre, adv_x_temp = forward_indiv(model_fn, adv_x[idx], y[idx], overshoot)
            adv_x[idx] = adv_x_temp
            target_y[idx] = pre
            if early_stop:
                correct[idx] = False
        curr_steps += 1

    adv_x = torch.cat(adv_x).detach()

    if return_target_y:
        return adv_x, target_y
    return adv_x


def forward_indiv(model, x, label, overshoot):
    x.requires_grad = True
    fs = get_logits(model, x)[0]
    _, pre = torch.max(fs, dim=0)
    if pre != label:
        return (True, pre, x)

    ws = construct_jacobian(fs, x)
    x = x.detach()

    f_0 = fs[label]
    w_0 = ws[label]

    wrong_classes = [i for i in range(len(fs)) if i != label]
    f_k = fs[wrong_classes]
    w_k = ws[wrong_classes]

    f_prime = f_k - f_0
    w_prime = w_k - w_0
    value = torch.abs(f_prime) \
            / torch.norm(nn.Flatten()(w_prime), p=2, dim=1)
    _, hat_L = torch.min(value, 0)

    delta = (torch.abs(f_prime[hat_L])*w_prime[hat_L]
             / (torch.norm(w_prime[hat_L], p=2)**2))

    target_label = hat_L if hat_L < label else hat_L+1

    adv_x = x + (1+ overshoot)*delta
    adv_x = torch.clamp(adv_x, min=0, max=1).detach()
    return (False, target_label, adv_x)

def get_logits(model, inputs, labels=None, *args, **kwargs):
    logits = model(inputs)
    return logits


def construct_jacobian(y, x):
    x_grads = []
    for idx, y_element in enumerate(y):
        if x.grad is not None:
            x.grad.zero_()
        y_element.backward(retain_graph=(False or idx+1 < len(y)))
        x_grads.append(x.grad.clone().detach())
    return torch.stack(x_grads).reshape(*y.shape, *x.shape)