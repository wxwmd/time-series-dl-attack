import torch
from torch import nn


def mifsgm(model, images, labels, device, eps=8 / 255, alpha=2 / 255, steps=10, decay=1.0):
    r"""
    Overridden.
    """
    images = images.clone().detach().to(device)
    labels = labels.clone().detach().to(device)

    momentum = torch.zeros_like(images).detach().to(device)

    loss = nn.CrossEntropyLoss()

    adv_images = images.clone().detach()

    for _ in range(steps):
        adv_images.requires_grad = True
        outputs = get_logits(model, adv_images)

        # Calculate loss
        cost = loss(outputs, labels)

        # Update adversarial images
        grad = torch.autograd.grad(cost, adv_images,
                                   retain_graph=False, create_graph=False)[0]

        grad = grad / torch.mean(torch.abs(grad), dim=(1, 2, 3), keepdim=True)
        grad = grad + momentum * decay
        momentum = grad

        adv_images = adv_images.detach() + alpha * grad.sign()
        delta = torch.clamp(adv_images - images, min=-eps, max=eps)
        adv_images = torch.clamp(images + delta, min=0, max=1).detach()

    return adv_images


def get_logits(model, inputs):
    logits = model(inputs)
    return logits
