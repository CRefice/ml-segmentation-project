import torch


def evaluate(device, model, loader, criterion):
    """ Evaluate the model on the given data according to the specified criterion. """
    model.eval()
    n_val = len(loader)
    tot = 0
    with torch.no_grad():
        for image, labels in loader:
            pred = model(image.to(device))
            tot += criterion(pred, labels.to(device))
    return tot / n_val
