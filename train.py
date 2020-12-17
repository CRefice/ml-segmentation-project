import torch
import copy
import time

from eval import evaluate


class EarlyStopping:
    """ This class stops the training process early when the validation loss is no longer increasing. """

    def __init__(self, mode="min", min_delta=0, patience=10, percentage=False):
        """
        Create a new EarlyStopping object.

        Arguments:
        mode -- Either "min" or "max", depending on whether the metric given to step() indicates a better score
                with a lower or higher value respectively. (Default: "min")
        min_delta -- Bias to apply to the metric passed to step when determining an improvement. (Default: 0)
        patience -- The amount of epochs that must pass showing no improvement in orderr for the training to stop. (Default: 10)
        percentage -- If true, compute improvement delta as a function of a percentage of the best value so far,
                      instead of absolute difference. (Default: False)
        """

        self.mode = mode
        self.min_delta = min_delta
        self.patience = patience
        self.best = None
        self.num_bad_epochs = 0
        self.is_better = None
        self._init_is_better(mode, min_delta, percentage)

        if patience == 0:
            self.is_better = lambda a, b: True
            self.step = lambda a: False

    def step(self, metrics):
        if self.best is None:
            self.best = metrics
            return False

        if torch.isnan(metrics):
            return True

        if self.is_better(metrics, self.best):
            self.num_bad_epochs = 0
            self.best = metrics
        else:
            self.num_bad_epochs += 1

        if self.num_bad_epochs >= self.patience:
            return True

        return False

    def _init_is_better(self, mode, min_delta, percentage):
        if mode not in {"min", "max"}:
            raise ValueError("mode " + mode + " is unknown!")
        if not percentage:
            if mode == "min":
                self.is_better = lambda a, best: a < best - min_delta
            if mode == "max":
                self.is_better = lambda a, best: a > best + min_delta
        else:
            if mode == "min":
                self.is_better = lambda a, best: a < best - (best * min_delta / 100)
            if mode == "max":
                self.is_better = lambda a, best: a > best + (best * min_delta / 100)


def train_model(
    device, model, optimizer, criterion, train_loader, val_loader, num_epochs=25
):
    """
    Train the model with the given parameters, evaluating it once for each epoch.

    Arguments:
    device -- torch device to run the optimization on (prefer the GPU for this)
    model -- the model to optimize.
    optimizer -- the optimizing algorithm to use.
    criterion -- the loss function to feed to the optimizer.
    train_loader -- batch data loader for the training set.
    val_loader -- batch data loader for the validation set.
    num_epochs -- the number of epochs to train the model for. (Default: 25)

    Returns:
    Tuple of (model, best validation loss)
    """
    best_model_wts = copy.deepcopy(model.state_dict())
    best_loss = float("inf")
    es = EarlyStopping(patience=5)

    for epoch in range(num_epochs):
        since = time.time()

        print("Epoch {}/{}".format(epoch + 1, num_epochs))
        print("-" * 10)
        # Training phase
        model.train()

        epoch_loss = 0
        epoch_samples = len(train_loader)

        torch.cuda.empty_cache()
        for inputs, labels in train_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            # zero the parameter gradients
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        # scheduler.step()

        print("Epoch train loss: {}".format(epoch_loss / epoch_samples))
        # print("Epoch dice loss: {}".format(epoch_dice_loss / epoch_samples))

        # Evaluation phase
        model.eval()
        val_loss = evaluate(device, model, val_loader, criterion)
        if es.step(val_loss):
            break  # early stop criterion is met, we can stop now

        if val_loss < best_loss:
            best_model_wts = copy.deepcopy(model.state_dict())
            best_loss = val_loss

        print("Epoch validation loss: {}".format(val_loss))

        time_elapsed = time.time() - since
        print("Took {:.0f}m {:.0f}s\n".format(time_elapsed // 60, time_elapsed % 60))

    model.load_state_dict(best_model_wts)
    return model, best_loss
