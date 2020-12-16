import torch
import torch.nn.functional as F


def _batch_dot(a, b):
    """ Compute the batch-wise dot product between all vectors in a and in b. """
    batch_size = a.size()[0]
    a = a.view(batch_size, 1, -1).float()
    b = b.view(batch_size, -1, 1).float()
    return a.bmm(b).squeeze()


def _batch_sum(a):
    """ Compute the batch-wise sum between all vectors in a and in b. """
    batch_size = a.size()[0]
    return a.view(batch_size, -1).sum(dim=1)


class DiceLoss:
    """
    Loss function for the Sorensen-Dice performance metric.
    This function works on batches of images and can be used to compute gradients.
    """

    def __call__(self, output, target):
        """
        Compute the loss function between the output generated by a model and the ground-truth labels.

        Arguments:
        output -- [C, N, H, W] tensor, where C == batch index.
        target -- [C, N, H, W] tensor, where C == batch index.
        output and target must have the same size.

        Returns:
        A scalar, the average dice loss over all batches.
        """
        target = F.one_hot(target).permute(0, 3, 1, 2).contiguous()
        eps = 0.0001
        inter = _batch_dot(output, target)
        union = _batch_sum(output ** 2) + _batch_sum(target ** 2) + eps
        t = (2 * inter.float() + eps) / union.float()
        return 1 - t.mean()


class CombinedLoss:
    def __init__(self, loss1, loss2, mix=0.5):
        """
        Combines two differentiable loss functions into one through a weighted sum.
        The resulting function is also differentiable.

        Arguments:
        loss1, loss2 -- The two loss functions to be combined.
        mix -- The relative weight given to the first loss function.
               Default is 0.5 (both functions weighed equally).
        """
        self.a = loss1
        self.b = loss2
        self.mix = mix

    def __call__(self, output, target):
        """
        Compute the combined loss function between the generated output and ground-truth labels.

        Arguments:
        output -- tensor generated by the model
        target -- ground-truth tensor.
        output and target must have the same size.
        """
        wa = self.mix
        wb = 1 - wa
        return wa * self.a(output, target) + wb * self.b(output, target)


def _find_occupancies(n_classes, labels):
    """
    Computes the average occupancy (percentage over entire dataset)
    of all classes in the range [0, n_classes)

    Arguments:
    n_classes -- Number of distinct classes contained in the labels
    labels -- Iterable over label tensors assigning each pixel to a class number
    """
    occupancies = torch.empty(n_classes)
    num_samples = 0
    for label in labels:
        for clas in range(n_classes):
            occupancies[clas] += torch.count_nonzero(label == clas)
        num_samples += torch.numel(label)
    return occupancies / num_samples


def find_class_weights(n_classes, labels):
    """
    Given a dataset with imbalanced classes, finds the weights
    (to be used with a cross entropy loss) of all classes in the range [0, n_classes),
    computed as 1 / sqrt(occupancy) and normalized to sum to 1.

    Arguments:
    n_classes -- Number of distinct classes contained in the labels
    labels -- Iterable over label tensors assigning each pixel to a class number
    """
    occ = _find_occupancies(n_classes, labels)
    occ = 1 / occ.sqrt()
    return occ / sum(occ)
