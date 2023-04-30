import torch
import torch.nn as nn
import torch.nn.functional as F


EPSILON = 1e-5


def classified_correctly(output: torch.tensor, y: torch.tensor, targeted: bool):
    prediction = torch.argmax(output, dim=1)
    correct = prediction == y if targeted else prediction != y

    return correct.all()


class PGDAttack:
    """
    White-box L_inf PGD attack using the cross-entropy loss
    """
    def __init__(self, model, eps=8/255., n=50, alpha=1/255.,
                 rand_init=True, early_stop=True):
        """
        Parameters:
        - model: model to attack
        - eps: attack's maximum norm
        - n: max # attack iterations
        - alpha: step size at each iteration
        - rand_init: a flag denoting whether to randomly initialize
              adversarial samples in the range [x-eps, x+eps]
        - early_stop: a flag denoting whether to stop perturbing a 
              sample once the attack goal is met. If the goal is met
              for all samples in the batch, then the attack returns
              early, before completing all the iterations.
        """
        self.model = model
        self.n = n
        self.alpha = alpha
        self.eps = eps
        self.rand_init = rand_init
        self.early_stop = early_stop
        self.loss_func = nn.CrossEntropyLoss(reduction='none')

    def execute(self, x, y, targeted=False):
        """
        Executes the attack on a batch of samples x. y contains the true labels 
        in case of untargeted attacks, and the target labels in case of targeted 
        attacks. The method returns the adversarially perturbed samples, which
        lie in the ranges [0, 1] and [x-eps, x+eps]. The attack optionally 
        performs random initialization and early stopping, depending on the 
        self.rand_init and self.early_stop flags.
        """
        if self.rand_init:
            delta = torch.rand_like(x, requires_grad=True)
            delta = (2 * delta - 1) * self.eps  # change noise to be in [-eps, eps]
        else:
            delta = torch.zeros_like(x, requires_grad=True)

        for _ in range(self.n):
            prediction = self.model(x + delta)

            if self.early_stop and classified_correctly(prediction, y, targeted):
                break

            loss = torch.nn.functional.cross_entropy(prediction, y)
            loss = -1 * loss if targeted else loss

            grad = torch.autograd.grad(loss, delta)[0]

            delta += (self.alpha * torch.sign(grad))

            delta = torch.clamp(delta, -self.eps, self.eps)  # Make sure the perturbation is still at e magnitude
            delta = torch.clamp(x + delta, 0, 1) - x  # make sure the image is still a valid image

            assert torch.all(delta >= -self.eps - EPSILON) and torch.all(delta <= self.eps + EPSILON)
            assert torch.all(x + delta >= 0) and torch.all(x + delta <= 1)

        adversarial = x + delta

        assert torch.all(delta >= -self.eps - EPSILON) and torch.all(delta <= self.eps + EPSILON)
        assert torch.all(adversarial >= 0) and torch.all(adversarial <= 1)

        return adversarial


class NESBBoxPGDAttack:
    """
    Query-based black-box L_inf PGD attack using the cross-entropy loss, 
    where gradients are estimated using Natural Evolutionary Strategies 
    (NES).
    """
    def __init__(self, model, eps=8/255., n=50, alpha=1/255., momentum=0.,
                 k=200, sigma=1/255., rand_init=True, early_stop=True):
        """
        Parameters:
        - model: model to attack
        - eps: attack's maximum norm
        - n: max # attack iterations
        - alpha: PGD's step size at each iteration
        - momentum: a value in [0., 1.) controlling the "weight" of
             historical gradients estimating gradients at each iteration
        - k: the model is queries 2*k times at each iteration via 
              antithetic sampling to approximate the gradients
        - sigma: the std of the Gaussian noise used for querying
        - rand_init: a flag denoting whether to randomly initialize
              adversarial samples in the range [x-eps, x+eps]
        - early_stop: a flag denoting whether to stop perturbing a 
              sample once the attack goal is met. If the goal is met
              for all samples in the batch, then the attack returns
              early, before completing all the iterations.
        """
        self.model = model
        self.eps = eps
        self.n = n
        self.alpha = alpha
        self.momentum = momentum
        self.k = k
        self.sigma=sigma
        self.rand_init = rand_init
        self.early_stop = early_stop
        self.loss_func = nn.CrossEntropyLoss(reduction='none')

    def estimate_gradient(self, x, y, targeted):
        grad = torch.zeros_like(x, requires_grad=False)

        for _ in range(self.k):
            mu = 2 * torch.rand_like(x, requires_grad=False) - 1
            x_plus = torch.clamp(x + self.sigma * mu, 0, 1)
            x_minus = torch.clamp(x - self.sigma * mu, 0, 1)

            with torch.no_grad():
                diff = self.loss_func(self.model(x_plus), y) - self.loss_func(self.model(x_minus), y)

            grad += diff.view(len(diff), 1, 1, 1) * mu  # / self.sigma
            # grad = grad / torch.norm(grad)

        grad /= (2 * self.k * self.sigma)
        grad = -1 * grad if targeted else grad

        return grad

    def execute(self, x, y, targeted=False):
        """
        Executes the attack on a batch of samples x. y contains the true labels 
        in case of untargeted attacks, and the target labels in case of targeted 
        attacks. The method returns:
        1- The adversarially perturbed samples, which lie in the ranges [0, 1] 
            and [x-eps, x+eps].
        2- A vector with dimensionality len(x) containing the number of queries for
            each sample in x.
        """
        if self.rand_init:
            delta = torch.rand_like(x, requires_grad=False)
            delta = (2 * delta - 1) * self.eps  # change noise to be in [-eps, eps]
        else:
            delta = torch.zeros_like(x, requires_grad=False)

        iter_number = self.n
        grad_ema = torch.zeros_like(x)
        for i in range(self.n):
            with torch.no_grad():
                prediction = self.model(x + delta)

            if self.early_stop and classified_correctly(prediction, y, targeted):
                iter_number = i
                break

            grad = self.estimate_gradient(x + delta, y, targeted)
            grad_ema = self.momentum * grad_ema + (1 - self.momentum) * grad

            delta += (self.alpha * torch.sign(grad_ema))

            delta = torch.clamp(delta, -self.eps, self.eps)  # Make sure the perturbation is still at e magnitude
            delta = torch.clamp(x + delta, 0, 1) - x  # make sure the image is still a valid image

            assert torch.all(delta >= -self.eps - EPSILON) and torch.all(delta <= self.eps + EPSILON)
            assert torch.all(x + delta >= 0) and torch.all(x + delta <= 1)

        adversarial = x + delta

        assert torch.all(delta >= -self.eps - EPSILON) and torch.all(delta <= self.eps + EPSILON)
        assert torch.all(adversarial >= 0) and torch.all(adversarial <= 1)

        return adversarial, torch.zeros_like(y) + iter_number * self.k * 2


class PGDEnsembleAttack:
    """
    White-box L_inf PGD attack against an ensemble of models using the 
    cross-entropy loss
    """
    def __init__(self, models, eps=8/255., n=50, alpha=1/255.,
                 rand_init=True, early_stop=True):
        """
        Parameters:
        - models (a sequence): an ensemble of models to attack (i.e., the
              attack aims to decrease their expected loss)
        - eps: attack's maximum norm
        - n: max # attack iterations
        - alpha: PGD's step size at each iteration
        - rand_init: a flag denoting whether to randomly initialize
              adversarial samples in the range [x-eps, x+eps]
        - early_stop: a flag denoting whether to stop perturbing a 
              sample once the attack goal is met. If the goal is met
              for all samples in the batch, then the attack returns
              early, before completing all the iterations.
        """
        self.models = models
        self.n = n
        self.alpha = alpha
        self.eps = eps
        self.rand_init = rand_init
        self.early_stop = early_stop
        self.loss_func = nn.CrossEntropyLoss()

    def execute(self, x, y, targeted=False):
        """
        Executes the attack on a batch of samples x. y contains the true labels 
        in case of untargeted attacks, and the target labels in case of targeted 
        attacks. The method returns the adversarially perturbed samples, which
        lie in the ranges [0, 1] and [x-eps, x+eps].
        """
        if self.rand_init:
            delta = torch.rand_like(x, requires_grad=True)
            delta = (2 * delta - 1) * self.eps  # change noise to be in [-eps, eps]
        else:
            delta = torch.zeros_like(x, requires_grad=True)

        for _ in range(self.n):
            predictions = [model(x + delta) for model in self.models]

            if self.early_stop and all(classified_correctly(prediction, y, targeted) for prediction in predictions):
                break

            loss = sum(torch.nn.functional.cross_entropy(prediction, y) for prediction in predictions)
            loss = -1 * loss if targeted else loss

            grad = torch.autograd.grad(loss, delta)[0]

            delta += (self.alpha * torch.sign(grad))

            delta = torch.clamp(delta, -self.eps, self.eps)  # Make sure the perturbation is still at e magnitude
            delta = torch.clamp(x + delta, 0, 1) - x  # make sure the image is still a valid image

            assert torch.all(delta >= -self.eps - EPSILON) and torch.all(delta <= self.eps + EPSILON)
            assert torch.all(x + delta >= 0) and torch.all(x + delta <= 1)

        adversarial = x + delta

        assert torch.all(delta >= -self.eps - EPSILON) and torch.all(delta <= self.eps + EPSILON)
        assert torch.all(adversarial >= 0) and torch.all(adversarial <= 1)

        return adversarial
