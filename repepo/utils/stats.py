from torch.distributions.bernoulli import Bernoulli
from torch.distributions.kl import kl_divergence


def bernoulli_kl_dist(base_prob: float, steered_prob: float) -> float:
    """
    Use KL divergence to measure the distance between two Bernoulli distributions
    If steered prob is less than the base prob, returns a negative value
    """
    base = Bernoulli(base_prob)
    steered = Bernoulli(steered_prob)
    kl_div = kl_divergence(base, steered).item()
    return kl_div if steered_prob > base_prob else -kl_div


def bernoulli_js_dist(base_prob: float, steered_prob: float) -> float:
    """
    Use Jenson-shannon distance to measure the distance between two Bernoulli distributions
    If steered prob is less than the base prob, returns a negative value
    """
    base = Bernoulli(base_prob)
    steered = Bernoulli(steered_prob)
    mid = Bernoulli((base_prob + steered_prob) / 2)
    val = (0.5 * kl_divergence(base, mid) + 0.5 * kl_divergence(steered, mid)).item()
    return val if steered_prob > base_prob else -val
