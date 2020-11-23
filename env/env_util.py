import math

import numpy as np
import scipy.signal


def dict_get(d, k, default=0.0):
    if k in d:
        return d[k]
    return default


def one_hot(size, index):
    # arr = np.zeros([size])
    arr = [0.0 for i in range(size)]
    arr[index] = 1.0
    return arr


def one_hot_none(size):
    # arr = np.zeros([size])
    arr = [0.0 for i in range(size)]
    return arr


def one_hot_categories(categories):
    def call(category):
        one_hots = [0. for _ in range(len(categories))]
        index = categories.index(category)
        one_hots[index] = 1.
        return one_hots
    return call


def scalar_normalize(max_, min_):
    def call(scaler):
        return (scaler - min_) / (max_ - min_)
    return call


def type_index(type_dict, name):
    for k, v in type_dict.items():
        if name.find(k) != -1:
            return v
    return -1


def calculate_2d_distance(pos1, pos2):
    dx = pos1[0] - pos2[0]
    dy = pos1[1] - pos2[1]
    return math.sqrt(dx * dx + dy * dy)


def calculate_3d_distance(pos1, pos2):
    dx = pos1[0] - pos2[0]
    dy = pos1[1] - pos2[1]
    dz = pos1[2] - pos2[2]
    return math.sqrt(dx * dx + dy * dy + dz * dz)


def normalize_point(point):
    length = math.sqrt(point[0] * point[0] + point[1] * point[1] + point[2] * point[2])
    return [point[0] / length, point[1] / length, point[2] / length]


def sigmoid(x):
    y = 1.0 / (1.0 + np.exp(-x))
    return y


def sample_units(logits):
    probs = sigmoid(logits)
    u = np.random.uniform(size=np.shape(probs))
    return probs >= u


def gumbel_action(logits):
    u = np.random.uniform(size=np.shape(logits))
    return np.argmax(logits - np.log(-np.log(u)), axis=-1)


def sample_actions(logits):
    select_units_logits = logits[0]
    action_logits = logits[1:]
    select_units = sample_units(select_units_logits)
    actions = []
    for logit in action_logits:
        actions.append(gumbel_action(logit))
    return select_units, actions


def discounted_sum(x, gamma):
    """ Calculate discounted forward sum of a sequence at each point
        out[i] = x[i] + gamma * x[i+1] + gamma^2 * x[i+2] + ...
    """
    return scipy.signal.lfilter([1.0], [1.0, -gamma], x[::-1], axis=0)[::-1]


def calc_gae(values, rewards, gamma, lam):
    """generalized advantage estimator.
    https://arxiv.org/pdf/1506.02438.pdf
    """
    # temporal differences
    tds = rewards[:-1] + gamma * values[1:] - values[:-1]
    tds = np.append(tds, rewards[-1:] - values[-1:], axis=0)
    advantages = discounted_sum(tds, gamma * lam)
    return advantages
