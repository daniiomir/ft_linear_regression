import csv
import random
import pickle
import argparse
import numpy as np
from typing import Union


class LinearRegression:
    def __init__(self, learning_rate: float, verbose: int):
        self.lr = learning_rate
        self.verbose = verbose
        self.theta0 = 0.0
        self.theta1 = 0.0
        self.tmp_theta0 = random.random()
        self.tmp_theta1 = random.random()
        self.data_mean = 0.0
        self.data_std = 0.0

        if self.lr > 1:
            self.lr = 0.1
            print('Warning! Too big learning rate. It was set to 0.1.')
        if self.lr < 0.000001:
            self.lr = 0.1
            print('Warning! Too small learning rate. It was set to 0.1.')

    def fit(self, data: np.ndarray, target: np.ndarray):
        delta_loss = 1
        epoch = 1
        data = self.scale_data(data)
        tmp_loss = self.loss(target, self.tmp_predict(data))
        while abs(delta_loss) > 0.000001:
            self.tmp_theta0 -= self.lr * (self.tmp_predict(data) - target).mean()
            self.tmp_theta1 -= self.lr * ((self.tmp_predict(data) - target) * data).mean()
            new_tmp_loss = self.loss(target, self.tmp_predict(data))
            delta_loss = new_tmp_loss - tmp_loss
            tmp_loss = new_tmp_loss
            print('Epoch: {}, loss: {}'.format(epoch, tmp_loss))
            self.theta0 = self.tmp_theta0
            self.theta1 = self.tmp_theta1
            epoch += 1
        if self.verbose:
            print('Fitted in {}, theta0 - {}, theta1 - {}'.format(epoch, self.theta0, self.theta1))

    def loss(self, real: np.ndarray, pred: np.ndarray) -> float:
        return ((pred - real) ** 2).mean()

    def tmp_predict(self, data: np.ndarray):
        return (self.tmp_theta1 * data) + self.tmp_theta0

    def predict(self, value: Union[int, float]):
        if value > 0:
            return int((self.theta1 * self.scale_data(value, False)) + self.theta0)
        raise Exception('Error! Wrong input value - {}. Value should be number bigger than zero.'.format(value))

    def scale_data(self, data: Union[np.ndarray, int, float], is_train: bool = True):
        if is_train:
            self.data_mean = np.mean(data)
            self.data_std = np.std(data, ddof=1)
        return (data - self.data_mean) / self.data_std


def parse_args_fit():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', default='data.csv')
    parser.add_argument('--learning_rate', default=0.1, type=float)
    parser.add_argument('--save_path', default='model.pkl')
    parser.add_argument('--verbose', default=1, type=int)
    args = parser.parse_args()
    return args.__dict__


def parse_args_predict():
    parser = argparse.ArgumentParser()
    parser.add_argument('--load_path', default='model.pkl')
    parser.add_argument('--input_value', default=-1, type=int)
    args = parser.parse_args()
    return args.__dict__


def parse_args_visualize():
    parser = argparse.ArgumentParser()
    parser.add_argument('--load_path', default='model.pkl')
    parser.add_argument('--data_path', default='data.csv')
    args = parser.parse_args()
    return args.__dict__


def save_model(model: object, path: str):
    with open(path, 'wb') as f:
        pickle.dump(model, f, protocol=4)


def load_model(path: str):
    with open(path, 'rb') as f:
        model = pickle.load(f)
    return model


def read_data(path: str):
    km = np.array([])
    price = np.array([])
    with open(path, 'r') as f:
        for row in csv.DictReader(f):
            km = np.append(km, int(row['km']))
            price = np.append(price, int(row['price']))
    return km, price
