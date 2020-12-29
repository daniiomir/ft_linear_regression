import csv
import random
import pickle
import argparse
import numpy as np


class LinearRegression:
    def __init__(self, learning_rate, loss, verbose):
        self.theta0 = 0.0
        self.theta1 = 0.0
        self.tmp_theta0 = random.random()
        self.tmp_theta1 = random.random()
        self.loss = loss
        self.lr = learning_rate
        self.verbose = verbose

        if self.lr > 1:
            self.lr = 0.1
            print('Warning! Too big learning rate. It was set to 0.1.')
        if self.lr < 0.000001:
            self.lr = 0.1
            print('Warning! Too small learning rate. It was set to 0.1.')

        if self.loss == 'mse':
            self.loss = MSELoss()
        elif self.loss == 'mae':
            self.loss = MAELoss()
        else:
            print(f'Error! Selected wrong loss function: {self.loss}')
            quit()

    def fit(self, data, target):
        delta_loss = 1
        epoch = 1
        tmp_loss = self.loss(target, self.tmp_predict(data))
        while abs(delta_loss) > 0.000001:
            self.tmp_theta0 -= self.lr * (self.tmp_predict(data) - target).mean()
            self.tmp_theta1 -= self.lr * ((self.tmp_predict(data) - target) * data).mean()
            new_tmp_loss = self.loss(target, self.tmp_predict(data))
            delta_loss = new_tmp_loss - tmp_loss
            tmp_loss = new_tmp_loss
            print(f'Epoch: {epoch}, loss: {tmp_loss}')
            self.theta0 = self.tmp_theta0
            self.theta1 = self.tmp_theta1
            epoch += 1
        if self.verbose:
            print(f'Fitted in {epoch}, theta0 - {self.theta0}, theta1 - {self.theta1}')

    def tmp_predict(self, data):
        return (self.tmp_theta1 * data) + self.tmp_theta0

    def predict(self, value):
        return (self.theta1 * float(value)) + self.theta0


class MSELoss:
    def __call__(self, real, pred):
        if isinstance(real, np.ndarray) and isinstance(pred, np.ndarray):
            return ((pred - real) ** 2).mean()
        else:
            print('Error! Targets are not in numpy array!')
            quit()


class MAELoss:
    def __call__(self, real, pred):
        if isinstance(real, np.ndarray) and isinstance(pred, np.ndarray):
            return (np.abs(pred - real)).mean()
        else:
            print('Error! Targets are not in numpy array!')
            quit()


def parse_args_fit():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', default='data.csv')
    parser.add_argument('--loss', default='mse')
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


def save_model(model, path):
    with open(path, 'wb') as f:
        pickle.dump(model, f, protocol=4)


def load_model(path):
    with open(path, 'rb') as f:
        model = pickle.load(f)
    return model


def read_data(path):
    km = np.array([])
    price = np.array([])
    with open(path, 'r') as f:
        for row in csv.DictReader(f):
            km = np.append(km, int(row['km']))
            price = np.append(price, int(row['price']))
    return km, price


def standart_scale(data):
    return (data - np.mean(data)) / np.std(data, ddof=1)
