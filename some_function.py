import socket
import datetime
import time as t

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.gaussian_process import GaussianProcessRegressor
import sklearn.gaussian_process.kernels as skk
from sklearn import linear_model


def convey_params_get_cost(params):  # 向labview传递实验参数，接受实验结果
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

    # 获取本地主机名
    host = socket.gethostname()

    # 设置端口号
    port = 8888

    # 连接服务，指定主机和端口

    s.connect((host, port))
    print('connected')
    data = ''
    for j in range(len(params)):
        data = data + str(params[j]) + ' '
    # str_para = str(parameter)
    # data = str_para + str('%04d' % len(str_para))
    s.send(data.encode('utf-8'))
    print('send parameter', data)

    # 接收小于 1024 字节的数据
    msg = s.recv(1024)
    cost = msg.decode('utf-8')  # 实验成本函数，可以设定为要优化的物理量或者与其相关的一个函数
    cost = float(cost)
    print('received cost', cost)
    s.close()
    return cost

def array_to_csv(filename, array):
    data = pd.DataFrame(array)
    data.to_csv(filename, header=0, index=0)

def today_str():
    year = t.strftime('%Y',t.localtime())
    month = t.strftime('%m',t.localtime())
    day = t.strftime('%d',t.localtime())
    weekday = t.strftime('%A',t.localtime())
    if day[0]=='0':
        day = day[1]
    if month[0] == '0':
        month = month[1]
    today = year+'年'+month+'月'+day+'日'+' '+get_week_day(date=datetime.date.today())
    return today

def get_week_day(date):
    week_day = {
        0: '星期一',
        1: '星期二',
        2: '星期三',
        3: '星期四',
        4: '星期五',
        5: '星期六',
        6: '星期日',
    }
    day = date.weekday()  # weekday()可以获得是星期几
    return week_day[day]

def gaussian_interpolation(params, a=0, b=100, m=200, show_plot=True):
    x_train = np.atleast_2d(np.linspace(a, b, len(params))).T
    y_train = params
    x_test = np.atleast_2d(np.linspace(0, 100, m)).T

    kernel = skk.RBF()
    model = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=100, normalize_y=True)

    model.fit(x_train, y_train)
    y_pred, sigma_pred = model.predict(x_test, return_std=True)

    plt.figure(figsize=(16, 9))
    plt.plot(x_train, y_train,linewidth='0.5', color='#000000')
    plt.scatter(x_test, y_pred)
    plt.scatter(x_test, y_pred+sigma_pred, c='#00CED1')
    plt.scatter(x_test, y_pred-sigma_pred, c='#DC143C')
    if show_plot:
        plt.show()

    return y_pred[1:m-1], sigma_pred[1:m-1]

def linear_regre_1d(x,y):
    x = np.atleast_2d(x).T
    y = np.atleast_2d(y).T

    model = linear_model.LinearRegression()
    model.fit(x, y)

    k = model.coef_[0][0]
    b = model.intercept_[0]
    return k, b

def add_zero(array):
    z = np.array([0.0])
    array = np.concatenate([z, array, z])
    return array

def change_colon_into_dash(start_datetime):
    start_time = ''
    for x in start_datetime:
        if x == ':':
            x = '-'
        start_time += x
    return start_time

def wave_normalization(wave):
    # wave -= np.min(wave)
    '''
    m = np.max(wave)
    gate = 0.02*m
    wave = wave[np.where(wave>gate)]
    '''
    wave = add_zero(wave)
    return wave / np.mean(wave) / 3

def estim_noise_uncer(res):
    n = len(res)
    if n < 10:
        a = 3
    else:
        a = 5
    l = np.array_split(res, a)
    ans = []
    for i in l:
        ans.append(np.var(i))
    ans = np.array(ans)
    return np.std(ans)