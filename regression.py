import numpy as np
import pandas as pd 

def dataProcess(df):
    """数据预处理"""
    x_list, y_list = [], []
    # 将空数据填充为0
    df = df.replace(['NR'],[0.0])
    # 转换数据类型
    array = np.array(df).astype(float)
    # 将数据集拆分
    for i in range(0,4320,18):
        for j in range(24-9):
            mat = array[i:i+18, j:j+9]
            label = array[i+9, j+9]
            x_list.append(mat)
            y_list.append(label)
    x = np.array(x_list) 
    y = np.array(y_list) 

    return x, y, array

def train(x_train, y_train, epoch):
    """训练模型"""
    bias =  0 # 偏置值初始化
    weights = np.ones(9) # 权重初始化
    learning_rate = 1 # 初始学习率
    reg_rate = 0.001 #正则项系数
    bg2_sum = 0 #用于存放偏置值的梯度平方和
    wg2_sum = np.zeros(9) # 用于存放权重的梯度平方和

    for i in range(epoch):
        b_g = 0 # Loss对b的偏微分
        w_g = np.zeros(9) # Loss对w的偏微分
        # 在所有数据上计算Loss_label的梯度
        for j in range(3200):
            b_g += (y_train[j] - weights.dot(x_train[j,9,:]) - bias) * (-1)
            for k in range(9):
                w_g[k] += (y_train[j] - weights.dot(x_train[j,9,:]) - bias) * (-x_train[j,9,k])
        # 求平均
        b_g /= 3200
        weights /= 3200
        # 加上loss_regularization在w上的梯度
        for m in range(9):
            w_g[m] += reg_rate * weights[m]
        # adagrad
        bg2_sum += b_g**2
        wg2_sum += w_g**2
        # 更新权重和偏置
        bias -= learning_rate/bg2_sum**0.5 * b_g
        weights -= learning_rate/wg2_sum**0.5 * w_g
        if i%200 == 0:
            loss = 0
            for j in range(3200):
                loss += (y_train[j] - weights.dot(x_train[j,9,:])- bias)**2
            print('after {} epochs, the loss on train data is:'.format(i), loss/3200)
    return weights, bias

def validate(x_val, y_val, weights, bias):
    loss = 0
    for i in range(400):
        loss += (y_val[i] - weights.dot(x_val[i,9,:])-bias)**2
    return loss/400

def main():
     # 从csv中读取有用的信息
     # 由于大家获取数据集的渠道不同，所以数据集的编码格式可能不同
     # 若读取失败，可在参数栏中加入encoding = 'gb18030'
    df = pd.read_csv('train.csv', usecols=range(3,27), encoding = 'gb18030')
    x, y, _ = dataProcess(df)
    # 划分训练集和验证集
    x_train, y_train = x[0:3200], y[0:3200]
    x_val, y_val = x[3200:3600], y[3200:3600]
    epoch = 2000 # 训练轮数
    w, b = train(x_train, y_train, epoch)
    loss = validate(x_val, y_val, w, b)
    print('The loss on val data is', loss)

if __name__ == '__main__':
    main()

    
