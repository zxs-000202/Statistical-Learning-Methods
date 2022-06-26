import numpy as np
import time
from tqdm import tqdm
def readdata(filename):
    print("start read data")
    dataArr = []
    labelArr = []
    fr = open(filename, "r")
    for line in tqdm(fr.readlines()):
        curline = line.strip().split(',')
        if int(curline[0])>=5:
            labelArr.append(1)
        else:
            labelArr.append(-1)
        dataArr.append([int(num)/255 for num in curline[1:]])

    return dataArr, labelArr

def train_model(traindata, trainlabel, iter = 30):
    print("start train model")
    datamat = np.mat(traindata)
    labelmat = np.mat(trainlabel).T
    m, n = datamat.shape
    w = np.zeros((1,n))
    b = 0
    lr = 0.0001
    for i in range(iter):
        for j in range(m):
            xi = datamat[j] # 1*784
            yi = labelmat[j] # 1*1
            if -1 * yi * (w * xi.T + b) >= 0:
                w = w + lr * yi * xi
                b = b + lr * yi
        print("round {} : 30".format(i+1))

    return w, b

def model_test(testdata, testlabel, w, b):
    print("start test model")
    datamat = np.mat(testdata)
    labelmat = np.mat(testlabel).T
    m, n = datamat.shape
    err = 0
    for i in range(m):
        xi = datamat[i]
        yi = labelmat[i]
        if -1 * yi * (w * xi.T + b) >= 0:
            err += 1
    acc = 1 - (err/m)

    return acc

if __name__ == "__main__":
    start_time = time.time()
    # 读取数据
    traindata, trainlabel = readdata("../Mnist/mnist_train.csv")
    testdata, testlabel =readdata("../Mnist/mnist_test.csv")
    # 训练模型
    w, b = train_model(traindata, trainlabel, iter = 30)
    # 测试模型
    acc = model_test(testdata, testlabel, w, b)

    end_time = time.time()
    print("总用时{}s".format(end_time-start_time))
    print("准确率为{}".format(acc))