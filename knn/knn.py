import numpy as np
import time
from tqdm import tqdm


def readdata(filename):
    dataArr = []
    labelArr = []

    fr = open(filename, "r")
    for line in tqdm(fr.readlines()):
        curline = line.strip().split(",")
        labelArr.append(int(curline[0]))
        dataArr.append([int(num)/255 for num in curline[1:]])
    return dataArr, labelArr

def caldis(x1, x2):
    # 计算欧式距离
    return np.sqrt(np.sum(np.square(x1-x2)))

def getcloest(traindatamat, trainlabelmat, x, topK):
    # 该函数作用是返回与x最近邻的那个类别
    m, n = traindatamat.shape
    distlist = [0] * m
    for i in range(m):
        x1 = traindatamat[i]
        x2 = x
        dis = caldis(x1, x2)
        distlist[i] = dis
    kindex = np.argsort(np.array(distlist))[:topK]

    klist = [0] * 10

    for i in range(topK):
        klist[int(trainlabelmat[kindex[i]])] += 1

    return klist.index(max(klist))



def model_test(traindata, trainlabel, testdata, testlabel, topK=25):
    traindatamat = np.mat(traindata)
    trainlabelmat = np.mat(trainlabel).T
    testdatamat = np.mat(testdata)
    testlabelmat = np.mat(testlabel).T

    err = 0

    for i in range(200):
        x = testdatamat[i]
        y = getcloest(traindatamat, trainlabelmat, x, topK)
        if y != int(testlabelmat[i]):
            err += 1
        print("bound {} / 200".format(i))

    acc = 1 - (err/200)

    return acc



if __name__ == "__main__":
    start_time = time.time()

    # 读数据
    traindata, trainlabel = readdata("../Mnist/mnist_train.csv")
    testdata, testlabel = readdata("../Mnist/mnist_test.csv")

    # 测试集
    acc = model_test(traindata, trainlabel, testdata, testlabel, topK=25)
    end_time = time.time()
    print("准确率为{}".format(acc))
    print("总计用时{}秒".format(end_time-start_time))
