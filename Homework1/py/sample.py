def stocGradAscent1(dataMatrix, classLabels, numIter=150):
    m,n = shape(dataMatrix)
    weights = ones(n)   #initialize to all ones
    for j in range(numIter):   #增加迭代次数
        dataIndex = list(range(m))     #这里应该修改
        for i in range(m):
            alpha = 4/(1.0+j+i)+0.0001    #alpha在每次迭代中都会减小，但是不会为0，这样保证在多次迭代之后新数据仍然对参数有影响
            randIndex = int(random.uniform(0,len(dataIndex))) #随机抽取样本来更新参数
            h = sigmoid(sum(dataMatrix[randIndex]*weights))
            error = classLabels[randIndex] - h
            weights = weights + alpha * error * dataMatrix[randIndex]
            del(dataIndex[randIndex])
    return weights

#classifyVector的第一个参数为回归系数,weight为特征向量，这里将输入这两个参数来计算sigmoid值，如果只大于0.5，则返回1，否则返回0.
def classifyVector(inX, weights):
    prob = sigmoid(sum(inX*weights))
    if prob > 0.5: return 1.0    #如果不想得到分类，则直接返回prob概率
    else: return 0.0
 
#打开数据集并对数据集进行处理
def colicTest():
    frTrain = open('horseColicTraining.txt'); frTest = open('horseColicTest.txt')
    trainingSet = []; trainingLabels = []
    
    #获取训练集的数据，并将其存放在list中
    for line in frTrain.readlines():
        currLine = line.strip().split('\t')
        lineArr =[]            #用于存放每一行的数据
        for i in range(21):    #这里的range(21)是为了循环每一列的值，总共有22列
            lineArr.append(float(currLine[i]))
        trainingSet.append(lineArr)
        trainingLabels.append(float(currLine[21]))    
    trainWeights = stocGradAscent1(array(trainingSet), trainingLabels, 50)   #用改进的随机梯度算法计算回归系数
   
    #计算测试集的错误率
    errorCount = 0; numTestVec = 0.0
    for line in frTest.readlines():
        numTestVec += 1.0
        currLine = line.strip().split('\t')
        lineArr =[]
        for i in range(21):
            lineArr.append(float(currLine[i]))
            #如果预测值和实际值不相同，则令错误个数加1
        if int(classifyVector(array(lineArr), trainWeights))!= int(currLine[21]):
            errorCount += 1
    errorRate = (float(errorCount)/numTestVec)    #最后计算总的错误率
    print ("the error rate of this test is: %f" % errorRate)
    return errorRate
 
#调用coicTest函数10次并求平均值
def multiTest():
    numTests = 10; errorSum=0.0
    for k in range(numTests):
        errorSum += colicTest()
    print ("after %d iterations the average error rate is: %f" % (numTests, errorSum/float(numTests)))
 
multiTest()
