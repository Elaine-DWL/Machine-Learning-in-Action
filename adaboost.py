import numpy as np
def loadSimpData():
    dataMat = np.array([[1, 2.1],
        [2, 1.1],
        [1.3, 1],
        [1, 1],
        [2, 1]])
    classLabels = [1.0, 1.0, -1, -1, 1]
    return dataMat, classLabels


datMat, classLabels = loadSimpData() # 数据导入
