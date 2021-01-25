# -*- coding: utf-8-*-
class confusionMatrix():
    """
    混淆矩阵
    """
    def __init__(self, tp, tn, fp, fn):
        self.tp = tp
        self.tn = tn
        self.fp = fp
        self.fn = fn

    def doPrecision(self):
        '''
        精确率：预测为正的样本中有多少是对的
        '''
        return self.tp / (self.tp + self.fp)

    def doRecall(self):
        '''
        召回率：正的样本中有多少被预测对了
        '''
        return self.tp / (self.tp + self.fn)

    def fScore(self,b):
        '''
        F2认为召回率权重高于精确率
        F0.5认为精确率权重高于召回率
        '''
        precision = self.doPrecision()
        recall = self.doRecall()
        return (1 + b*b)*precision*recall / (b*b*precision+recall)

    def acc(self,):
        return (self.tp + self.tn) / (self.tp + self.tn + self.fp + self.fn)

    def printInfo(self):
        # print('召回率=',self.doPrecision())
        # print('精确率=',self.doRecall())
        # print('模型准确率=',self.acc())
        print('F1-score=',self.fScore(1))
        # print('F2-score=',self.fScore(2))
        # print('F0.5-score=',self.fScore(0.5))

def caculate(tp,tn,fp,fn):
    print('Merger:')
    merger = confusionMatrix(tp,tn,fp,fn)
    merger.printInfo()
    # print('Galaxy:')
    # galaxy = confusionMatrix(tn,tp,fn,fp)
    # galaxy.printInfo()
    
if __name__ == "__main__":
    # tp:真阳 tn：真阴 fp：假阳 fn：假阴
    # caculate(tp= 313, tn= 19949, fp= 48, fn= 9)
    # caculate(tp= 313, tn= 19069, fp= 48, fn= 626)
    caculate(tp= 313, tn= 19600, fp= 95, fn= 9)
    # caculate(tp= 311, tn= 19615, fp= 80, fn= 11)