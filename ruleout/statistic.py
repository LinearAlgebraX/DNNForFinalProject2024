import pandas as pd
import numpy as np
import glob
import os

def apfd(right,sort):
    length=np.sum(sort!=0)
    if length!=len(sort):
        sort[sort==0]=np.random.permutation(len(sort)-length)+length+1
    sum_all=np.sum(sort.values[right.values!=1])
    n=len(sort)
    m=pd.value_counts(right)[0]
    return 1-float(sum_all)/(n*m)+1./(2*n)


if __name__=='__main__':
    lst=glob.glob('./output_mnist/mnist_deep_metric.csv')
    data_dict={}
    for i in lst:
        name=os.path.basename(i)[:-4]
        data_dict[name]=pd.read_csv(i,index_col=0)

    for key in data_dict.keys():
        print(key)
        print('misclassified tests:{}'.format(pd.value_counts(data_dict[key].right)[0]))
        print('total tests:{}'.format(len(data_dict[key])))
        print('APFD:{}'.format(apfd(data_dict[key].right,data_dict[key].cam)))
        print('==============================')
