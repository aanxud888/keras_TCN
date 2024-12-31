if __name__ == '__main__':
    import numpy as np
    tcnlayer = TCN()
    out = tcnlayer(np.zeros((1,5,32))) # [bs,seql,dim]
    print(out.shape)

当前的参数将会使感受野提升4倍，即输出时间维度一个时刻能够反应其之前4个时刻的特征
The current parameter will increase the receptive field by 4 times, which means that the output time dimension can reflect the features of 4 times before it at one moment
(None, 5, 32)
核心思路：使用valid卷积，卷积核大小和stride大小取相同的值，Conv1d只会沿着一个方向（序列正方向）进行移动，因此卷积核计算的特征具有因果特性（与pading=='causal'效果一样）。每经过一层卷积，得到的每个时刻就代表一个kernel_size个感受野。
