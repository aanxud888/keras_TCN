from keras.layers import  Lambda,Dense,Layer,Conv1D
import tensorflow as tf

class TCNCell(Layer):
    """
    sumary_line:
    Chinese:让输入的时间序列[bs,seql,dim]提升kernel_size倍的感受野
    English: Double the receptive field of the input time series [bs, seql, dim]
    """
    def __init__(self, filters=32,ks=3,activation=None,name=None):
        self.filters = filters
        self.ks = ks
        self.activation = activation
        super(TCNCell, self).__init__(name=name)


    def build(self, input_shape):
        assert len(input_shape) == 3, f"Input shape should be [batch, timesteps, features], but got {input_shape}"
        self.input_shape = input_shape
        bs,seq_l,dim = input_shape
        if input_shape[1]==1:
            self.out = Dense(self.filters,activation='relu')
        else:
            if not seq_l%self.ks == 0:
                self.maxlen = seq_l+self.ks-seq_l%self.ks
                self.pad_layer = Lambda(lambda x: tf.pad(tensor=x, paddings=[[0,0],[self.maxlen-seq_l, 0], [0, 0]], constant_values=0),output_shape=(self.maxlen,dim))
                assert self.maxlen%self.ks == 0, 'kernel size should be divisible by input length'
            self.tcn_cell = Conv1D(filters=self.filters, kernel_size=self.ks, strides=self.ks,activation=self.activation,padding='valid')
        super(TCNCell, self).build(input_shape)
    
    def call(self,x):
        if x.shape[1]==1 and hasattr(self,'out'):
            return self.out(x)
        else:
            if hasattr(self, 'pad_layer') and hasattr(self,'maxlen'):
                x = self.pad_layer(x)
                x = self.tcn_cell(x)
                return x
            else:
                return self.tcn_cell(x)
    

    
class TCN(Layer):

    """
    input: (batch_size,seq_len,feature_dim)
    output: (batch_size,output_len,feature_dim)
    """

    def __init__(self,filters_list=[32,64,128],kernel_size_list=[3,3,3],seq_len=32,name='TCN'):
        assert len(filters_list) == len(kernel_size_list), "filters_list and kernel_size_list must have the same length"
        self.l = len(filters_list)
        assert seq_len is not None and seq_len > 2**self.l, f"seql is None or receptive field must be smaller than squence length, please check"
        self.filters_list = filters_list
        self.kernel_size_list = kernel_size_list
        self.seql = seq_len
        self.print_receptive_field()
        super(TCN,self).__init__(name=name)

    def cala_receptive_field(self):
        ce_list = []
        for idx,ks in enumerate(self.kernel_size_list):
            if idx == 0:
                ce_list.append(ks)
            else:
                ce_list.append(ce_list[-1]*ks)
        return ce_list[-1]



    def print_receptive_field(self):
        ce = self.cala_receptive_field()
        print(f'当前的参数将会使感受野提升{ce}倍，即输出时间维度一个时刻能够反应其之前{ce}个时刻的特征')
        print(f'The current parameter will increase the receptive field by {ce} times,' + ' '+
              f'which means that the output time dimension can reflect the features of {ce} times before it at one moment')


    def build(self, input_shape):
        bs,seql,dim  = input_shape
        assert seql==self.seql, f'输入序列长度{seql}与设定的序列长度{self.seql}不一致' + ' ' + f'The input sequence length {seql} does not match the set sequence length {self.seql}'
        self.tcn_cell_layers = []
        for i in range(self.l):
            self.tcn_cell_layers.append(
                TCNCell(filters=self.filters_list[i],ks=self.kernel_size_list[i])
            )
        super(TCN, self).build(input_shape)
    
    def call(self,x):
        for i in range(self.l):
            x = self.tcn_cell_layers[i](x)
        return x
    

    
if __name__ == '__main__':
    import numpy as np
    tcnlayer = TCN()
    out = tcnlayer(np.zeros((1,32,768)))
    print(out.shape)
