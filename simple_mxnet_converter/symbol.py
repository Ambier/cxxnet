import mxnet as mx
data = mx.symbol.Variable(name='data')
conv_3a_1x1 = mx.symbol.Convolution(name='conv_3a_1x1',kernel=(1, 1),num_filter=64,pad=(0, 0),stride=(1, 1),data=data)
bn_3a_1x1 = mx.symbol.BatchNorm(name='bn_3a_1x1',data=conv_3a_1x1)
relu_3a_1x1 = mx.symbol.Activation(act_type='relu',name='relu_3a_1x1',data=bn_3a_1x1)
conv_3a_3x3_reduce = mx.symbol.Convolution(name='conv_3a_3x3_reduce',kernel=(1, 1),num_filter=64,pad=(0, 0),stride=(1, 1),data=data)
bn_3a_3x3_reduce = mx.symbol.BatchNorm(name='bn_3a_3x3_reduce',data=conv_3a_3x3_reduce)
relu_3a_3x3_reduce = mx.symbol.Activation(act_type='relu',name='relu_3a_3x3_reduce',data=bn_3a_3x3_reduce)
conv_3a_3x3 = mx.symbol.Convolution(name='conv_3a_3x3',kernel=(3, 3),num_filter=64,pad=(1, 1),stride=(1, 1),data=relu_3a_3x3_reduce)
bn_3a_3x3 = mx.symbol.BatchNorm(name='bn_3a_3x3',data=conv_3a_3x3)
relu_3a_3x3 = mx.symbol.Activation(act_type='relu',name='relu_3a_3x3',data=bn_3a_3x3)
conv_3a_double_3x3_reduce = mx.symbol.Convolution(name='conv_3a_double_3x3_reduce',kernel=(1, 1),num_filter=64,pad=(0, 0),stride=(1, 1),data=data)
bn_3a_double_3x3_reduce = mx.symbol.BatchNorm(name='bn_3a_double_3x3_reduce',data=conv_3a_double_3x3_reduce)
relu_3a_double_3x3_reduce = mx.symbol.Activation(act_type='relu',name='relu_3a_double_3x3_reduce',data=bn_3a_double_3x3_reduce)
conv_3a_double_3x3_0 = mx.symbol.Convolution(name='conv_3a_double_3x3_0',kernel=(3, 3),num_filter=96,pad=(1, 1),stride=(1, 1),data=relu_3a_double_3x3_reduce)
bn_3a_double_3x3_0 = mx.symbol.BatchNorm(name='bn_3a_double_3x3_0',data=conv_3a_double_3x3_0)
relu_3a_double_3x3_0 = mx.symbol.Activation(act_type='relu',name='relu_3a_double_3x3_0',data=bn_3a_double_3x3_0)
conv_3a_double_3x3_1 = mx.symbol.Convolution(name='conv_3a_double_3x3_1',kernel=(3, 3),num_filter=96,pad=(1, 1),stride=(1, 1),data=relu_3a_double_3x3_0)
bn_3a_double_3x3_1 = mx.symbol.BatchNorm(name='bn_3a_double_3x3_1',data=conv_3a_double_3x3_1)
relu_3a_double_3x3_1 = mx.symbol.Activation(act_type='relu',name='relu_3a_double_3x3_1',data=bn_3a_double_3x3_1)
avg_pool_3a_pool = mx.symbol.Pooling(pool_type='avg',name='avg_pool_3a_pool',kernel=(3, 3),stride=(1, 1),pad=(1, 1),data=data)
conv_3a_proj = mx.symbol.Convolution(name='conv_3a_proj',kernel=(1, 1),num_filter=32,pad=(0, 0),stride=(1, 1),data=avg_pool_3a_pool)
bn_3a_proj = mx.symbol.BatchNorm(name='bn_3a_proj',data=conv_3a_proj)
relu_3a_proj = mx.symbol.Activation(act_type='relu',name='relu_3a_proj',data=bn_3a_proj)
ch_concat_3a_chconcat = mx.symbol.Concat(name='ch_concat_3a_chconcat',*[relu_3a_1x1,relu_3a_3x3,relu_3a_double_3x3_1,relu_3a_proj])
conv_3b_1x1 = mx.symbol.Convolution(name='conv_3b_1x1',kernel=(1, 1),num_filter=64,pad=(0, 0),stride=(1, 1),data=ch_concat_3a_chconcat)
bn_3b_1x1 = mx.symbol.BatchNorm(name='bn_3b_1x1',data=conv_3b_1x1)
relu_3b_1x1 = mx.symbol.Activation(act_type='relu',name='relu_3b_1x1',data=bn_3b_1x1)
conv_3b_3x3_reduce = mx.symbol.Convolution(name='conv_3b_3x3_reduce',kernel=(1, 1),num_filter=64,pad=(0, 0),stride=(1, 1),data=ch_concat_3a_chconcat)
bn_3b_3x3_reduce = mx.symbol.BatchNorm(name='bn_3b_3x3_reduce',data=conv_3b_3x3_reduce)
relu_3b_3x3_reduce = mx.symbol.Activation(act_type='relu',name='relu_3b_3x3_reduce',data=bn_3b_3x3_reduce)
conv_3b_3x3 = mx.symbol.Convolution(name='conv_3b_3x3',kernel=(3, 3),num_filter=96,pad=(1, 1),stride=(1, 1),data=relu_3b_3x3_reduce)
bn_3b_3x3 = mx.symbol.BatchNorm(name='bn_3b_3x3',data=conv_3b_3x3)
relu_3b_3x3 = mx.symbol.Activation(act_type='relu',name='relu_3b_3x3',data=bn_3b_3x3)
conv_3b_double_3x3_reduce = mx.symbol.Convolution(name='conv_3b_double_3x3_reduce',kernel=(1, 1),num_filter=64,pad=(0, 0),stride=(1, 1),data=ch_concat_3a_chconcat)
bn_3b_double_3x3_reduce = mx.symbol.BatchNorm(name='bn_3b_double_3x3_reduce',data=conv_3b_double_3x3_reduce)
relu_3b_double_3x3_reduce = mx.symbol.Activation(act_type='relu',name='relu_3b_double_3x3_reduce',data=bn_3b_double_3x3_reduce)
conv_3b_double_3x3_0 = mx.symbol.Convolution(name='conv_3b_double_3x3_0',kernel=(3, 3),num_filter=96,pad=(1, 1),stride=(1, 1),data=relu_3b_double_3x3_reduce)
bn_3b_double_3x3_0 = mx.symbol.BatchNorm(name='bn_3b_double_3x3_0',data=conv_3b_double_3x3_0)
relu_3b_double_3x3_0 = mx.symbol.Activation(act_type='relu',name='relu_3b_double_3x3_0',data=bn_3b_double_3x3_0)
conv_3b_double_3x3_1 = mx.symbol.Convolution(name='conv_3b_double_3x3_1',kernel=(3, 3),num_filter=96,pad=(1, 1),stride=(1, 1),data=relu_3b_double_3x3_0)
bn_3b_double_3x3_1 = mx.symbol.BatchNorm(name='bn_3b_double_3x3_1',data=conv_3b_double_3x3_1)
relu_3b_double_3x3_1 = mx.symbol.Activation(act_type='relu',name='relu_3b_double_3x3_1',data=bn_3b_double_3x3_1)
avg_pool_3b_pool = mx.symbol.Pooling(pool_type='avg',name='avg_pool_3b_pool',kernel=(3, 3),stride=(1, 1),pad=(1, 1),data=ch_concat_3a_chconcat)
conv_3b_proj = mx.symbol.Convolution(name='conv_3b_proj',kernel=(1, 1),num_filter=64,pad=(0, 0),stride=(1, 1),data=avg_pool_3b_pool)
bn_3b_proj = mx.symbol.BatchNorm(name='bn_3b_proj',data=conv_3b_proj)
relu_3b_proj = mx.symbol.Activation(act_type='relu',name='relu_3b_proj',data=bn_3b_proj)
ch_concat_3b_chconcat = mx.symbol.Concat(name='ch_concat_3b_chconcat',*[relu_3b_1x1,relu_3b_3x3,relu_3b_double_3x3_1,relu_3b_proj])
conv_3c_3x3_reduce = mx.symbol.Convolution(name='conv_3c_3x3_reduce',kernel=(1, 1),num_filter=128,pad=(0, 0),stride=(1, 1),data=ch_concat_3b_chconcat)
bn_3c_3x3_reduce = mx.symbol.BatchNorm(name='bn_3c_3x3_reduce',data=conv_3c_3x3_reduce)
relu_3c_3x3_reduce = mx.symbol.Activation(act_type='relu',name='relu_3c_3x3_reduce',data=bn_3c_3x3_reduce)
conv_3c_3x3 = mx.symbol.Convolution(name='conv_3c_3x3',kernel=(3, 3),num_filter=160,pad=(1, 1),stride=(2, 2),data=relu_3c_3x3_reduce)
bn_3c_3x3 = mx.symbol.BatchNorm(name='bn_3c_3x3',data=conv_3c_3x3)
relu_3c_3x3 = mx.symbol.Activation(act_type='relu',name='relu_3c_3x3',data=bn_3c_3x3)
conv_3c_double_3x3_reduce = mx.symbol.Convolution(name='conv_3c_double_3x3_reduce',kernel=(1, 1),num_filter=64,pad=(0, 0),stride=(1, 1),data=ch_concat_3b_chconcat)
bn_3c_double_3x3_reduce = mx.symbol.BatchNorm(name='bn_3c_double_3x3_reduce',data=conv_3c_double_3x3_reduce)
relu_3c_double_3x3_reduce = mx.symbol.Activation(act_type='relu',name='relu_3c_double_3x3_reduce',data=bn_3c_double_3x3_reduce)
conv_3c_double_3x3_0 = mx.symbol.Convolution(name='conv_3c_double_3x3_0',kernel=(3, 3),num_filter=96,pad=(1, 1),stride=(1, 1),data=relu_3c_double_3x3_reduce)
bn_3c_double_3x3_0 = mx.symbol.BatchNorm(name='bn_3c_double_3x3_0',data=conv_3c_double_3x3_0)
relu_3c_double_3x3_0 = mx.symbol.Activation(act_type='relu',name='relu_3c_double_3x3_0',data=bn_3c_double_3x3_0)
conv_3c_double_3x3_1 = mx.symbol.Convolution(name='conv_3c_double_3x3_1',kernel=(3, 3),num_filter=96,pad=(1, 1),stride=(2, 2),data=relu_3c_double_3x3_0)
bn_3c_double_3x3_1 = mx.symbol.BatchNorm(name='bn_3c_double_3x3_1',data=conv_3c_double_3x3_1)
relu_3c_double_3x3_1 = mx.symbol.Activation(act_type='relu',name='relu_3c_double_3x3_1',data=bn_3c_double_3x3_1)
max_pool_3c_pool = mx.symbol.Pooling(pool_type='max',name='max_pool_3c_pool',kernel=(3, 3),stride=(2, 2),data=ch_concat_3b_chconcat)
ch_concat_3c_chconcat = mx.symbol.Concat(name='ch_concat_3c_chconcat',*[relu_3c_3x3,relu_3c_double_3x3_1,max_pool_3c_pool])
conv_4a_1x1 = mx.symbol.Convolution(name='conv_4a_1x1',kernel=(1, 1),num_filter=224,pad=(0, 0),stride=(1, 1),data=ch_concat_3c_chconcat)
bn_4a_1x1 = mx.symbol.BatchNorm(name='bn_4a_1x1',data=conv_4a_1x1)
relu_4a_1x1 = mx.symbol.Activation(act_type='relu',name='relu_4a_1x1',data=bn_4a_1x1)
conv_4a_3x3_reduce = mx.symbol.Convolution(name='conv_4a_3x3_reduce',kernel=(1, 1),num_filter=64,pad=(0, 0),stride=(1, 1),data=ch_concat_3c_chconcat)
bn_4a_3x3_reduce = mx.symbol.BatchNorm(name='bn_4a_3x3_reduce',data=conv_4a_3x3_reduce)
relu_4a_3x3_reduce = mx.symbol.Activation(act_type='relu',name='relu_4a_3x3_reduce',data=bn_4a_3x3_reduce)
conv_4a_3x3 = mx.symbol.Convolution(name='conv_4a_3x3',kernel=(3, 3),num_filter=96,pad=(1, 1),stride=(1, 1),data=relu_4a_3x3_reduce)
bn_4a_3x3 = mx.symbol.BatchNorm(name='bn_4a_3x3',data=conv_4a_3x3)
relu_4a_3x3 = mx.symbol.Activation(act_type='relu',name='relu_4a_3x3',data=bn_4a_3x3)
conv_4a_double_3x3_reduce = mx.symbol.Convolution(name='conv_4a_double_3x3_reduce',kernel=(1, 1),num_filter=96,pad=(0, 0),stride=(1, 1),data=ch_concat_3c_chconcat)
bn_4a_double_3x3_reduce = mx.symbol.BatchNorm(name='bn_4a_double_3x3_reduce',data=conv_4a_double_3x3_reduce)
relu_4a_double_3x3_reduce = mx.symbol.Activation(act_type='relu',name='relu_4a_double_3x3_reduce',data=bn_4a_double_3x3_reduce)
conv_4a_double_3x3_0 = mx.symbol.Convolution(name='conv_4a_double_3x3_0',kernel=(3, 3),num_filter=128,pad=(1, 1),stride=(1, 1),data=relu_4a_double_3x3_reduce)
bn_4a_double_3x3_0 = mx.symbol.BatchNorm(name='bn_4a_double_3x3_0',data=conv_4a_double_3x3_0)
relu_4a_double_3x3_0 = mx.symbol.Activation(act_type='relu',name='relu_4a_double_3x3_0',data=bn_4a_double_3x3_0)
conv_4a_double_3x3_1 = mx.symbol.Convolution(name='conv_4a_double_3x3_1',kernel=(3, 3),num_filter=128,pad=(1, 1),stride=(1, 1),data=relu_4a_double_3x3_0)
bn_4a_double_3x3_1 = mx.symbol.BatchNorm(name='bn_4a_double_3x3_1',data=conv_4a_double_3x3_1)
relu_4a_double_3x3_1 = mx.symbol.Activation(act_type='relu',name='relu_4a_double_3x3_1',data=bn_4a_double_3x3_1)
avg_pool_4a_pool = mx.symbol.Pooling(pool_type='avg',name='avg_pool_4a_pool',kernel=(3, 3),stride=(1, 1),pad=(1, 1),data=ch_concat_3c_chconcat)
conv_4a_proj = mx.symbol.Convolution(name='conv_4a_proj',kernel=(1, 1),num_filter=128,pad=(0, 0),stride=(1, 1),data=avg_pool_4a_pool)
bn_4a_proj = mx.symbol.BatchNorm(name='bn_4a_proj',data=conv_4a_proj)
relu_4a_proj = mx.symbol.Activation(act_type='relu',name='relu_4a_proj',data=bn_4a_proj)
ch_concat_4a_chconcat = mx.symbol.Concat(name='ch_concat_4a_chconcat',*[relu_4a_1x1,relu_4a_3x3,relu_4a_double_3x3_1,relu_4a_proj])
conv_4b_1x1 = mx.symbol.Convolution(name='conv_4b_1x1',kernel=(1, 1),num_filter=192,pad=(0, 0),stride=(1, 1),data=ch_concat_4a_chconcat)
bn_4b_1x1 = mx.symbol.BatchNorm(name='bn_4b_1x1',data=conv_4b_1x1)
relu_4b_1x1 = mx.symbol.Activation(act_type='relu',name='relu_4b_1x1',data=bn_4b_1x1)
conv_4b_3x3_reduce = mx.symbol.Convolution(name='conv_4b_3x3_reduce',kernel=(1, 1),num_filter=96,pad=(0, 0),stride=(1, 1),data=ch_concat_4a_chconcat)
bn_4b_3x3_reduce = mx.symbol.BatchNorm(name='bn_4b_3x3_reduce',data=conv_4b_3x3_reduce)
relu_4b_3x3_reduce = mx.symbol.Activation(act_type='relu',name='relu_4b_3x3_reduce',data=bn_4b_3x3_reduce)
conv_4b_3x3 = mx.symbol.Convolution(name='conv_4b_3x3',kernel=(3, 3),num_filter=128,pad=(1, 1),stride=(1, 1),data=relu_4b_3x3_reduce)
bn_4b_3x3 = mx.symbol.BatchNorm(name='bn_4b_3x3',data=conv_4b_3x3)
relu_4b_3x3 = mx.symbol.Activation(act_type='relu',name='relu_4b_3x3',data=bn_4b_3x3)
conv_4b_double_3x3_reduce = mx.symbol.Convolution(name='conv_4b_double_3x3_reduce',kernel=(1, 1),num_filter=96,pad=(0, 0),stride=(1, 1),data=ch_concat_4a_chconcat)
bn_4b_double_3x3_reduce = mx.symbol.BatchNorm(name='bn_4b_double_3x3_reduce',data=conv_4b_double_3x3_reduce)
relu_4b_double_3x3_reduce = mx.symbol.Activation(act_type='relu',name='relu_4b_double_3x3_reduce',data=bn_4b_double_3x3_reduce)
conv_4b_double_3x3_0 = mx.symbol.Convolution(name='conv_4b_double_3x3_0',kernel=(3, 3),num_filter=128,pad=(1, 1),stride=(1, 1),data=relu_4b_double_3x3_reduce)
bn_4b_double_3x3_0 = mx.symbol.BatchNorm(name='bn_4b_double_3x3_0',data=conv_4b_double_3x3_0)
relu_4b_double_3x3_0 = mx.symbol.Activation(act_type='relu',name='relu_4b_double_3x3_0',data=bn_4b_double_3x3_0)
conv_4b_double_3x3_1 = mx.symbol.Convolution(name='conv_4b_double_3x3_1',kernel=(3, 3),num_filter=128,pad=(1, 1),stride=(1, 1),data=relu_4b_double_3x3_0)
bn_4b_double_3x3_1 = mx.symbol.BatchNorm(name='bn_4b_double_3x3_1',data=conv_4b_double_3x3_1)
relu_4b_double_3x3_1 = mx.symbol.Activation(act_type='relu',name='relu_4b_double_3x3_1',data=bn_4b_double_3x3_1)
avg_pool_4b_pool = mx.symbol.Pooling(pool_type='avg',name='avg_pool_4b_pool',kernel=(3, 3),stride=(1, 1),pad=(1, 1),data=ch_concat_4a_chconcat)
conv_4b_proj = mx.symbol.Convolution(name='conv_4b_proj',kernel=(1, 1),num_filter=128,pad=(0, 0),stride=(1, 1),data=avg_pool_4b_pool)
bn_4b_proj = mx.symbol.BatchNorm(name='bn_4b_proj',data=conv_4b_proj)
relu_4b_proj = mx.symbol.Activation(act_type='relu',name='relu_4b_proj',data=bn_4b_proj)
ch_concat_4b_chconcat = mx.symbol.Concat(name='ch_concat_4b_chconcat',*[relu_4b_1x1,relu_4b_3x3,relu_4b_double_3x3_1,relu_4b_proj])
conv_4c_1x1 = mx.symbol.Convolution(name='conv_4c_1x1',kernel=(1, 1),num_filter=160,pad=(0, 0),stride=(1, 1),data=ch_concat_4b_chconcat)
bn_4c_1x1 = mx.symbol.BatchNorm(name='bn_4c_1x1',data=conv_4c_1x1)
relu_4c_1x1 = mx.symbol.Activation(act_type='relu',name='relu_4c_1x1',data=bn_4c_1x1)
conv_4c_3x3_reduce = mx.symbol.Convolution(name='conv_4c_3x3_reduce',kernel=(1, 1),num_filter=128,pad=(0, 0),stride=(1, 1),data=ch_concat_4b_chconcat)
bn_4c_3x3_reduce = mx.symbol.BatchNorm(name='bn_4c_3x3_reduce',data=conv_4c_3x3_reduce)
relu_4c_3x3_reduce = mx.symbol.Activation(act_type='relu',name='relu_4c_3x3_reduce',data=bn_4c_3x3_reduce)
conv_4c_3x3 = mx.symbol.Convolution(name='conv_4c_3x3',kernel=(3, 3),num_filter=160,pad=(1, 1),stride=(1, 1),data=relu_4c_3x3_reduce)
bn_4c_3x3 = mx.symbol.BatchNorm(name='bn_4c_3x3',data=conv_4c_3x3)
relu_4c_3x3 = mx.symbol.Activation(act_type='relu',name='relu_4c_3x3',data=bn_4c_3x3)
conv_4c_double_3x3_reduce = mx.symbol.Convolution(name='conv_4c_double_3x3_reduce',kernel=(1, 1),num_filter=128,pad=(0, 0),stride=(1, 1),data=ch_concat_4b_chconcat)
bn_4c_double_3x3_reduce = mx.symbol.BatchNorm(name='bn_4c_double_3x3_reduce',data=conv_4c_double_3x3_reduce)
relu_4c_double_3x3_reduce = mx.symbol.Activation(act_type='relu',name='relu_4c_double_3x3_reduce',data=bn_4c_double_3x3_reduce)
conv_4c_double_3x3_0 = mx.symbol.Convolution(name='conv_4c_double_3x3_0',kernel=(3, 3),num_filter=160,pad=(1, 1),stride=(1, 1),data=relu_4c_double_3x3_reduce)
bn_4c_double_3x3_0 = mx.symbol.BatchNorm(name='bn_4c_double_3x3_0',data=conv_4c_double_3x3_0)
relu_4c_double_3x3_0 = mx.symbol.Activation(act_type='relu',name='relu_4c_double_3x3_0',data=bn_4c_double_3x3_0)
conv_4c_double_3x3_1 = mx.symbol.Convolution(name='conv_4c_double_3x3_1',kernel=(3, 3),num_filter=160,pad=(1, 1),stride=(1, 1),data=relu_4c_double_3x3_0)
bn_4c_double_3x3_1 = mx.symbol.BatchNorm(name='bn_4c_double_3x3_1',data=conv_4c_double_3x3_1)
relu_4c_double_3x3_1 = mx.symbol.Activation(act_type='relu',name='relu_4c_double_3x3_1',data=bn_4c_double_3x3_1)
avg_pool_4c_pool = mx.symbol.Pooling(pool_type='avg',name='avg_pool_4c_pool',kernel=(3, 3),stride=(1, 1),pad=(1, 1),data=ch_concat_4b_chconcat)
conv_4c_proj = mx.symbol.Convolution(name='conv_4c_proj',kernel=(1, 1),num_filter=128,pad=(0, 0),stride=(1, 1),data=avg_pool_4c_pool)
bn_4c_proj = mx.symbol.BatchNorm(name='bn_4c_proj',data=conv_4c_proj)
relu_4c_proj = mx.symbol.Activation(act_type='relu',name='relu_4c_proj',data=bn_4c_proj)
ch_concat_4c_chconcat = mx.symbol.Concat(name='ch_concat_4c_chconcat',*[relu_4c_1x1,relu_4c_3x3,relu_4c_double_3x3_1,relu_4c_proj])
conv_4d_1x1 = mx.symbol.Convolution(name='conv_4d_1x1',kernel=(1, 1),num_filter=96,pad=(0, 0),stride=(1, 1),data=ch_concat_4c_chconcat)
bn_4d_1x1 = mx.symbol.BatchNorm(name='bn_4d_1x1',data=conv_4d_1x1)
relu_4d_1x1 = mx.symbol.Activation(act_type='relu',name='relu_4d_1x1',data=bn_4d_1x1)
conv_4d_3x3_reduce = mx.symbol.Convolution(name='conv_4d_3x3_reduce',kernel=(1, 1),num_filter=128,pad=(0, 0),stride=(1, 1),data=ch_concat_4c_chconcat)
bn_4d_3x3_reduce = mx.symbol.BatchNorm(name='bn_4d_3x3_reduce',data=conv_4d_3x3_reduce)
relu_4d_3x3_reduce = mx.symbol.Activation(act_type='relu',name='relu_4d_3x3_reduce',data=bn_4d_3x3_reduce)
conv_4d_3x3 = mx.symbol.Convolution(name='conv_4d_3x3',kernel=(3, 3),num_filter=192,pad=(1, 1),stride=(1, 1),data=relu_4d_3x3_reduce)
bn_4d_3x3 = mx.symbol.BatchNorm(name='bn_4d_3x3',data=conv_4d_3x3)
relu_4d_3x3 = mx.symbol.Activation(act_type='relu',name='relu_4d_3x3',data=bn_4d_3x3)
conv_4d_double_3x3_reduce = mx.symbol.Convolution(name='conv_4d_double_3x3_reduce',kernel=(1, 1),num_filter=160,pad=(0, 0),stride=(1, 1),data=ch_concat_4c_chconcat)
bn_4d_double_3x3_reduce = mx.symbol.BatchNorm(name='bn_4d_double_3x3_reduce',data=conv_4d_double_3x3_reduce)
relu_4d_double_3x3_reduce = mx.symbol.Activation(act_type='relu',name='relu_4d_double_3x3_reduce',data=bn_4d_double_3x3_reduce)
conv_4d_double_3x3_0 = mx.symbol.Convolution(name='conv_4d_double_3x3_0',kernel=(3, 3),num_filter=192,pad=(1, 1),stride=(1, 1),data=relu_4d_double_3x3_reduce)
bn_4d_double_3x3_0 = mx.symbol.BatchNorm(name='bn_4d_double_3x3_0',data=conv_4d_double_3x3_0)
relu_4d_double_3x3_0 = mx.symbol.Activation(act_type='relu',name='relu_4d_double_3x3_0',data=bn_4d_double_3x3_0)
conv_4d_double_3x3_1 = mx.symbol.Convolution(name='conv_4d_double_3x3_1',kernel=(3, 3),num_filter=192,pad=(1, 1),stride=(1, 1),data=relu_4d_double_3x3_0)
bn_4d_double_3x3_1 = mx.symbol.BatchNorm(name='bn_4d_double_3x3_1',data=conv_4d_double_3x3_1)
relu_4d_double_3x3_1 = mx.symbol.Activation(act_type='relu',name='relu_4d_double_3x3_1',data=bn_4d_double_3x3_1)
avg_pool_4d_pool = mx.symbol.Pooling(pool_type='avg',name='avg_pool_4d_pool',kernel=(3, 3),stride=(1, 1),pad=(1, 1),data=ch_concat_4c_chconcat)
conv_4d_proj = mx.symbol.Convolution(name='conv_4d_proj',kernel=(1, 1),num_filter=128,pad=(0, 0),stride=(1, 1),data=avg_pool_4d_pool)
bn_4d_proj = mx.symbol.BatchNorm(name='bn_4d_proj',data=conv_4d_proj)
relu_4d_proj = mx.symbol.Activation(act_type='relu',name='relu_4d_proj',data=bn_4d_proj)
ch_concat_4d_chconcat = mx.symbol.Concat(name='ch_concat_4d_chconcat',*[relu_4d_1x1,relu_4d_3x3,relu_4d_double_3x3_1,relu_4d_proj])
conv_4e_3x3_reduce = mx.symbol.Convolution(name='conv_4e_3x3_reduce',kernel=(1, 1),num_filter=128,pad=(0, 0),stride=(1, 1),data=ch_concat_4d_chconcat)
bn_4e_3x3_reduce = mx.symbol.BatchNorm(name='bn_4e_3x3_reduce',data=conv_4e_3x3_reduce)
relu_4e_3x3_reduce = mx.symbol.Activation(act_type='relu',name='relu_4e_3x3_reduce',data=bn_4e_3x3_reduce)
conv_4e_3x3 = mx.symbol.Convolution(name='conv_4e_3x3',kernel=(3, 3),num_filter=192,pad=(1, 1),stride=(2, 2),data=relu_4e_3x3_reduce)
bn_4e_3x3 = mx.symbol.BatchNorm(name='bn_4e_3x3',data=conv_4e_3x3)
relu_4e_3x3 = mx.symbol.Activation(act_type='relu',name='relu_4e_3x3',data=bn_4e_3x3)
conv_4e_double_3x3_reduce = mx.symbol.Convolution(name='conv_4e_double_3x3_reduce',kernel=(1, 1),num_filter=192,pad=(0, 0),stride=(1, 1),data=ch_concat_4d_chconcat)
bn_4e_double_3x3_reduce = mx.symbol.BatchNorm(name='bn_4e_double_3x3_reduce',data=conv_4e_double_3x3_reduce)
relu_4e_double_3x3_reduce = mx.symbol.Activation(act_type='relu',name='relu_4e_double_3x3_reduce',data=bn_4e_double_3x3_reduce)
conv_4e_double_3x3_0 = mx.symbol.Convolution(name='conv_4e_double_3x3_0',kernel=(3, 3),num_filter=256,pad=(1, 1),stride=(1, 1),data=relu_4e_double_3x3_reduce)
bn_4e_double_3x3_0 = mx.symbol.BatchNorm(name='bn_4e_double_3x3_0',data=conv_4e_double_3x3_0)
relu_4e_double_3x3_0 = mx.symbol.Activation(act_type='relu',name='relu_4e_double_3x3_0',data=bn_4e_double_3x3_0)
conv_4e_double_3x3_1 = mx.symbol.Convolution(name='conv_4e_double_3x3_1',kernel=(3, 3),num_filter=256,pad=(1, 1),stride=(2, 2),data=relu_4e_double_3x3_0)
bn_4e_double_3x3_1 = mx.symbol.BatchNorm(name='bn_4e_double_3x3_1',data=conv_4e_double_3x3_1)
relu_4e_double_3x3_1 = mx.symbol.Activation(act_type='relu',name='relu_4e_double_3x3_1',data=bn_4e_double_3x3_1)
max_pool_4e_pool = mx.symbol.Pooling(pool_type='max',name='max_pool_4e_pool',kernel=(3, 3),stride=(2, 2),data=ch_concat_4d_chconcat)
ch_concat_4e_chconcat = mx.symbol.Concat(name='ch_concat_4e_chconcat',*[relu_4e_3x3,relu_4e_double_3x3_1,max_pool_4e_pool])
conv_5a_1x1 = mx.symbol.Convolution(name='conv_5a_1x1',kernel=(1, 1),num_filter=352,pad=(0, 0),stride=(1, 1),data=ch_concat_4e_chconcat)
bn_5a_1x1 = mx.symbol.BatchNorm(name='bn_5a_1x1',data=conv_5a_1x1)
relu_5a_1x1 = mx.symbol.Activation(act_type='relu',name='relu_5a_1x1',data=bn_5a_1x1)
conv_5a_3x3_reduce = mx.symbol.Convolution(name='conv_5a_3x3_reduce',kernel=(1, 1),num_filter=192,pad=(0, 0),stride=(1, 1),data=ch_concat_4e_chconcat)
bn_5a_3x3_reduce = mx.symbol.BatchNorm(name='bn_5a_3x3_reduce',data=conv_5a_3x3_reduce)
relu_5a_3x3_reduce = mx.symbol.Activation(act_type='relu',name='relu_5a_3x3_reduce',data=bn_5a_3x3_reduce)
conv_5a_3x3 = mx.symbol.Convolution(name='conv_5a_3x3',kernel=(3, 3),num_filter=320,pad=(1, 1),stride=(1, 1),data=relu_5a_3x3_reduce)
bn_5a_3x3 = mx.symbol.BatchNorm(name='bn_5a_3x3',data=conv_5a_3x3)
relu_5a_3x3 = mx.symbol.Activation(act_type='relu',name='relu_5a_3x3',data=bn_5a_3x3)
conv_5a_double_3x3_reduce = mx.symbol.Convolution(name='conv_5a_double_3x3_reduce',kernel=(1, 1),num_filter=160,pad=(0, 0),stride=(1, 1),data=ch_concat_4e_chconcat)
bn_5a_double_3x3_reduce = mx.symbol.BatchNorm(name='bn_5a_double_3x3_reduce',data=conv_5a_double_3x3_reduce)
relu_5a_double_3x3_reduce = mx.symbol.Activation(act_type='relu',name='relu_5a_double_3x3_reduce',data=bn_5a_double_3x3_reduce)
conv_5a_double_3x3_0 = mx.symbol.Convolution(name='conv_5a_double_3x3_0',kernel=(3, 3),num_filter=224,pad=(1, 1),stride=(1, 1),data=relu_5a_double_3x3_reduce)
bn_5a_double_3x3_0 = mx.symbol.BatchNorm(name='bn_5a_double_3x3_0',data=conv_5a_double_3x3_0)
relu_5a_double_3x3_0 = mx.symbol.Activation(act_type='relu',name='relu_5a_double_3x3_0',data=bn_5a_double_3x3_0)
conv_5a_double_3x3_1 = mx.symbol.Convolution(name='conv_5a_double_3x3_1',kernel=(3, 3),num_filter=224,pad=(1, 1),stride=(1, 1),data=relu_5a_double_3x3_0)
bn_5a_double_3x3_1 = mx.symbol.BatchNorm(name='bn_5a_double_3x3_1',data=conv_5a_double_3x3_1)
relu_5a_double_3x3_1 = mx.symbol.Activation(act_type='relu',name='relu_5a_double_3x3_1',data=bn_5a_double_3x3_1)
avg_pool_5a_pool = mx.symbol.Pooling(pool_type='avg',name='avg_pool_5a_pool',kernel=(3, 3),stride=(1, 1),pad=(1, 1),data=ch_concat_4e_chconcat)
conv_5a_proj = mx.symbol.Convolution(name='conv_5a_proj',kernel=(1, 1),num_filter=128,pad=(0, 0),stride=(1, 1),data=avg_pool_5a_pool)
bn_5a_proj = mx.symbol.BatchNorm(name='bn_5a_proj',data=conv_5a_proj)
relu_5a_proj = mx.symbol.Activation(act_type='relu',name='relu_5a_proj',data=bn_5a_proj)
ch_concat_5a_chconcat = mx.symbol.Concat(name='ch_concat_5a_chconcat',*[relu_5a_1x1,relu_5a_3x3,relu_5a_double_3x3_1,relu_5a_proj])
conv_5b_1x1 = mx.symbol.Convolution(name='conv_5b_1x1',kernel=(1, 1),num_filter=352,pad=(0, 0),stride=(1, 1),data=ch_concat_5a_chconcat)
bn_5b_1x1 = mx.symbol.BatchNorm(name='bn_5b_1x1',data=conv_5b_1x1)
relu_5b_1x1 = mx.symbol.Activation(act_type='relu',name='relu_5b_1x1',data=bn_5b_1x1)
conv_5b_3x3_reduce = mx.symbol.Convolution(name='conv_5b_3x3_reduce',kernel=(1, 1),num_filter=192,pad=(0, 0),stride=(1, 1),data=ch_concat_5a_chconcat)
bn_5b_3x3_reduce = mx.symbol.BatchNorm(name='bn_5b_3x3_reduce',data=conv_5b_3x3_reduce)
relu_5b_3x3_reduce = mx.symbol.Activation(act_type='relu',name='relu_5b_3x3_reduce',data=bn_5b_3x3_reduce)
conv_5b_3x3 = mx.symbol.Convolution(name='conv_5b_3x3',kernel=(3, 3),num_filter=320,pad=(1, 1),stride=(1, 1),data=relu_5b_3x3_reduce)
bn_5b_3x3 = mx.symbol.BatchNorm(name='bn_5b_3x3',data=conv_5b_3x3)
relu_5b_3x3 = mx.symbol.Activation(act_type='relu',name='relu_5b_3x3',data=bn_5b_3x3)
conv_5b_double_3x3_reduce = mx.symbol.Convolution(name='conv_5b_double_3x3_reduce',kernel=(1, 1),num_filter=192,pad=(0, 0),stride=(1, 1),data=ch_concat_5a_chconcat)
bn_5b_double_3x3_reduce = mx.symbol.BatchNorm(name='bn_5b_double_3x3_reduce',data=conv_5b_double_3x3_reduce)
relu_5b_double_3x3_reduce = mx.symbol.Activation(act_type='relu',name='relu_5b_double_3x3_reduce',data=bn_5b_double_3x3_reduce)
conv_5b_double_3x3_0 = mx.symbol.Convolution(name='conv_5b_double_3x3_0',kernel=(3, 3),num_filter=224,pad=(1, 1),stride=(1, 1),data=relu_5b_double_3x3_reduce)
bn_5b_double_3x3_0 = mx.symbol.BatchNorm(name='bn_5b_double_3x3_0',data=conv_5b_double_3x3_0)
relu_5b_double_3x3_0 = mx.symbol.Activation(act_type='relu',name='relu_5b_double_3x3_0',data=bn_5b_double_3x3_0)
conv_5b_double_3x3_1 = mx.symbol.Convolution(name='conv_5b_double_3x3_1',kernel=(3, 3),num_filter=224,pad=(1, 1),stride=(1, 1),data=relu_5b_double_3x3_0)
bn_5b_double_3x3_1 = mx.symbol.BatchNorm(name='bn_5b_double_3x3_1',data=conv_5b_double_3x3_1)
relu_5b_double_3x3_1 = mx.symbol.Activation(act_type='relu',name='relu_5b_double_3x3_1',data=bn_5b_double_3x3_1)
max_pool_5b_pool = mx.symbol.Pooling(pool_type='max',name='max_pool_5b_pool',kernel=(3, 3),stride=(1, 1),pad=(1, 1),data=ch_concat_5a_chconcat)
conv_5b_proj = mx.symbol.Convolution(name='conv_5b_proj',kernel=(1, 1),num_filter=128,pad=(0, 0),stride=(1, 1),data=max_pool_5b_pool)
bn_5b_proj = mx.symbol.BatchNorm(name='bn_5b_proj',data=conv_5b_proj)
relu_5b_proj = mx.symbol.Activation(act_type='relu',name='relu_5b_proj',data=bn_5b_proj)
ch_concat_5b_chconcat = mx.symbol.Concat(name='ch_concat_5b_chconcat',*[relu_5b_1x1,relu_5b_3x3,relu_5b_double_3x3_1,relu_5b_proj])
global_pool = mx.symbol.Pooling(pool_type='avg',name='global_pool',kernel=(7, 7),stride=(1, 1),data=ch_concat_5b_chconcat)
flatten = mx.symbol.Flatten(name='flatten',data=global_pool)
fc = mx.symbol.FullyConnected(name='fc',num_hidden=100,data=flatten)
softmax = mx.symbol.Softmax(name='softmax',data=fc)



####################
# load weight
import os
import numpy as np
ROOT = "./0001/"
candidates = os.listdir(ROOT)

arg_names = softmax.list_arguments()
aux_names = softmax.list_auxiliary_states()
arg_shapes, output_shapes, aux_shapes = softmax.infer_shape(data=(128,3,28,28))

arg_shape_dic = dict(zip(arg_names, arg_shapes))
aux_shape_dic = dict(zip(aux_names, aux_shapes))

args = {}
for item in candidates:
    key = item[:-4]
    tmp = np.load(ROOT + item)
    if key in arg_shape_dic:
        target_shape = arg_shape_dic[key]
    if key in aux_shape_dic:
        target_shape = aux_shape_dic[key]
    tmp = tmp.reshape(target_shape)
    args[key] = mx.nd.zeros(tmp.shape)
    args[key][:] = tmp

arg_params = {}
aux_params = {}

for name in arg_names:
    if name in args:
        arg_params[name] = args[name]
for name in aux_names:
    if name in args:
        aux_params[name] = args[name]

model = mx.model.FeedForward(ctx=mx.cpu(), symbol=softmax,
        arg_params=arg_params, aux_params=aux_params, num_round=1,
        learning_rate=0.05, momentum=0.9, wd=0.0001)

model.save("cifar_100_cxx")

