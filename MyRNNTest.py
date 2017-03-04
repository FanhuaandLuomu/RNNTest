#coding:utf-8
# numpy实现RNN 
# 预测前向传播(考虑前一个时间步历史信息)
# 误差反向传播(考虑后一个时间步误差)  
import copy,numpy as np
np.random.seed(0)

# compute sigmoid 
def sigmoid(x):
	output=1.0/(1.0+np.exp(-x))
	return output

def sigmoid_output_to_derivative(output):
	return output*(1 - output)

#training data generation
int2binary={}
binary_dim=8

largest_number=pow(2,binary_dim)

# 转化为8位二进制
binary=np.unpackbits(np.array([range(largest_number)],dtype=np.uint8).T,axis=1)
for i in range(largest_number):
	int2binary[i]=binary[i]

# input variables
alpha=0.1
input_dim=2
hidden_dim=16
output_dim=1

# initialize neural networks weights   (0~2)-1  ->  -1~1
synapse_0=2*np.random.random((input_dim,hidden_dim))-1
synapse_1=2*np.random.random((hidden_dim,output_dim))-1
synapse_h=2*np.random.random((hidden_dim,hidden_dim))-1

synapse_0_update=np.zeros_like(synapse_0)
synapse_1_update=np.zeros_like(synapse_1)
synapse_h_update=np.zeros_like(synapse_h)

# training logic
for j in range(100000):
	# generate a simple addition problem (a+b=c)
	a_int=np.random.randint(largest_number/2)   # int version  x1
	a=int2binary[a_int]

	b_int=np.random.randint(largest_number/2)  # int version x2
	b=int2binary[b_int]

	# true answer
	c_int=a_int+b_int
	c=int2binary[c_int]

	# where we'll store our best guess (binary encoded)
	d=np.zeros_like(c)

	overallError=0

	layer_2_deltas=list()
	layer_1_values=list()
	layer_1_values.append(np.zeros(hidden_dim))

	# moving along the position in the binary encoding
	for position in range(binary_dim):
		# generate input and output
		# 两个8位二进制 对应位组成一个训练样本  1*2
		X=np.array([[a[binary_dim-position-1],b[binary_dim-position-1]]])
		y=np.array([[c[binary_dim-position-1]]]).T

		# hidden layer (input~+pre_hidden)
		layer_1=sigmoid(np.dot(X,synapse_0)+np.dot(layer_1_values[-1],synapse_h))

		# output layer (new binary representation)
		layer_2=sigmoid(np.dot(layer_1,synapse_1))

		layer_2_error=y-layer_2
		layer_2_deltas.append((layer_2_error)*sigmoid_output_to_derivative(layer_2))
		overallError+=np.abs(layer_2_error[0])

		# decode estimate so we can print it output_dim  四舍五入
		d[binary_dim-position-1]=np.round(layer_2[0][0])

		# store hidden layer so we can use it in the next timestep
		# 存储 h_t-1  历史信息 供下一个时间步使用
		layer_1_values.append(copy.deepcopy(layer_1))

	# 反向传播时，暂存后一个t+1时间步的反向传播误差
	future_layer_1_delta=np.zeros(hidden_dim)

	for position in range(binary_dim):
		X=np.array([[a[position],b[position]]])
		layer_1=layer_1_values[-position-1]  # 后一个
		prev_layer_1=layer_1_values[-position-2]  # 前一个

		# error at output layer
		layer_2_delta=layer_2_deltas[-position-1]
		# error at hidden layer  隐藏层误差要算上后一个时间步的误差
		layer_1_delta=(future_layer_1_delta.dot(synapse_h.T)+layer_2_delta.dot(synapse_1.T))*\
				sigmoid_output_to_derivative(layer_1)
		# let's update all our weights so we can try again
		synapse_1_update+=np.atleast_2d(layer_1).T.dot(layer_2_delta)
		synapse_h_update+=np.atleast_2d(prev_layer_1).T.dot(layer_1_delta)
		synapse_0_update+=X.T.dot(layer_1_delta)

		future_layer_1_delta=layer_1_delta  # 后一个时间步的误差

	# 更新权重  alpha 步长
	synapse_0+=synapse_0_update*alpha   
	synapse_1+=synapse_1_update*alpha
	synapse_h+=synapse_h_update*alpha

	# 权重更新单元置0
	synapse_0_update*=0
	synapse_1_update*=0
	synapse_h_update*=0

	# print out progress
	if j%1000==0:
		print 'Error:'+str(overallError)
		print 'Pred:'+str(d)
		print 'True:'+str(c)
		out=0
		for index,x in enumerate(reversed(d)):
			out+=x*pow(2,index)
		print str(a_int)+"+"+str(b_int)+"="+str(out)
		print '---------------'




