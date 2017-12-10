import keras
from keras.layers import Dense, GRU, TimeDistributed
from keras.models import Model, Input


import time
import numpy as np

def create_model(X):
    # defines few stacked GRUs

    l1 = GRU(20, return_sequences=True)(X)
    l2 = GRU(20, return_sequences=True)(l1)
    l3 = TimeDistributed(Dense(1))(l2)

    return l3

EPOCHS = 10


#try_set_default_device(gpu(0))
#try_set_default_device(cpu())

batch_size, seq_len, input_dim = (None, None, 5)

output_shape = (batch_size, seq_len, 1)

input_shape = [batch_size, seq_len, input_dim]

#Y = C.input_variable(shape=output_shape)

X = Input((seq_len, input_dim))

Y_model = create_model(X)

model = Model(inputs=X, outputs=Y_model)

model.compile(optimizer='adam', loss='mse')


# create some mocked data (4 sequences of different lenghts
arr_dtype = np.float32
s1_x = np.random.randn(1000, input_dim).astype(dtype=arr_dtype)
s2_x = np.random.randn(400, input_dim).astype(dtype=arr_dtype)
s3_x = np.random.randn(300, input_dim).astype(dtype=arr_dtype)
s4_x = np.random.randn(600, input_dim).astype(dtype=arr_dtype)

s1_y = np.random.randn(1000, 1).astype(dtype=arr_dtype)
s2_y = np.random.randn(400, 1).astype(dtype=arr_dtype)
s3_y = np.random.randn(300, 1).astype(dtype=arr_dtype)
s4_y = np.random.randn(600, 1).astype(dtype=arr_dtype)

data_x = [s1_x, s2_x, s3_x, s4_x]
data_y = [s1_y, s2_y, s3_y, s4_y]

loss_summary = []
start = time.time()
for epoch in range(0, EPOCHS):

    model.train_on_batch(x=data_x, y=data_y)

print("Optimized training took {0:.1f} sec".format(time.time() - start))