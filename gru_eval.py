from cntk.layers import Recurrence
import cntk as C
from cntk.learners import sgd
from cntk.logging import ProgressPrinter
from cntk.layers import Dense, Sequential, GRU
from cntk.device import try_set_default_device, gpu, cpu

import time
import numpy as np

def create_model(X):
    # defines few stacked GRUs

    l1 = Recurrence(step_function=GRU(shape=20))(X)
    l2 = Recurrence(step_function=GRU(shape=20))(l1)
    l3 = Dense(shape=1)(l2)

    return l3

EPOCHS = 10

#try_set_default_device(gpu(0))
try_set_default_device(cpu())

batch_size, seq_len, input_dim = (None, None, 5)

output_shape = (batch_size, seq_len, 1)

input_shape = [batch_size, seq_len, input_dim]

#Y = C.input_variable(shape=output_shape)

X = C.sequence.input_variable(input_dim)
Y = C.sequence.input_variable(1)

Y_model = create_model(X)

# loss function
error = loss = C.squared_error(Y_model, Y)

# define optimizer
learner = C.adam(Y_model.parameters, lr=0.02, momentum=0.99)

trainer = C.Trainer(Y_model, (loss, error), [learner])


# create some mocked data (4 sequences of different lenghts
arr_dtype = np.float32
s1_x = np.random.randn(10000, input_dim).astype(dtype=arr_dtype)
s2_x = np.random.randn(400, input_dim).astype(dtype=arr_dtype)
s3_x = np.random.randn(300, input_dim).astype(dtype=arr_dtype)
s4_x = np.random.randn(600, input_dim).astype(dtype=arr_dtype)

s1_y = np.random.randn(10000, 1).astype(dtype=arr_dtype)
s2_y = np.random.randn(400, 1).astype(dtype=arr_dtype)
s3_y = np.random.randn(300, 1).astype(dtype=arr_dtype)
s4_y = np.random.randn(600, 1).astype(dtype=arr_dtype)

data_x = [s1_x, s2_x, s3_x, s4_x]
data_y = [s1_y, s2_y, s3_y, s4_y]

loss_summary = []
start = time.time()
for epoch in range(0, EPOCHS):

    trainer.train_minibatch({X: data_x, Y: data_y})

    training_loss = trainer.previous_minibatch_loss_average
    loss_summary.append(training_loss)
    #print("epoch: {}, loss: {:.5f}".format(epoch, training_loss))

print("Optimized training took {0:.1f} sec".format(time.time() - start))


for epoch in range(0, EPOCHS):

    trainer.train_minibatch({X: np.random.randn(4, 10000, input_dim).astype(dtype=arr_dtype), Y: np.random.randn(4, 10000, 1).astype(dtype=arr_dtype)})

    training_loss = trainer.previous_minibatch_loss_average
    loss_summary.append(training_loss)
    #print("epoch: {}, loss: {:.5f}".format(epoch, training_loss))

print("Normal training took {0:.1f} sec".format(time.time() - start))