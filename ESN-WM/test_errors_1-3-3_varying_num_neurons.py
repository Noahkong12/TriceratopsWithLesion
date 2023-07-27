import numpy as np
import matplotlib.pyplot as plt
from data import generate_data, smoothen, str_to_bmp, convert_data 
from model_modified import generate_model, train_model, test_model
from identify_neurons import identify_neurons
from lesion import lesion
from identify_neurons_according_test_err import identify_neurons_according_test_err
import sys
import os


# 1-3-3 scalar task
# Random generator initialization
task = "1-3-3-scalar"
n_gate = 3
print(task)

np.random.seed(1)

# Build memory
model = generate_model(shape=(1+n_gate,1000,n_gate), sparsity=0.5,
                        radius=0.1, scaling=(1.0, 0.33), leak=1.0,
                        noise=(0.000, 0.0001, 0.000))

# Training data
n = 25000
values = np.random.uniform(-1, +1, n)
ticks = np.random.uniform(0, 1, (n, n_gate)) < 0.01
train_data = generate_data(values, ticks)

# Testing data
n = 2500
values = smoothen(np.random.uniform(-1, +1, n))
ticks = np.random.uniform(0, 1, (n, n_gate)) < 0.01
test_data = generate_data(values, ticks, last = train_data["output"][-1])

error = train_model(model, train_data)
print("Training error : {0}".format(error))

error_wo_lesion = test_model(model, test_data, 42)
print("Testing error without lesion : {0}".format(error_wo_lesion))


error0s = []
error1s = []
error2s = []
error_alls = []
for n in range(1,1000):
    _, lesion_neurons = identify_neurons_according_test_err('error_lesions.npy', n, 'least significant', 'all')
    ## lesion correspoing weights of selected neurons
    lesioned_model = lesion(model, lesion_neurons)
    error_w_lesion = test_model(lesioned_model, test_data, 42)
    error0s.append(error_w_lesion["error0"])
    error1s.append(error_w_lesion["error1"])
    error2s.append(error_w_lesion["error2"])
    error_alls.append(error_w_lesion["error_whole"])

np.save("error0s.npy", error0s)
np.save("error1s.npy", error1s)
np.save("error2s.npy", error2s)
np.save("error_alls.npy", error_alls)

plt.plot(error_alls)
plt.show()