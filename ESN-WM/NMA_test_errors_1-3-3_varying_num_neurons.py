
'''
1. This scrip uses the assessment on impact of individual neurons, i.e. test errors when lesion single neuron.
2. data imported from Tuan's file, which integrated lesions by zero out W_in, W_rc, W_fb
3. Here, we compare the test errors on lesioned model with lesioned neurons varing from 0 - 1000. 
4. The neurons can be selected based on "most significant", "least significant", or "random". 
5. The selection can be based on either "all" (all 3 outputs), "output0", "output1", or "output2"
6. save the test errors on "all" (all 3 outputs), "output0", "output1", or "output2"
7. make plots of test errors ***

Note: Figure_1.png shows the test error on all 3 outputs, with lesion_neurons selection scheme ('least significant', 'all'), 
      we can expect to see that th test error increases wrt number of leision_neurons, 
      however, this aint the case. I am guessing its the noise in our model can contribute to this? 
'''


import numpy as np
import matplotlib.pyplot as plt
from data import generate_data, smoothen, str_to_bmp, convert_data 
from NMA_model_modified import generate_model, train_model, test_model
from NMA_identify_neurons import identify_neurons
from NMA_lesion import lesion
from NMA_identify_neurons_according_test_err import identify_neurons_according_test_err
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
np.save('test_error_wo_lesion', error_wo_lesion)


error0s = []
error1s = []
error2s = []
error_alls = []

selection_method = 'random'
selection_ouput = 'all'

for n in np.linspace(0, 1000, 101, dtype=int):  # We only taking 101 points here, including 0 and 1000
    _, lesion_neurons = identify_neurons_according_test_err('error_lesions.npy', n, selection_method, selection_ouput)  # _ is a dict, details see identify_neurons_according_test_err.py
    ## lesion correspoing weights of selected neurons
    lesioned_model = lesion(model, lesion_neurons)
    error_w_lesion = test_model(lesioned_model, test_data, 42)
    error0s.append(error_w_lesion["error0"])
    error1s.append(error_w_lesion["error1"])
    error2s.append(error_w_lesion["error2"])
    error_alls.append(error_w_lesion["error_whole"])
 
np.save(f"error0s_{selection_method}_{selection_ouput}.npy", error0s)
np.save(f"error1s_{selection_method}_{selection_ouput}.npy", error1s)
np.save(f"error2s_{selection_method}_{selection_ouput}.npy", error2s)
np.save(f"error_alls_{selection_method}_{selection_ouput}.npy", error_alls)

plt.plot(error_alls)
plt.show()