import numpy as np
import matplotlib.pyplot as plt


W_out = np.load("W_out.npy")
plt.plot(np.squeeze(W_out))
#plt.show()

internals = np.load("internal.npy")
for n in range(3):
    plt.plot(internals[:,n])

#plt.show()

# 1 x 1000

sort_W = np.argsort(np.absolute(W_out))
print(sort_W[:,-10:])