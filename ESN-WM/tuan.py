from model import generate_model

n_gate = 1
model = generate_model(shape=(1+n_gate,1000,n_gate),
                       sparsity=0.5, radius=0.1, scaling=(1.0,1.0),
                       leak=1.0, noise=(0.0000, 0.0001, 0.0001))

model