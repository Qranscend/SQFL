import pennylane as qml
import torch
import numpy as np

n_qubits = 2
dev = qml.device("default.mixed", wires=n_qubits)

def quantum_feature_map(x):
    qml.Hadamard(wires=0)
    qml.Hadamard(wires=1)
    qml.RY(x[0], wires=0)
    qml.RY(x[1], wires=1)
    qml.CNOT(wires=[0, 1])

@qml.qnode(dev, interface="torch", diff_method="backprop")
def quantum_kernel(params, x1, x2):
    quantum_feature_map(x1)
    qml.adjoint(quantum_feature_map)(x2)
    return qml.probs(wires=[0, 1])

def quantum_svm_classifier(params, x):
    support_vectors = [torch.tensor([np.pi / 2, np.pi / 2], dtype=torch.float32), torch.tensor([0, 0], dtype=torch.float32)]
    labels = torch.tensor([1, -1], dtype=torch.float32)

    kernel_sum = 0
    for i, sv in enumerate(support_vectors):
        kernel_val = quantum_kernel(params, x, sv)[0]
        kernel_sum += labels[i] * kernel_val

    return 1 if kernel_sum >= 0 else -1

def superposed_focal_loss(params, y_true1, y_true2, alpha=0.25, gamma=2.0):
    probs = quantum_kernel(params, torch.tensor([np.pi / 2, np.pi / 2], dtype=torch.float32), torch.tensor([0, 0], dtype=torch.float32))
    y_pred1 = probs[1]
    y_pred2 = probs[2]

    y_true1 = torch.tensor(y_true1, dtype=torch.float32, requires_grad=False)
    y_true2 = torch.tensor(y_true2, dtype=torch.float32, requires_grad=False)

    loss1 = -alpha * (1 - y_pred1) ** gamma * y_true1 * torch.log(y_pred1 + 1e-8)
    loss2 = -alpha * (1 - y_pred2) ** gamma * y_true2 * torch.log(y_pred2 + 1e-8)

    return (loss1 + loss2).requires_grad_()

params = torch.tensor(np.random.rand(2), dtype=torch.float32, requires_grad=True)
optimizer = torch.optim.Adam([params], lr=0.1)

for epoch in range(50):
    optimizer.zero_grad()
    loss = superposed_focal_loss(params, y_true1=1, y_true2=1)
    loss.backward()
    optimizer.step()

    if epoch % 10 == 0:
        pred_class = quantum_svm_classifier(params, torch.tensor([np.pi / 4, np.pi / 4], dtype=torch.float32))
        print(f"Epoch {epoch}: Loss = {loss.item()}, Predicted Class = {pred_class}")
