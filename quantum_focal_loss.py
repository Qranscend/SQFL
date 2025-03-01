import pennylane as qml
import torch
import numpy as np

n_qubits = 2
dev = qml.device("default.mixed", wires=n_qubits)

@qml.qnode(dev, interface="torch")
def quantum_circuit(params):
    qml.Hadamard(wires=0)
    qml.RY(params[0], wires=0)
    qml.RY(params[1], wires=1)
    qml.CNOT(wires=[0, 1])
    return qml.state()

def superposed_focal_loss(params, y_true1, y_true2, alpha=0.25, gamma=2.0):
    rho = quantum_circuit(params) 
    probs = torch.diag(rho).real

    y_pred1 = probs[1]  
    y_pred2 = probs[2]  

    y_true1 = torch.tensor(y_true1, dtype=torch.float32)
    y_true2 = torch.tensor(y_true2, dtype=torch.float32)

    
    loss1 = -alpha * (1 - y_pred1) ** gamma * y_true1 * torch.log(y_pred1 + 1e-8)
    loss2 = -alpha * (1 - y_pred2) ** gamma * y_true2 * torch.log(y_pred2 + 1e-8)

    return loss1 + loss2


params = torch.tensor(np.random.rand(2), dtype=torch.float32, requires_grad=True)


optimizer = torch.optim.Adam([params], lr=0.1)


for epoch in range(50):
    optimizer.zero_grad()
    loss = superposed_focal_loss(params, y_true1=1, y_true2=1)  # İki zor sınıfı da 1 olarak kabul ettik
    loss.backward()
    optimizer.step()
    if epoch % 10 == 0:
        print(f"Epoch {epoch}: Loss = {loss.item()}")
