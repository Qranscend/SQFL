import pennylane as qml
import torch
import numpy as np

n_qubits = 8  
dev = qml.device("default.mixed", wires=n_qubits)

@qml.qnode(dev, interface="torch")
def quantum_circuit(params):

    for i in range(n_qubits):  
        qml.Hadamard(wires=i)
    

    for i in range(n_qubits):  
        qml.RX(params[i], wires=i)
        qml.RZ(params[i + n_qubits], wires=i)
    

    for i in range(n_qubits - 1):  
        qml.CRX(params[2 * n_qubits + i], wires=[i, i + 1])


    for i in range(0, n_qubits - 2, 2):
        qml.MultiControlledX(wires=[i, i + 1, i + 2], control_values=[1, 1]) 
    

    qml.SWAP(wires=[0, n_qubits - 1])
    qml.SWAP(wires=[1, n_qubits - 2])


    for i in range(n_qubits):  
        qml.RX(params[i + 3 * n_qubits], wires=i)
        qml.RZ(params[i + 4 * n_qubits], wires=i)

    return qml.state()

def superposed_focal_loss(params, y_true, alpha=0.25, gamma=2.0):
    rho = quantum_circuit(params)  
    probs = torch.diag(rho).real
    
 
    losses = 0
    for i in range(n_qubits):
        y_pred = probs[2**i] 
        y_true_i = torch.tensor(y_true[i], dtype=torch.float32)

        loss = -alpha * (1 - y_pred) ** gamma * y_true_i * torch.log(y_pred + 1e-8)
        losses += loss

    return losses  

params = torch.tensor(np.random.rand(5 * n_qubits), dtype=torch.float32, requires_grad=True)  

optimizer = torch.optim.Adam([params], lr=0.1)


for epoch in range(50):
    optimizer.zero_grad()
    y_true = [1] * n_qubits 
    loss = superposed_focal_loss(params, y_true)
    loss.backward()
    optimizer.step()
    if epoch % 10 == 0:
        print(f"Epoch {epoch}: Loss = {loss.item()}")
