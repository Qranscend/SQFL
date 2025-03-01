import pennylane as qml
import torch
import numpy as np

n_qubits = 3  
dev = qml.device("default.mixed", wires=n_qubits)

@qml.qnode(dev, interface="torch")
def quantum_circuit(params):
    for i in range(3):  
        qml.Hadamard(wires=i)  
        
    for i in range(3):  
        qml.RX(params[i], wires=i)
        qml.RZ(params[i + 3], wires=i)
    
    qml.CRX(params[6], wires=[0, 1])
    qml.CRX(params[7], wires=[1, 2])
    qml.CRX(params[8], wires=[2, 0])
    
    qml.CZ(wires=[0, 2])
    qml.MultiControlledX(wires=[0, 1, 2], control_values=[1, 1]) # CCNOT = Toffoli
    qml.SWAP(wires=[1, 2])
    
    for i in range(3):  
        qml.RX(params[i + 9], wires=i)
        qml.RZ(params[i + 12], wires=i)
    
    return qml.state()

def superposed_focal_loss(params, y_true1, y_true2, y_true3, alpha=0.25, gamma=2.0):
    rho = quantum_circuit(params)  
    probs = torch.diag(rho).real

    y_pred1 = probs[1]
    y_pred2 = probs[2] 
    y_pred3 = probs[4]  

    y_true1 = torch.tensor(y_true1, dtype=torch.float32)
    y_true2 = torch.tensor(y_true2, dtype=torch.float32)
    y_true3 = torch.tensor(y_true3, dtype=torch.float32)

    loss1 = -alpha * (1 - y_pred1) ** gamma * y_true1 * torch.log(y_pred1 + 1e-8)
    loss2 = -alpha * (1 - y_pred2) ** gamma * y_true2 * torch.log(y_pred2 + 1e-8)
    loss3 = -alpha * (1 - y_pred3) ** gamma * y_true3 * torch.log(y_pred3 + 1e-8)

    return loss1 + loss2 + loss3


params = torch.tensor(np.random.rand(15), dtype=torch.float32, requires_grad=True)  

optimizer = torch.optim.Adam([params], lr=0.1)

for epoch in range(50):
    optimizer.zero_grad()
    loss = superposed_focal_loss(params, y_true1=1, y_true2=1, y_true3=1)
    loss.backward()
    optimizer.step()
    if epoch % 10 == 0:
        print(f"Epoch {epoch}: Loss = {loss.item()}")
