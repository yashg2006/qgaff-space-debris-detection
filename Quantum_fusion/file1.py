import pennylane as qml
import torch
import torch.nn as nn

n_qubits = 4
dev = qml.device("default.qubit", wires=n_qubits)

@qml.qnode(dev, interface="torch")
def quantum_circuit(inputs, weights):
    qml.AngleEmbedding(inputs, wires=range(n_qubits))
    qml.StronglyEntanglingLayers(weights, wires=range(n_qubits))
    return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]


class QuantumLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.weights = nn.Parameter(torch.randn(1, n_qubits, 3))

    def forward(self, x):
        outputs = []
        for i in range(x.shape[0]):
            q_out = quantum_circuit(x[i], self.weights)
            # Ensure q_out is a tensor while preserving the computation graph
            if isinstance(q_out, (list, tuple)):
                q_out = torch.stack(q_out)
            outputs.append(q_out)
        return torch.stack(outputs).to(torch.float32)


class QuantumFusionModel(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.compress = nn.Linear(256, 4)
        self.quantum = QuantumLayer()
        self.classifier = nn.Linear(4, num_classes)

    def forward(self, features):
        x = self.compress(features.float())
        x = self.quantum(x)
        return self.classifier(x)
