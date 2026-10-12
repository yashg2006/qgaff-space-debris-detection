import pennylane as qml
import torch
import torch.nn as nn
import torch.nn.functional as F

n_qubits = 4
dev = qml.device("default.qubit", wires=n_qubits)

@qml.qnode(dev, interface="torch")
def quantum_circuit(inputs, weights):
    # AngleEmbedding to encode 4 features into 4 qubits
    qml.AngleEmbedding(inputs, wires=range(n_qubits))
    # StronglyEntanglingLayers for parametrized quantum circuit
    qml.StronglyEntanglingLayers(weights, wires=range(n_qubits))
    # Return expectation values (4-dim vector)
    return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]

class QuantumFusionModel(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        # 1. Linear compression (256 -> 4)
        self.compress = nn.Linear(256, 4)
        
        # 2. PennyLane TorchLayer
        weight_shapes = {"weights": (2, n_qubits, 3)} # 2 layers
        self.q_layer = qml.qnn.TorchLayer(quantum_circuit, weight_shapes)
        
        # 3. Final Classifier (4 -> num_classes)
        self.classifier = nn.Linear(4, num_classes)
        
    def forward(self, x, return_features=False):
        # Initial input check
        x = x.float()
        
        # Compression
        x = self.compress(x)
        
        # Normalization for quantum embedding (important for AngleEmbedding)
        # We wrap it in tanh to keep values in [-pi, pi] or similar spread
        # although AngleEmbedding handles any value, -pi/2 to pi/2 is standard.
        x = torch.tanh(x) * (3.14159 / 2.0)
        
        # Quantum Layer
        q_out = self.q_layer(x)
        
        # Final classification
        logits = self.classifier(q_out)
        
        if return_features:
            return logits, q_out
        return logits
