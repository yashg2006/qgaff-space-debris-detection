import pennylane as qml
import torch
import torch.nn as nn

# Spark dataset: 11 classes
SPARK_CLASSES = [
    "debris", "lisa_pathfinder", "soho", "xmm_newton", "cheops",
    "smart_1", "proba_2", "proba_3_csc", "proba_3_ocs",
    "earth_observation_sat_1", "double_star"
]
NUM_CLASSES = len(SPARK_CLASSES)   # 11

N_QUBITS = 4
dev = qml.device("default.qubit", wires=N_QUBITS)

@qml.qnode(dev, interface="torch")
def quantum_circuit(inputs, weights):
    qml.AngleEmbedding(inputs, wires=range(N_QUBITS))
    qml.StronglyEntanglingLayers(weights, wires=range(N_QUBITS))
    return [qml.expval(qml.PauliZ(i)) for i in range(N_QUBITS)]


class QuantumLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.weights = nn.Parameter(torch.randn(1, N_QUBITS, 3))

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
    """
    compress: 256 → 4   (match N_QUBITS)
    quantum : 4  → 4    (PauliZ expectations)
    classify : 4  → 11  (Spark classes)
    """
    def __init__(self, num_classes=NUM_CLASSES):
        super().__init__()
        self.compress   = nn.Linear(256, N_QUBITS)
        self.quantum    = QuantumLayer()
        self.classifier = nn.Linear(N_QUBITS, num_classes)

    def forward(self, features):
        x = self.compress(features.float())
        x = self.quantum(x)
        return self.classifier(x)
