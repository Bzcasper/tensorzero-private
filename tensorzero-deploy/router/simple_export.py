import torch
import torch.nn as nn

class NeuralRouterModel(nn.Module):
    def __init__(self):
        super(NeuralRouterModel, self).__init__()
        self.embedding = nn.Embedding(1000, 64)
        self.gru = nn.GRU(64, 32, batch_first=True)
        self.fc = nn.Linear(32, 9)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        embedded = self.embedding(x)
        _, hn = self.gru(embedded)
        hidden = hn[-1]
        logits = self.fc(hidden)
        return self.softmax(logits)

model = NeuralRouterModel()
dummy_input = torch.randint(0, 1000, (1, 10), dtype=torch.long)

print("Exporting...")
torch.onnx.export(
    model, 
    dummy_input, 
    "router.onnx", 
    input_names=["input_ids"], 
    output_names=["output"],
    opset_version=14
)
print("Done.")
