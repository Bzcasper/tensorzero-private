import torch
import torch.nn as nn

class DummyModel(nn.Module):
    def __init__(self):
        super(DummyModel, self).__init__()
        self.linear = nn.Linear(1, 1)

    def forward(self, x):
        return self.linear(x)

model = DummyModel()
dummy_input = torch.randn(1, 1)
torch.onnx.export(model, dummy_input, "dummy_model.onnx", input_names=["input"], output_names=["output"])
print("Created dummy_model.onnx")
