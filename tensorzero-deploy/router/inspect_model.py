import onnx

model = onnx.load("dummy_model.onnx")
print("Input:")
for input in model.graph.input:
    print(input.name, input.type.tensor_type.shape)

print("\nOutput:")
for output in model.graph.output:
    print(output.name, output.type.tensor_type.shape)

print("\nNodes:")
for node in model.graph.node:
    print(node.op_type, node.name)
