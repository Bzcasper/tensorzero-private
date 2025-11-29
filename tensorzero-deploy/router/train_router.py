import torch
import torch.nn as nn
import torch.onnx
from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders, processors

# 1. Define the Model (FastGRNN-like / GRU)
class NeuralRouterModel(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, output_dim):
        super(NeuralRouterModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.fc = nn.Linear(embed_dim, output_dim)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        # x: [batch, seq_len]
        embedded = self.embedding(x)
        # Mean pooling
        pooled = torch.mean(embedded, dim=1)
        logits = self.fc(pooled)
        probs = self.softmax(logits)
        return probs

# 2. Train/Save Tokenizer
def create_tokenizer(path="tokenizer.json"):
    tokenizer = Tokenizer(models.BPE())
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()
    
    trainer = trainers.BpeTrainer(vocab_size=1000, min_frequency=2, special_tokens=["<unk>", "<s>", "</s>"])
    
    # Train on some dummy data
    data = [
        "Write a python script to sort a list",
        "Explain quantum physics",
        "Who is the president of the US?",
        "Translate hello to spanish",
        "Generate a creative story about a robot",
        "Fix this bug in my code",
        "What is the capital of France?",
        "Write a poem about the sea",
        "Summarize this article",
        "How do I cook pasta?"
    ]
    
    tokenizer.train_from_iterator(data, trainer=trainer)
    tokenizer.save(path)
    return tokenizer

# 3. Export Model
def export_model(model_path="router.onnx", tokenizer_path="tokenizer.json"):
    # Config
    VOCAB_SIZE = 1000
    EMBED_DIM = 64
    HIDDEN_DIM = 32
    OUTPUT_DIM = 9 # Number of variants in tensorzero.toml (approx)
    
    # Create Tokenizer
    create_tokenizer(tokenizer_path)
    
    # Create Model
    model = NeuralRouterModel(VOCAB_SIZE, EMBED_DIM, HIDDEN_DIM, OUTPUT_DIM)
    model.eval()
    
    # Dummy Input
    dummy_input = torch.randint(0, VOCAB_SIZE, (1, 10), dtype=torch.long)
    
    # Export
    try:
        torch.onnx.export(
            model, 
            dummy_input, 
            model_path, 
            input_names=["input_ids"], 
            output_names=["output"],
            dynamic_axes={
                "input_ids": {0: "batch_size", 1: "seq_len"},
                "output": {0: "batch_size"}
            },
            opset_version=14
        )
    except Exception as e:
        print(f"Export failed with dynamic axes: {e}")
        print("Retrying with static axes...")
        torch.onnx.export(
            model, 
            dummy_input, 
            model_path, 
            input_names=["input_ids"], 
            output_names=["output"],
            opset_version=14
        )
    print(f"Exported model to {model_path} and tokenizer to {tokenizer_path}")

if __name__ == "__main__":
    export_model()
