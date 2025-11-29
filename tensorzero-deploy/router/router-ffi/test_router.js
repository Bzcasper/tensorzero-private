const { NeuralRouter } = require('./index.js');
const path = require('path');

const modelPath = path.join(__dirname, '../router.onnx');
const tokenizerPath = path.join(__dirname, '../tokenizer.json');

console.log(`Loading model from ${modelPath}`);
console.log(`Loading tokenizer from ${tokenizerPath}`);

try {
  const router = new NeuralRouter(modelPath, tokenizerPath);
  console.log('Router loaded successfully');

  const text = "Write a python script to sort a list";
  console.log(`Routing text: "${text}"`);
  
  const probs = router.route(text);
  console.log('Probabilities:', probs);
  
  // Check if probs sum to ~1
  const sum = probs.reduce((a, b) => a + b, 0);
  console.log('Sum:', sum);
  
  if (Math.abs(sum - 1.0) > 0.01) {
      console.error('Probabilities do not sum to 1');
      process.exit(1);
  }

} catch (e) {
  console.error('Error:', e);
  process.exit(1);
}
