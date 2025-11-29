const { NeuralRouter } = require('./router-ffi/router-ffi.linux-x64-gnu.node');

async function testRouter() {
  console.log('Testing NeuralRouter integration...');
  
  try {
    const router = new NeuralRouter('./router.onnx', './tokenizer.json');
    console.log('Router loaded successfully');
    
    const testTexts = [
      'Write a Python function to sort a list',
      'Explain quantum physics',
      'What is the weather today?',
      'Debug this JavaScript code',
      'Write a creative story'
    ];
    
    for (const text of testTexts) {
      const scores = router.route(text);
      const sum = scores.reduce((a, b) => a + b, 0);
      const maxIndex = scores.indexOf(Math.max(...scores));
      
      console.log(`\nText: "${text}"`);
      console.log(`Scores: [${scores.map(s => s.toFixed(4)).join(', ')}]`);
      console.log(`Sum: ${sum.toFixed(4)} (should be ~1.0)`);
      console.log(`Best variant: ${maxIndex}`);
    }
    
    console.log('\n✅ Router integration test passed!');
    
  } catch (error) {
    console.error('❌ Router test failed:', error);
    process.exit(1);
  }
}

testRouter();
