# Complete LLM Project - Fast CSE Model# SimpleGPT: An Efficient Small-Scale Language Model for Computer Science Education



A lightweight language model specifically designed for Computer Science educational content generation and understanding.[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)

## 🏗️ Model Architecture[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)



**Model:** SimpleGPT-19M  ## Overview

**Parameters:** 19,053,568 (19.05M)  

**Architecture:** 6-layer Transformer with character-level tokenization  SimpleGPT is a lightweight, educational implementation of a GPT-style transformer language model designed specifically for computer science education and research. With only 19 million parameters, it demonstrates efficient training and evaluation methodologies while maintaining competitive performance on computer science domain tasks.



## 📊 Latest Model Performance (fast_cse_model_final)## Key Features



- **Training Loss:** 1.441- **Compact Architecture**: 19M parameters optimized for educational use and resource-constrained environments

- **Validation Loss:** 1.207  - **Enhanced Training Pipeline**: Includes advanced regularization techniques (dropout, weight decay, early stopping)

- **Perplexity:** 3.34- **Comprehensive Evaluation Suite**: Multiple evaluation scripts for performance assessment and comparison

- **Training Date:** September 17, 2025- **Educational Focus**: Trained on computer science textbooks and academic content

- **Corpus:** 79.6MB CS educational content- **Production-Ready**: Includes Docker support and comprehensive testing framework



## 🚀 Quick Start## Architecture



### Training- **Model Size**: 19 million parameters

```bash- **Context Length**: 512 tokens

python src/fast_cse_trainer.py- **Vocabulary**: Character-level tokenization

```- **Architecture**: Standard transformer decoder with 12 layers, 12 attention heads, 768 hidden dimensions

- **Regularization**: Dropout (0.15), weight decay (0.05), gradient clipping

### Evaluation

```bash## Quick Start

python src/simple_gpt_evaluator.py

```### Installation



## 📁 Project Structure```bash

git clone https://github.com/your-username/simplegpt

```cd simplegpt

├── src/                    # Training and evaluation codepip install -r requirements.txt

├── models/                 # Model checkpoints and results```

├── configs/               # Training configurations

├── docs/                  # Documentation### Training

├── tests/                 # Model testing scripts

└── data/                  # Training corpus (excluded from git)```bash

```python src/fast_cse_trainer.py

```

## 📋 Model Specifications

### Evaluation

- **Layers:** 6 Transformer encoder layers

- **Hidden Dimension:** 512```bash

- **Attention Heads:** 8python src/simple_gpt_evaluator.py --model_path models/fast_cse_model_epoch_1

- **Vocabulary:** 34 character-level tokenspython src/sota_evaluation_system.py --model_path models/fast_cse_model_epoch_1

- **Sequence Length:** 256 tokens```

- **Dropout:** 0.15

### Testing

## 📊 Performance History

```bash

| Model | Epoch | Perplexity | Training Data |python tests/simple_gpt_cse_tester.py

|-------|-------|------------|---------------|python tests/simple_gpt_checkpoint_tester.py

| fast_cse_model_epoch_1 | 1 | 2.57 | Small corpus |```

| fast_cse_model_epoch_2 | 2 | 2.70 | Small corpus |

| fast_cse_model_final | 1 | 3.34 | Full corpus (79.6MB) |## Project Structure



## 📄 Citation```

simplegpt/

```bibtex├── src/                              # Core source code

@software{fast_cse_model,│   ├── fast_cse_trainer.py          # Main training script

  title={Fast CSE Model: Lightweight Language Model for CS Education},│   ├── ultra_fast_training_rtx4060.py # RTX-optimized training

  author={AbhayRao38},│   ├── simple_gpt_evaluator.py      # Primary evaluation script

  year={2025},│   ├── sota_evaluation_system.py    # SOTA comparison evaluation

  url={https://github.com/AbhayRao38/LLM}│   ├── simple_gpt_cse_tester.py     # CSE-specific testing

}│   └── simple_gpt_checkpoint_tester.py # Checkpoint validation

```├── models/                           # Trained model checkpoints

│   ├── fast_cse_model_epoch_1/      # Epoch 1 checkpoint

## ⚠️ Note on Large Files│   ├── fast_cse_model_epoch_2/      # Epoch 2 checkpoint

│   └── fast_cse_model_epoch_2_best/ # Best performing model

Model weights (.pt files) and large corpus files are excluded from this repository due to GitHub size limits. The model can be retrained using the provided scripts.├── data/                            # Training and evaluation data

│   ├── cse_corpus_robust.txt        # Primary training corpus

## 🔧 Requirements│   ├── enhanced_cse_corpus_v2.txt   # Enhanced training data

│   ├── fast_cleaned_corpus.txt      # Cleaned corpus variant

- PyTorch│   ├── massive_cs_corpus.txt        # Large-scale corpus

- CUDA (for GPU training)│   └── ultra_fast_cs_corpus.txt     # Optimized corpus

- Python 3.8+├── tests/                           # Test scripts

├── configs/                         # Configuration files

See `requirements.txt` for full dependencies.├── docs/                           # Documentation
├── Dockerfile                      # Container configuration
└── requirements.txt               # Python dependencies
```

## Training Data

The model is trained on a comprehensive corpus of computer science textbooks and academic materials, totaling approximately 80MB of text data. The corpus includes:

- Computer science textbooks
- Academic papers and tutorials
- Programming documentation
- Algorithm descriptions and explanations

## Performance

- **Training Time**: ~2-3 hours on RTX 4060
- **Memory Usage**: ~12GB GPU memory
- **Perplexity**: Competitive with similar-sized models
- **Domain Performance**: Optimized for computer science tasks

## Evaluation

The project includes a comprehensive evaluation suite:

1. **Perplexity Evaluation**: Standard language modeling metrics
2. **SOTA Comparison**: Benchmarking against state-of-the-art models
3. **Domain-Specific Testing**: Computer science knowledge assessment
4. **Checkpoint Validation**: Model consistency and quality checks

## Configuration

Training and model configurations are stored in the `configs/` directory:

- `rtx4060_optimized_config.json`: RTX 4060-optimized settings
- `training_config_*.json`: Various training configurations
- `system_config.yaml`: System-wide settings

## Contributing

We welcome contributions! Please see our [contributing guidelines](docs/CONTRIBUTING.md) for details.

## Citation

If you use SimpleGPT in your research, please cite:

```bibtex
@software{simplegpt2024,
  title={SimpleGPT: An Efficient Small-Scale Language Model for Computer Science Education},
  author={Your Name},
  year={2024},
  url={https://github.com/your-username/simplegpt}
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Built with PyTorch and Transformers library
- Inspired by the GPT architecture and educational AI initiatives
- Thanks to the open-source community for tools and resources

## Support

For questions and support, please open an issue on GitHub or contact [your-email@domain.com].