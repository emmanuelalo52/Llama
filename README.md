# LLama Implementation

This repository provides a custom implementation of a transformer-based LLama model. It includes essential components for model architecture, training, inference, and extending language model capabilities with efficient configurations.

## Features

- **LLama Transformer Architecture**: Built from scratch with support for customizable dimensions, layers, and heads.
- **Text Completion**: Supports text completion and zero/few-shot prompt generation.
- **Differential Attention Mechanism**: Utilizes advanced attention mechanisms for enhanced model performance.
- **Rotary Positional Embeddings**: Implements rotary positional embeddings for improved sequence modeling.
- **Optimized Inference**: Includes top-p sampling and temperature scaling for generating diverse outputs.

---

## Directory Structure

```
|-- inference.py
|-- llama_architecture.py
|-- using_difftrans.py
```

### Key Files

1. **inference.py**
   - Entry point for inference with a pre-trained LLama model.
   - Implements text generation with options for sampling methods and prompt handling.

2. **llama_architecture.py**
   - Contains the core model architecture, including attention mechanisms, feed-forward layers, and rotary positional embeddings.
   - Defines the `LlamaConfig` dataclass for customizing model configurations.

3. **using_difftrans.py**
   - Explores differential attention mechanisms.
   - Provides additional experimental functionality for extending the core LLama architecture.

---

## Installation

1. Clone the repository:
   ```bash
   git clone <repository_url>
   cd <repository_name>
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

   Ensure you have PyTorch installed with GPU support for optimal performance:
   ```bash
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu117
   ```

---

## Usage

### Inference
Run the `inference.py` file to test text generation with a pre-trained LLama model:
```bash
python inference.py
```
Modify `prompts` in `inference.py` to experiment with custom input text.

### Model Customization
Adjust the configuration in `llama_architecture.py`:
```python
config = LlamaConfig(
    dim=4096,
    n_layers=32,
    n_heads=32,
    max_seq_len=2048,
    max_batch_size=32,
    device='cuda'
)
```

---

## Requirements

- Python 3.8+
- PyTorch 2.0+
- tqdm
- sentencepiece

---

## Contributions

Feel free to contribute by opening issues or submitting pull requests. Ensure code changes are well-documented and tested.

---

## License

This project is licensed under the MIT License. See the LICENSE file for details.

---

## Acknowledgments

Thanks to the PyTorch and Hugging Face communities for resources and inspiration. This project is tailored for enthusiasts and professionals seeking efficient and extendable language model implementations.
