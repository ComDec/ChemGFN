<div align="center">

# ChemGFN: Generative Flow Networks for Molecular Design

[![python](https://img.shields.io/badge/-Python_3.9_%7C_3.10-blue?logo=python&logoColor=white)](https://www.python.org/)
[![pytorch](https://img.shields.io/badge/PyTorch_2.0+-ee4c2c?logo=pytorch&logoColor=white)](https://pytorch.org/get-started/locally/)
[![lightning](https://img.shields.io/badge/-Lightning_2.0+-792ee5?logo=pytorchlightning&logoColor=white)](https://pytorchlightning.ai/)
[![hydra](https://img.shields.io/badge/Config-Hydra_1.3-89b8cd)](https://hydra.cc/)
[![transformers](https://img.shields.io/badge/🤗-Transformers-yellow)](https://huggingface.co/transformers/)
[![license](https://img.shields.io/badge/License-MIT-green.svg?labelColor=gray)](#license)

A powerful framework for molecular generation using Generative Flow Networks (GFlowNets) with Large Language Models.

[Features](#-features) • [Installation](#-installation) • [Quick Start](#-quick-start) • [Documentation](#-documentation) • [Citation](#-citation)

</div>

---

## 📌 Introduction

**ChemGFN** is a state-of-the-art framework that combines **Generative Flow Networks** with **Large Language Models** for molecular design and optimization. The framework enables efficient exploration of chemical space while maintaining syntactic validity through grammar-constrained generation.

### Key Highlights

✨ **LLM-based Molecular Generation**
Leverages pre-trained language models (GPT2, LLaMA-3) with LoRA fine-tuning for efficient molecular generation.

🎯 **Grammar-Constrained Sampling**
Ensures 100% syntactically valid SMILES strings through EBNF grammar constraints.

⚡ **Efficient Training**
- Modified Sub-Trajectory Balance (SubTB) loss with token-level balancing
- Replay buffer for high-quality sample reuse
- Dataset buffer mixing for improved exploration

🧪 **Flexible Reward Functions**
- Support for multiple molecular properties (logP, QED, SA, etc.)
- Custom reward functions via RDKit or PartialSMILES backend
- Target-guided optimization

🔬 **Molecular Optimization**
- De novo generation
- Scaffold-based design
- Sidechain optimization

---

## 🌟 Features

### Core Capabilities

- **Multiple Model Backends**: GPT2, GPT-J, LLaMA-3.2-1B/8B with LoRA support
- **Grammar Enforcement**: EBNF-based grammar processors (prefix/infix modes)
- **Advanced Training**:
  - Modified SubTB loss with token coverage balancing
  - Replay buffer with similarity-based deduplication
  - Dynamic temperature and scaling factor scheduling
- **Molecular Validation**:
  - RDKit-based SMILES validation
  - PartialSMILES incremental validation
  - Custom scoring functions (logP, QED, SA, similarity)
- **Production Ready**:
  - Comprehensive test suite (150+ tests)
  - Hydra-based configuration management
  - PyTorch Lightning training pipeline
  - Multi-GPU and distributed training support

---

## 📦 Installation

### Prerequisites

- Python 3.9 or 3.10
- CUDA 11.8+ (for GPU support)
- conda or mamba (recommended)

### Option 1: Conda Environment (Recommended)

```bash
# Clone the repository
git clone https://github.com/yourusername/ChemGFN.git
cd ChemGFN

# Create conda environment
conda env create -f environment.yaml
conda activate chemgfn

# Install package in editable mode
pip install -e .
```

### Option 2: Pip Installation

```bash
# Clone and install
git clone https://github.com/yourusername/ChemGFN.git
cd ChemGFN

# Install dependencies
pip install -r requirements.txt

# Install package
pip install -e .
```

### Verify Installation

```bash
# Run tests
pytest tests/ -v

# Or use the test script
./run_tests.sh all
```

---

## 🚀 Quick Start

### Basic Training

```bash
# Train with default configuration
python chemgfn/train.py

# Train with specific experiment config
python chemgfn/train.py experiment=baseline_expr24

# Debug mode (fast iteration)
python chemgfn/train.py experiment=debug
```

### Custom Training

```bash
# Override specific parameters
python chemgfn/train.py \
  model.lora_config.r=16 \
  model.training_mixed_config.n_samples=16 \
  trainer.max_epochs=50

# Use different model
python chemgfn/train.py \
  model=llama3_smiles \
  model.net_config.pretrained_model_name_or_path="meta-llama/Llama-3.2-1B"

# Multi-GPU training
python chemgfn/train.py \
  trainer.devices=4 \
  trainer.strategy=ddp
```

### Evaluation

```bash
# Evaluate trained model
python chemgfn/eval.py \
  ckpt_path="logs/train/runs/YYYY-MM-DD_HH-MM-SS/checkpoints/last.ckpt"
```

---

## 📚 Documentation

### Configuration Guide

ChemGFN uses Hydra for configuration management. Key configuration files:

- **Model Config** (`configs/model/`): Model architecture, LoRA, rewards, constraints
- **Experiment Config** (`configs/experiment/`): Override specific parameters
- **Data Config** (`configs/data/`): Dataset and buffer sampling settings
- **Trainer Config** (`configs/trainer/`): PyTorch Lightning trainer settings

For detailed configuration guide, see [Configuration Guide](chemgfn/utils/configs/CONFIG_GUIDE.md).

### Key Configuration Parameters

#### Model Configuration

```yaml
# Model architecture
net_config:
  pretrained_model_name_or_path: "meta-llama/Llama-3.2-1B"

# LoRA fine-tuning
lora_config:
  r: 16                    # LoRA rank
  lora_alpha: 16           # LoRA scaling
  lora_dropout: 0.1

# Reward function
reward:
  _target_: chemgfn.models.reward.Reference_Target_Score_Positive_Mixed_Invalid_Mask
  illegal_vocab_penalty: -50
  grammar_disagree_penalty: -99
  sentence_validator:
    scorer: "logP"         # logP, QED, SA, similarity
    backend: "pa"          # rdkit or pa

# Generation constraints
constraint_config:
  min_sentence_len: 2
  max_sentence_len: 10
  grammar_path: ${paths.assets_dir}/SMILES_grammars/generic.ebnf
  apply_grammar: true
  illegal_vocab_penalty: -50

# Training configuration
training_mixed_config:
  subtb_lambda: 1.0        # SubTB length decay weight
  n_samples: 8             # Batch size
  use_buffer_prob: 0.25    # Replay buffer usage probability
  balance_start: 0.0       # Token balancing (start)
  balance_end: 1.0         # Token balancing (end)
  balance_horizon: 50000   # Steps to reach end balance
```

#### Experiment Configuration

```yaml
# Override only necessary parameters
model:
  reward:
    illegal_vocab_penalty: -80
    sentence_validator:
      scorer: "QED"

trainer:
  max_epochs: 100
```

---

## 🔬 Advanced Usage

### Custom Reward Functions

```python
# Define custom reward function
from chemgfn.models.reward import SentenceValidator

class CustomValidator(SentenceValidator):
    def __call__(self, sentences, tokenizer, target_molecule=None):
        # Your validation logic
        return {
            "invalid": invalid_mask,
            "valid_score": scores,
        }

# Use in configuration
reward:
  sentence_validator:
    _target_: your_module.CustomValidator
    # your parameters
```

### Buffer Sampling

```python
# Create buffer samples
python chemgfn/utils/buffer_sample_example.py \
  --action create \
  --output data/my_buffer.pt \
  --num-samples 1000

# Use in training
python chemgfn/train.py \
  data.buffer_sample_path=data/my_buffer.pt
```

### Distributed Training

```bash
# DDP on single node
python chemgfn/train.py \
  trainer.devices=8 \
  trainer.strategy=ddp

# DDP on multiple nodes (SLURM)
sbatch scripts/slurm_train.sh
```

---

## 🧪 Testing

ChemGFN includes a comprehensive test suite covering all major components:

```bash
# Run all tests
pytest tests/ -v

# Run specific test suite
pytest tests/test_gfn_utils.py -v     # GFN utilities
pytest tests/test_reward.py -v        # Reward functions
pytest tests/test_loss.py -v          # Loss functions
pytest tests/test_training_flow.py -v # Training pipeline

# Run with coverage
pytest tests/ --cov=chemgfn --cov-report=html

# Use convenience script
./run_tests.sh all          # All tests
./run_tests.sh fast         # Skip slow tests
./run_tests.sh coverage     # With coverage report
./run_tests.sh parallel     # Parallel execution
```

For detailed testing documentation, see [Test Suite Documentation](tests/README_TESTS.md).

---

## 📊 Project Structure

```
ChemGFN/
├── chemgfn/                    # Main package
│   ├── models/                 # Model implementations
│   │   ├── gfn.py             # GFlowNet module
│   │   └── reward.py          # Reward functions
│   ├── data/                   # Data modules
│   │   └── gfn_datamodule.py
│   ├── utils/                  # Utilities
│   │   ├── gfn_utils.py       # GFN core utilities
│   │   ├── cfg_grammar.py     # Grammar processors
│   │   └── rdkit_utils.py     # RDKit utilities
│   ├── train.py               # Training script
│   └── eval.py                # Evaluation script
│
├── configs/                    # Hydra configurations
│   ├── model/                 # Model configs
│   ├── experiment/            # Experiment configs
│   ├── data/                  # Data configs
│   └── trainer/               # Trainer configs
│
├── tests/                      # Test suite
│   ├── test_gfn_utils.py
│   ├── test_reward.py
│   ├── test_loss.py
│   └── test_training_flow.py
│
├── data/                       # Data directory
│   ├── SMILES/                # SMILES datasets
│   └── 24_points/             # 24-game datasets
│
├── assets/                     # Grammar and vocab files
│   ├── SMILES_grammars/
│   └── token_list/
│
└── notebooks/                  # Jupyter notebooks
```

---

## 🎯 Key Components

### 1. GFlowNet Model (`chemgfn/models/gfn.py`)

- LightningModule-based training
- LoRA fine-tuning support
- Modified SubTB loss with token balancing
- Replay buffer integration
- Dynamic scheduling (temperature, scaling factors)

### 2. Reward Functions (`chemgfn/models/reward.py`)

- `FrozenModelSentenceGivenPrompt`: Base model reward
- `Reference_Target_Score_Positive_Mixed_Invalid_Mask`: Mixed reward with validation
- `RDKitValidator`: SMILES validation via RDKit
- `PartialSMILESValidator`: Incremental validation

### 3. Loss Functions (`chemgfn/utils/gfn_utils.py`)

- `modified_subtb_loss`: SubTB with token-coverage balancing
- Supports early termination
- Gradient-friendly implementation

### 4. Grammar Processors (`chemgfn/utils/cfg_grammar.py`)

- EBNF grammar parsing
- Prefix/infix constraint modes
- Efficient logit masking

---

## 📖 Tutorials and Examples

### Example 1: Train for logP Optimization

```bash
python chemgfn/train.py \
  experiment=baseline_expr24 \
  model.reward.sentence_validator.scorer=logP \
  model.reward_config.scaling_factor_start=50 \
  model.reward_config.scaling_factor_end=100
```

### Example 2: Target-based Similarity Optimization

```yaml
# configs/experiment/similarity_opt.yaml
model:
  reward:
    sentence_validator:
      scorer: "similarity"
      target: "CCO"  # Target molecule
      threshold: 0.7
```

```bash
python chemgfn/train.py experiment=similarity_opt
```

### Example 3: Multi-objective Optimization

```python
# Custom multi-objective validator
class MultiObjectiveValidator(SentenceValidator):
    def score(self, smiles):
        logp = Descriptors.MolLogP(mol)
        qed = QED.qed(mol)
        sa = sascorer.calculateScore(mol)
        return 0.4 * logp + 0.4 * qed - 0.2 * sa
```

---

## 🔧 Troubleshooting

### Common Issues

**1. CUDA Out of Memory**

```bash
# Reduce batch size
python chemgfn/train.py model.training_mixed_config.n_samples=4

# Enable gradient checkpointing
python chemgfn/train.py model.compile=true
```

**2. Reward NaN**

```yaml
# Check temperature settings (must be > 0)
reward_config:
  reward_temp_end: 0.8  # Not 0!

# Check penalty values (not too negative)
reward:
  illegal_vocab_penalty: -80   # Not -1000!
```

**3. Grammar Issues**

```bash
# Verify grammar file
python -c "from chemgfn.utils.cfg_grammar import load_grammar; \
           load_grammar('assets/SMILES_grammars/generic.ebnf')"
```

**4. Configuration Errors**

```bash
# Validate configuration
python validate_config.py experiment=your_experiment

# Print full config
python chemgfn/train.py --cfg job
```

---

## 🤝 Contributing

We welcome contributions! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes
4. Run tests (`pytest tests/`)
5. Commit your changes (`git commit -m 'Add amazing feature'`)
6. Push to the branch (`git push origin feature/amazing-feature`)
7. Open a Pull Request

### Development Setup

```bash
# Install development dependencies
pip install -e ".[dev]"

# Install pre-commit hooks
pre-commit install

# Run linters
pre-commit run --all-files
```

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 📝 Citation

If you use ChemGFN in your research, please cite:

```bibtex
@software{chemgfn2024,
  title={ChemGFN: Generative Flow Networks for Molecular Design},
  author={Your Name},
  year={2024},
  url={https://github.com/yourusername/ChemGFN}
}
```

---

## 🙏 Acknowledgments

This project builds upon:

- [PyTorch Lightning](https://github.com/Lightning-AI/lightning) for training infrastructure
- [Hydra](https://github.com/facebookresearch/hydra) for configuration management
- [Transformers](https://github.com/huggingface/transformers) for pre-trained models
- [RDKit](https://github.com/rdkit/rdkit) for molecular validation
- [PartialSMILES](https://github.com/MolecularAI/Partial-SMILES) for incremental validation

---

## 📞 Contact

- **Issues**: [GitHub Issues](https://github.com/yourusername/ChemGFN/issues)
- **Discussions**: [GitHub Discussions](https://github.com/yourusername/ChemGFN/discussions)
- **Email**: your.email@example.com

---

<div align="center">

**[⬆ back to top](#chemgfn-generative-flow-networks-for-molecular-design)**

Made with ❤️ by the ChemGFN team

</div>
