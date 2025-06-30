# Early AKI Prediction using Word2Vec Token Embeddings from MIMIC-III

This repository contains the implementation of a novel approach to early Acute Kidney Injury (AKI) prediction in ICU settings using Word2Vec token embeddings and LSTM neural networks. The approach treats ICU measurements as a language, enabling distributional semantics to replace manual feature engineering.

## 📊 Key Results

- **Peak Performance**: 24h context with 12h prediction horizon achieved 0.761 AUPRC, a 4.2% improvement over baseline
- **Universal Gains**: Word2Vec provided significant AUPRC improvements across all tasks (+4.2% to +12.6%)
- **Time-Critical Windows**: Even for urgent 6-hour predictions, pretrained embeddings lifted AUPRC above 0.67
- **Clinical Interpretability**: t-SNE projections revealed clinically coherent clusters (cardiovascular, renal, electrolyte, etc.)

## 🏗️ Project Structure

```
organized_project/
├── preprocessing/                         # Data preprocessing pipeline
│   ├── 01_extract_patient_events.py      # Extract & filter patient events
│   ├── 02_validate_patient_data.py       # Validate data integrity
│   ├── 03_create_aki_episodes.py         # Create AKI episodes with exclusion criteria
│   ├── 04_truncate_patient_timeseries.py # Truncate to input length
│   ├── 05_create_train_test_splits.py    # Create train/test splits
│   ├── 06_generate_quantization_dict.py  # Generate quantization dictionary
│   ├── 07_quantize_patient_events.py     # Quantize and tokenize events
│   ├── 08_create_training_arrays.py      # Create training arrays
│   ├── 09_recreate_arrays_word2vec.py    # Recreate with Word2Vec vocabulary
│   └── run_preprocessing_pipeline.py      # Pipeline orchestrator
├── embeddings/
│   └── train_word2vec.py                 # Word2Vec training on medical events
├── training_evaluation/                   # Training and evaluation code
│   ├── models/                           # Neural network architectures
│   │   ├── embedders.py                  # Embedding layers
│   │   ├── lstms.py                      # LSTM implementations
│   │   └── models.py                     # Complete model architectures
│   └── utils/                            # Utility functions
├── config/
│   └── config.py                         # Centralized configuration
├── Research_Findings.ipynb               # Main notebook, please download and open in google colab!
└── requirements.txt                       # Project dependencies
```

## 🔬 Methodology

### Data Processing
1. **Patient Selection**: MIMIC-III v1.4 ICU stays with Metavision, age ≥18y, no transfers
2. **Pre-ICU Exclusions**:
   - AKI stage ≥1 before ICU admission
   - Severe chronic kidney dysfunction (eGFR < 15 mL/min/1.73 m²)
   - Prior dialysis/renal transplantation

### Tokenization Strategy
- **Continuous Variables**: Equal-Width binning (20 bins) based on training distribution
- **Categorical Variables**: Direct value mapping
- **Format**: `ITEMID_UOM:bin_index` or `ITEMID_UOM:value`
- **Vocabulary Size**: 18,494 unique tokens from 2,192 variables

### Word2Vec Training
- **Architecture**: Skip-gram with negative sampling
- **Parameters**: 128D vectors, window=10, min_count=5, epochs=10
- **Corpus**: ~100k-144k hourly "sentences" from patient stays

### LSTM Architecture
- **Input**: Sequence of hourly token embeddings
- **Processing**: Single-layer LSTM (hidden_dim=128)
- **Output**: Binary AKI prediction via sigmoid activation

## 🚀 Getting Started

### Prerequisites
```bash
pip install -r requirements.txt
```

### Configuration
Edit `config/config.py` to set your paths:
```python
DATA_ROOT_PATH = "/path/to/data"
MIMIC_CSV_PATH = "/path/to/mimic-iii"
MODELS_PATH = "/path/to/models"
RESULTS_PATH = "/path/to/results"
```

### Running the Pipeline

1. **Preprocess Data**:
```bash
cd preprocessing/
python run_preprocessing_pipeline.py
```

2. **Train Word2Vec**:
```bash
cd embeddings/
python train_word2vec.py
```

3. **Train LSTM Model**:
```bash
cd training_evaluation/
python train_model.py --input_hours 24 --horizon 12
```

## 📈 Performance Metrics

| Input→Horizon | Event Ratio | AUPRC (W2V) | AUPRC (Random) | AUPRC Gain | AUROC (W2V) | AUROC (Random) | AUROC Gain |
|--------------|-------------|-------------|----------------|------------|-------------|----------------|------------|
| 12h→6h       | 4.9%        | 0.671       | 0.596         | +12.6%     | 0.936       | 0.904         | +3.5%      |
| 12h→12h      | 7.0%        | 0.713       | 0.638         | +11.8%     | 0.941       | 0.918         | +2.5%      |
| 12h→24h      | 11.5%       | 0.630       | 0.571         | +10.3%     | 0.882       | 0.839         | +5.1%      |
| 24h→6h       | 7.5%        | 0.715       | 0.682         | +4.8%      | 0.924       | 0.900         | +2.7%      |
| 24h→12h      | 9.7%        | 0.761       | 0.731         | +4.2%      | 0.925       | 0.929         | -0.4%      |


## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- MIMIC-III database and its contributors
- OpenAI's GPT-4o for research support 
