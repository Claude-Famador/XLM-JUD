## Repository Structure

```text
├── augmentation/       # Back-translation and SMOTE augmentation modules
├── data/               # Data collection, cleaning, and dataset scripts
├── evaluation/         # Evaluation metrics and statistical significance testing
├── models/             # XLM-RoBERTa model training and baseline classifiers
├── outputs/            # Logs, figures, and model checkpoints (ignored in git)
├── config.py           # Central configuration for paths and hyperparameters
├── main.py             # Main entry point to run the full pipeline
├── requirements.txt    # Python package dependencies
└── README.md           # Project documentation
```

## Setup & Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/<YOUR_USERNAME>/<YOUR_REPO_NAME>.git
   cd <YOUR_REPO_NAME>
   ```

2. **Create and activate a virtual environment**:
   ```bash
   python -m venv venv
   # On Windows:
   venv\Scripts\activate
   # On macOS/Linux:
   source venv/bin/activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

## Running the Pipeline

Run the full pipeline using `main.py`:
```bash
python main.py
```
