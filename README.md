# SenasJanAI - Mahjong AI Pre-training Project

This project provides a complete pipeline to pre-train a Mahjong AI model based on the Transformer architecture. It parses Tenhou-style mahjong game logs (`.mjlog` files), converts them into a suitable format, and trains a model to predict human player actions.

This project has been refactored to run locally in a structured and easy-to-use manner, following the plan outlined in `事前学習の実行計画.md`.

---

## Getting Started

Follow these steps to set up the environment and run the training pipeline.

### 1. Environment Setup

It is highly recommended to use a Python virtual environment to avoid conflicts with other projects.

First, find your Python 3.10 (or compatible) executable. Then, create the virtual environment:

```shell
# Example path, replace with your actual Python path
C:\Users\YourUser\AppData\Local\Programs\Python\Python310\python.exe -m venv .venv
```

This will create a `.venv` directory in the project root.

### 2. Install Dependencies

Activate the virtual environment and install the required packages from `requirements.txt`.

```shell
# Activate the venv
.venv\Scripts\activate.bat

# Install packages
pip install -r requirements.txt
```

### 3. Data Preparation

Place your Tenhou-style mahjong log files (`.mjlog`) into the `data/raw_logs/` directory. A few sample files are already included.

---

## How to Run

The entire pipeline is managed by `main.py`. You can run the preprocessing and training steps together or separately.

Make sure your virtual environment is activated before running the commands.

### Run the Full Pipeline (Recommended)

This command will first process the log files and then immediately start training the model.

```shell
python main.py all
```

### Run Steps Individually

If you only need to perform a specific step, you can use the following commands.

**1. Preprocess Data Only:**

This reads the `.mjlog` files from `data/raw_logs/` and creates `train.npz`, `validation.npz`, and `test.npz` in the `data/processed/` directory.

```shell
python main.py preprocess
```

**2. Train Model Only:**

This loads the `.npz` files from `data/processed/` and runs the training process. The final model will be saved as `models/tenho_model.keras`.

```shell
python main.py pretrain
```

---

## About the Scripts

-   **`main.py`**: The main entry point for the application. It parses command-line arguments and orchestrates the preprocessing and training workflow.
-   **`preprocess.py`**: Contains the logic for parsing `.mjlog` files, vectorizing game states and actions according to the model's required input format, and saving the processed data as NumPy `.npz` archives.
-   **`pretrain.py`**: Defines the model training loop. It loads the processed data, builds the Transformer model, and runs the training using TensorFlow/Keras, complete with callbacks for saving the best model and stopping early.

## Note on Model Performance

The model's accuracy is highly dependent on the quantity and quality of the training data. The included sample logs are very few, so the resulting accuracy will be low. To train a powerful model, you should gather several thousand log files.