# Hybrid framework for energy-efficient ML inference with lightweight blockchain verification in edge computing environments.

## 📁 Project Structure
```
ecochain-ml/
├── README.md
├── requirements.txt
├── config/
│   ├── system_config.yaml
│   └── experiment_config.yaml
├── data/
│   ├── renewable_traces/
│   ├── workload_traces/
│   └── energy_profiles/
├── src/
│   ├── __init__.py
│   ├── scheduler/
│   │   ├── __init__.py
│   │   ├── energy_aware_scheduler.py
│   │   └── renewable_predictor.py
│   ├── blockchain/
│   │   ├── __init__.py
│   │   ├── verification_layer.py
│   │   └── pos_consensus.py
│   ├── inference/
│   │   ├── __init__.py
│   │   ├── model_executor.py
│   │   └── quantization.py
│   ├── monitoring/
│   │   ├── __init__.py
│   │   └── energy_monitor.py
│   └── simulator/
│       ├── __init__.py
│       ├── edge_node.py
│       └── network_simulator.py
├── experiments/
│   ├── baseline_comparison.py
│   ├── ablation_study.py
│   └── scalability_test.py
├── results/
│   ├── metrics/
│   ├── plots/
│   └── logs/
└── paper/
    ├── figures/
    ├── tables/
    └── manuscript.tex

```
## 🔹 Overview

EcoChain-ML integrates:

- Energy-Aware ML Inference Scheduler – Optimizes model execution based on energy availability and renewable sources.

- Lightweight Blockchain Verification – Verifies inference results with a low-overhead Proof-of-Stake protocol.

- Renewable-Aware Orchestration – Schedules tasks on nodes with high renewable energy while maintaining performance.

- Designed for research and simulation, this framework can run entirely on a standard PC with no special hardware.

## ⚡ Features

- Simulation of energy-aware ML inference on edge nodes.

- Renewable energy-aware scheduling for sustainable computation.

- Immutable blockchain verification of results and energy claims.

- Baseline comparisons for standard inference, energy optimization only, blockchain only, and integrated EcoChain-ML.

## 🛠️ Tech Stack

- ML Frameworks: PyTorch, TensorFlow, ONNX Runtime, Hugging Face Transformers

- Simulation: SimPy or custom Python simulator

- Blockchain: Web3.py, Ganache, Ethereum testnet (optional), custom PoS implementation

- Energy Tracking: CodeCarbon

- Visualization: Matplotlib, Seaborn, Jupyter Notebooks

## 🚀 Installation
Clone Project:
```
https://github.com/IamSadik/EcoChain-ML-Framework.git

```
Create a virtual environment:
```
python -m venv venv

```
Activate the virtual environment:
```
venv\Scripts\activate

```
Install Requirements:
```
pip install -r requirements.txt
```
## 🎯 Usage

Run Simulation:
```
python experiments/baseline_comparison.py
```

## 📊 Results & Visualization
- Visualize metrics and performance comparisons using Matplotlib and Seaborn.
- Generate plots for energy consumption and inference accuracy.
- Analyze experimental results using the scripts in the `experiments` directory.
- Review logs and metrics stored in the `results` directory for detailed insights.


## 📄 License

MIT License
