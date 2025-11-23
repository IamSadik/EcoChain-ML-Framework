# Hybrid framework for energy-efficient ML inference with lightweight blockchain verification in edge computing environments.

## 📁 Project Structure
```
/ecochain-ml/
├─ /simulator/             # SimPy or custom simulator
│   ├─ __init__.py
│   ├─ env.py              # Simulation environment setup
│   ├─ scheduler.py        # Energy-aware task scheduling
│   ├─ energy_profiles.py  # Renewable energy profiles
│   └─ run_simulation.py   # Script to run the simulation
├─ /edge_inference/        # Scripts to run ML inference on local PC
│   ├─ __init__.py
│   ├─ run_inference.py    # Main script for inference
│   ├─ measure_energy.py   # Track energy consumption
│   └─ models/             # Pre-trained or quantized ML models
├─ /blockchain/            # PoS prototype and verification
│   ├─ __init__.py
│   ├─ chain.py            # Blockchain core
│   └─ verifier.py         # Verify inference results & energy claims
├─ /experiments/           # Scripts to run experiments and analyze results
│   ├─ __init__.py
│   ├─ run_experiment.py
│   └─ analyze_results.py
├─ /notebooks/             # Jupyter notebooks for plots, metrics, analysis
│   └─ __init__.py
├─ requirements.txt        # Python dependencies
└─ README.md

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

- Blockchain: Web3.py, Ganache, Ethereum testnet (optional)

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
python -m simulator.run_simulation
```

Run ML Inference:
```
python -m edge_inference.run_inference
```

Measure Energy Consumption:
```
python -m edge_inference.measure_energy
```

Blockchain Verification (optional):
```
python -m blockchain.verifier
```

Analyze Experimental Results:
```
python -m experiments.analyze_results
```
## 📄 License

MIT License
