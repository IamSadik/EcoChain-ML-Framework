# Hybrid framework for energy-efficient ML inference with lightweight blockchain verification in edge computing environments.

## 📁 Project Structure
```
/ecochain-ml/
├─ /simulator/             # SimPy or custom simulator
│   ├─ env.py              # Simulation environment setup
│   ├─ scheduler.py        # Energy-aware task scheduling
│   └─ energy_profiles.py  # Renewable energy profiles
├─ /edge_inference/        # Scripts to run ML inference on local PC
│   ├─ run_inference.py    # Main script for inference
│   ├─ measure_energy.py   # Track energy consumption
│   └─ models/             # Pre-trained or quantized ML models
├─ /blockchain/            # PoS prototype and verification
│   ├─ chain.py            # Blockchain core
│   └─ verifier.py         # Verify inference results & energy claims
├─ /experiments/           # Scripts to run experiments and analyze results
│   ├─ run_experiment.py
│   └─ analyze_results.py
├─ /notebooks/             # Jupyter notebooks for plots, metrics, analysis
├─ requirements.txt        # Python dependencies
└─ README.md
```
## 🔹 Overview

EcoChain-ML integrates:

Energy-Aware ML Inference Scheduler – Optimizes model execution based on energy availability and renewable sources.

Lightweight Blockchain Verification – Verifies inference results with a low-overhead Proof-of-Stake protocol.

Renewable-Aware Orchestration – Schedules tasks on nodes with high renewable energy while maintaining performance.

Designed for research and simulation, this framework can run entirely on a standard PC with no special hardware.

## ⚡ Features

Simulation of energy-aware ML inference on edge nodes.

Renewable energy-aware scheduling for sustainable computation.

Immutable blockchain verification of results and energy claims.

Baseline comparisons for standard inference, energy optimization only, blockchain only, and integrated EcoChain-ML.

## 🛠️ Tech Stack

ML Frameworks: PyTorch, TensorFlow, ONNX Runtime, Hugging Face Transformers

Simulation: SimPy or custom Python simulator

Blockchain: Web3.py, Ganache, Ethereum testnet (optional)

Energy Tracking: CodeCarbon

Visualization: Matplotlib, Seaborn, Jupyter Notebooks

## 🚀 Installation
git clone https://github.com/yourusername/ecochain-ml.git
cd ecochain-ml
pip install -r requirements.txt

## 🎯 Usage

Run Simulation:
```
python simulator/env.py
```

Run ML Inference:
```
python edge_inference/run_inference.py
```

Measure Energy Consumption:
```
python edge_inference/measure_energy.py
```

Blockchain Verification (optional):
```
python blockchain/verifier.py
```

Analyze Experimental Results:
```
python experiments/analyze_results.py
```
##📄 License

MIT License
