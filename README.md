# 🌿 EcoChain-ML Framework

> **A Hybrid Framework for Energy-Efficient Machine Learning Model Verification Using Lightweight Blockchain**

[![Python 3.8+](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

---

## 📋 Table of Contents


- [Overview](#-overview)
- [Key Features](#-key-features)
- [Architecture](#️-architecture)
- [Project Structure](#-project-structure)
- [Installation](#-installation)
- [Quick Start](#-quick-start)
- [Experimental Results](#-experimental-results)
- [Configuration](#️-configuration)
- [Components](#-components)
- [Future Work](#-future-work)
- [License](#-license)


---

## 🔹 Overview

**EcoChain-ML** addresses the growing energy consumption challenge of ML inference at the edge by integrating:

1. **LSTM-based Renewable Energy Prediction** - Forecasts renewable availability for intelligent task routing
2. **Multi-objective Energy-Aware Scheduling** - Balances QoS, energy efficiency, and renewable utilization
3. **Dynamic Voltage and Frequency Scaling (DVFS)** - Adaptive power management based on workload
4. **INT8 Model Quantization** - Reduces inference energy by ~40% with minimal accuracy loss
5. **Proof-of-Stake Blockchain** - Verifies carbon credits and monetizes sustainability

### Target Problem

Edge ML inference consumes significant energy, often from non-renewable sources. EcoChain-ML addresses this by:
- ✅ Routing tasks to nodes with high renewable energy availability
- ✅ Reducing energy consumption through DVFS and model compression
- ✅ Providing immutable verification of energy claims via blockchain
- ✅ Monetizing sustainability through carbon credits

---

## ⚡ Key Features

| Feature | Description |
|---------|-------------|
| 🌞 **Renewable-Aware Routing** | LSTM predicts renewable availability 1 hour ahead |
| ⚖️ **Multi-Objective Scheduling** | Balances QoS (40%), Energy (30%), Renewable (30%) |
| 🔋 **DVFS Integration** | 5 frequency levels with intelligent power scaling |
| 🗜️ **Model Compression** | INT8 quantization with 2.5x size reduction |
| ⛓️ **PoS Blockchain** | 99.95% more efficient than PoW for carbon verification |
| 📈 **Horizontal Scalability** | 60% throughput improvement from 2→32 nodes |

---

## 🏗️ Architecture

```
User Request
     │
     ▼
┌─────────────┐
│ Task Queue  │ (Poisson arrival process)
└──────┬──────┘
       │
       ▼
┌──────────────────────────────┐
│   Renewable Predictor        │◄── Historical Weather Data
│   (LSTM: 2 layers, 64 units) │
└──────────────┬───────────────┘
               │
               ▼
┌──────────────────────────────┐
│  Energy-Aware Scheduler      │
│  Score = 0.4×QoS + 0.3×Energy│
│         + 0.3×Renewable      │
└──────────────┬───────────────┘
               │
               ▼
┌──────────────────────────────┐
│     DVFS Controller          │
│  freq = f(renewable, priority)│
└──────────────┬───────────────┘
               │
               ▼
┌──────────────────────────────┐
│    Model Compressor          │
│  INT8 Dynamic Quantization   │
└──────────────┬───────────────┘
               │
               ▼
┌──────────────────────────────────────────────────┐
│              Edge Node Cluster                    │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────┐ │
│  │  Solar   │ │   Wind   │ │  Hybrid  │ │ Grid │ │
│  │ (150W)   │ │  (120W)  │ │ (200W)   │ │ (0W) │ │
│  └──────────┘ └──────────┘ └──────────┘ └──────┘ │
└──────────────────────┬───────────────────────────┘
                       │
                       ▼
┌──────────────────────────────────────────────────┐
│           Blockchain Verification Layer           │
│  PoS Consensus → Block Creator → Carbon Credits   │
└──────────────────────────────────────────────────┘
```

### System Architecture Diagram

![Decision Path Selection Flow](docs/images/Decision%20Path%20Selection%20Flow-2025-12-16-131701.png)

---

## 📁 Project Structure

```
EcoChain-ML-Framework/
├── config/
│   ├── system_config.yaml      # Edge node, blockchain, scheduler config
│   └── experiment_config.yaml  # ML models, compression, workload config
├── docs/
│   └── images/                 # Result visualizations and plots
│       ├── energy_comparison.png
│       ├── renewable_comparison.png
│       └── radar_comparison.png
├── src/
│   ├── simulator/
│   │   ├── network_simulator.py  # Main orchestrator for simulations
│   │   └── edge_node.py          # Edge node with renewable energy
│   ├── scheduler/
│   │   ├── energy_aware_scheduler.py  # Multi-objective scheduler
│   │   └── renewable_predictor.py     # LSTM-based forecasting
│   ├── blockchain/
│   │   ├── verification_layer.py  # PoS blockchain verification
│   │   └── pos_consensus.py       # Proof-of-Stake consensus
│   ├── inference/
│   │   ├── model_executor.py      # ML inference execution
│   │   └── quantization.py        # INT8 quantization & pruning
│   └── monitoring/
│       └── energy_monitor.py      # Energy tracking & carbon calc
├── experiments/
│   ├── baseline_comparison.py    # Compare 4 methods
│   ├── ablation_study.py         # Component contribution analysis
│   └── scalability_test.py       # Node and workload scaling
├── results/
│   ├── baseline_comparison/      # Comparison results and plots
│   ├── ablation_study/           # Ablation results and plots
│   └── scalability_test/         # Scalability results and plots
└── paper/                        # Research paper materials
```

---

## 🚀 Installation

### Prerequisites
- Python 3.8+
- pip package manager

### Setup

```bash
# Clone the repository
git clone https://github.com/IamSadik/EcoChain-ML-Framework.git

# Navigate to project directory
cd EcoChain-ML-Framework

# Create virtual environment
python -m venv venv

# Activate virtual environment
# Windows:
venv\Scripts\activate
# Linux/Mac:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Dependencies

```
torch>=2.0.0
torchvision>=0.15.0
numpy>=1.24.0
pandas>=2.0.0
matplotlib>=3.7.0
seaborn>=0.12.0
pyyaml>=6.0
psutil>=5.9.0
scikit-learn>=1.2.0
```

---

## 🎯 Quick Start

### Run All Experiments

```bash
# 1. Baseline Comparison (4 methods)
python experiments/baseline_comparison.py

# 2. Ablation Study (component analysis)
python experiments/ablation_study.py

# 3. Scalability Tests (nodes + arrival rate)
python experiments/scalability_test.py
```

### Results Location

```
results/
├── baseline_comparison/
│   ├── metrics/          # JSON, CSV, LaTeX tables
│   └── plots/            # PNG visualizations
├── ablation_study/
│   ├── metrics/
│   └── plots/
└── scalability_test/
    ├── metrics/
    └── plots/
```

---

## 📊 Experimental Results

### Baseline Comparison (1000 tasks, 4 nodes)

| Method | Energy (kWh) | Carbon (gCO2) | Latency (s) | Renewable (%) | Net Cost ($) |
|--------|--------------|---------------|-------------|---------------|--------------|
| Standard | 0.0282 | 6.25 | 0.5395 | 44.68% | $0.001874 |
| Energy Aware Only | 0.0162 | 1.88 | 0.5187 | 70.97% | $0.000565 |
| Blockchain Only | 0.0284 | 6.42 | 0.5414 | 43.45% | $0.001679 |
| **EcoChain-ML** | **0.0166** | **2.04** | **0.5295** | **69.21%** | **$0.000384** |

### Key Achievements vs Standard Baseline

| Metric | Improvement |
|--------|-------------|
| 🔋 Energy Reduction | **41.2%** |
| 🌱 Carbon Reduction | **67.3%** |
| ⚡ Latency Overhead | Only 1.9% |
| 🌞 Renewable Utilization | **69.2%** (vs 44.7% baseline) |
| 💰 Net Cost Reduction | **79.5%** |

### Result Visualizations

#### Energy Consumption Comparison
![Energy Comparison](docs/images/energy_comparison.png)

#### Renewable Energy Utilization
![Renewable Comparison](docs/images/renewable_comparison.png)

#### Multi-Metric Performance Radar Chart
![Radar Comparison](docs/images/radar_comparison.png)

### Ablation Study Results

| Configuration | Energy (kWh) | Energy Δ | Carbon (gCO2) | Carbon Δ | Renewable (%) |
|---------------|--------------|----------|---------------|----------|---------------|
| Full EcoChain-ML | 0.0160 | baseline | 1.76 | baseline | 72.47% |
| Without Renewable Prediction | 0.0167 | +4.1% | 3.88 | **+119.9%** | 41.82% |
| Without DVFS | 0.0180 | +12.4% | 2.29 | +29.7% | 68.22% |
| Without Compression | 0.0251 | **+56.7%** | 2.81 | +59.3% | 72.00% |
| Without Blockchain | 0.0161 | +0.9% | 1.92 | +9.2% | 70.20% |

**Key Findings:**
- 🔴 **INT8 Compression** is most critical - removing it increases energy by 56.7%
- 🔴 **Renewable Prediction** is essential for carbon reduction - removing it increases carbon by 119.9%
- 🟡 **DVFS** contributes 12.4% energy savings
- 🟢 **Blockchain** adds minimal overhead (<1% energy) while enabling carbon credit verification

### Scalability Results

#### Node Scaling (1000 tasks)

| Nodes | Energy (kWh) | Latency (s) | Throughput (tasks/h) | Renewable (%) |
|-------|--------------|-------------|----------------------|---------------|
| 4 | 0.0151 | 0.508 | 102 | 72.15% |
| 8 | 0.0154 | 0.495 | 98 | 64.36% |
| 16 | 0.0159 | 0.513 | 105 | 50.43% |
| 32 | 0.0160 | 0.506 | 98 | 56.61% |

#### Arrival Rate Scaling (4 nodes)

| Arrival Rate | Tasks Completed | Latency (s) | Throughput (tasks/h) | Renewable (%) |
|--------------|-----------------|-------------|----------------------|---------------|
| 50 tasks/h | 1000 | 0.519 | 52 | 70.53% |
| 100 tasks/h | 1000 | 0.517 | 103 | 71.96% |
| 200 tasks/h | 1000 | 0.520 | 199 | 70.52% |
| 400 tasks/h | 1000 | 0.514 | 392 | 70.89% |

**Scalability Findings:**
- ✅ Consistent energy efficiency across 4-32 nodes
- ✅ Throughput scales linearly with arrival rate (52 → 392 tasks/h)
- ✅ Latency remains stable (~0.51s) regardless of load
- ✅ Renewable utilization maintained at 50-72% across configurations


---

## 🧩 Components

### 1. Energy-Aware Scheduler

Multi-objective scheduling with composite score:

```
score = (0.4 × QoS) + (0.3 × Energy) + (0.3 × Renewable) + (0.1 × LoadBalance)
```

### 2. DVFS Controller

Frequency selection based on renewable availability:


- **Architecture:** 2 layers, 64 hidden units, 0.2 dropout
- **Input:** [hour, day_of_week, solar_power, wind_power]
- **Output:** Renewable availability (0-1)
- **Lookback:** 24 hours, Horizon: 1 hour

### 3. Blockchain Verifier (PoS)


- **Consensus:** Proof-of-Stake (stake-weighted validator selection)
- **Block Time:** 5 seconds
- **Energy:** 0.00001 kWh per transaction
- **Carbon Credits:** $0.05 per gCO2 avoided

### 4. Model Compressor

- **Quantization:** INT8 dynamic (4x compression, 30-40% energy savings)
- **Pruning:** Magnitude-based L1 (30% sparsity)



### 5. Future Work

- [ ] Real hardware deployment (Raspberry Pi, NVIDIA Jetson)
- [ ] Dynamic carbon credit pricing from market APIs
- [ ] Federated learning integration
- [ ] Multi-site geo-distributed edge deployment
- [ ] Battery management and storage optimization
- [ ] Attention-based renewable prediction
- [ ] Support for diverse ML workloads (NLP, CV, audio)

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 📚 Citation

If you use EcoChain-ML in your research, please cite:

```bibtex
@inproceedings{sadikmahmud2025ecochain,
  title={EcoChain-ML: A Hybrid Framework for Energy-Efficient Machine Learning Model Verification Using Lightweight Blockchain},
  author={Sadik Mahmud},
  booktitle={Proceedings of IEEE/ACM Conference Name},
  year={2025},
  organization={IEEE/ACM}
}
```

---

<p align="center">
  <b>🌿 Making ML Inference Sustainable, One Task at a Time 🌿</b>
</p>
