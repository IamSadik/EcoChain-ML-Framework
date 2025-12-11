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
│  │ (1000W)  │ │  (800W)  │ │ (1200W)  │ │ (0W) │ │
│  └──────────┘ └──────────┘ └──────────┘ └──────┘ │
└──────────────────────┬───────────────────────────┘
                       │
                       ▼
┌──────────────────────────────────────────────────┐
│           Blockchain Verification Layer           │
│  PoS Consensus → Block Creator → Carbon Credits   │
└──────────────────────────────────────────────────┘
```

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

### Baseline Comparison (1000 tasks, 24 hours, 4 nodes)

| Method | Energy (kWh) | Carbon (gCO2) | Latency (s) | Renewable (%) | Net Cost ($) |
|--------|--------------|---------------|-------------|---------------|--------------|
| Standard | 0.0281 | 6.79 | 0.5349 | 39.53 | $0.002036 |
| Energy Aware Only | 0.0073 | 0.01 | 0.3319 | 99.52 | $0.000004 |
| Blockchain Only | 0.0287 | 6.88 | 0.5485 | 40.12 | **-$0.228454** |
| **EcoChain-ML** | **0.0074** | **0.02** | **0.3316** | **99.40** | **-$0.146888** |

### Key Achievements

| Metric | Improvement vs Standard |
|--------|------------------------|
| 🔋 Energy Reduction | **73.66%** |
| 🌱 Carbon Reduction | **99.74%** |
| ⚡ Latency Improvement | **38%** |
| 🌞 Renewable Increase | **+59.87 pp** |
| 💰 Net Cost | **PROFIT** (negative = earning from carbon credits) |

### Result Visualizations

#### Energy Consumption Comparison
![Energy Comparison](docs/images/energy_comparison.png)

#### Renewable Energy Utilization
![Renewable Comparison](docs/images/renewable_comparison.png)

#### Multi-Metric Performance Radar Chart
![Radar Comparison](docs/images/radar_comparison.png)

### Ablation Study Results

| Component Removed | Energy Δ | Carbon Δ | Impact |
|-------------------|----------|----------|--------|
| Renewable Prediction | -29.68% | **+10,866%** | 🔴 Critical |
| Model Compression | **+200%** | +1,249% | 🔴 Critical |
| DVFS | +18% | +266% | 🟡 Important |
| Blockchain | +4% | +49% | 🟢 Low overhead |

### Scalability Results

| Nodes | Latency (s) | Throughput (tasks/h) | Renewable (%) |
|-------|-------------|----------------------|---------------|
| 2 | 0.335 | 103 | 99.23 |
| 8 | 0.251 | 134 | 99.32 |
| 32 | **0.131** | **165** | 99.59 |

**Findings:** 60% latency reduction, 60% throughput increase, consistent >99% renewable utilization.

---

## ⚙️ Configuration

### System Configuration (`config/system_config.yaml`)

```yaml
# Edge Nodes
edge_nodes:
  - id: node_1
    name: Solar Edge Node
    renewable_source: solar
    renewable_capacity_watts: 1000
    cpu_cores: 4
    max_frequency_ghz: 2.4
    min_frequency_ghz: 0.8

# Scheduler Weights
scheduler:
  qos_weight: 0.40
  energy_weight: 0.30
  renewable_weight: 0.30

# Blockchain
blockchain:
  consensus: pos
  block_time_seconds: 5
  energy_per_transaction_kwh: 0.00001

# Carbon Parameters
monitoring:
  carbon_intensity_gco2_per_kwh: 400
  electricity_price_per_kwh: 0.12
  carbon_credit_rate: 0.05
```

### Experiment Configuration (`config/experiment_config.yaml`)

```yaml
# Workload
workload:
  num_tasks: 1000
  duration_hours: 24
  arrival_rate_per_hour: 100

# Model Compression
compression:
  quantization:
    enabled: true
    method: dynamic
    dtype: int8
  pruning:
    enabled: true
    pruning_ratio: 0.3
```

---

## 🧩 Components

### 1. Energy-Aware Scheduler

Multi-objective scheduling with composite score:

```
score = (0.4 × QoS) + (0.3 × Energy) + (0.3 × Renewable) + (0.1 × LoadBalance)
```

### 2. DVFS Controller

Frequency selection based on renewable availability:

```
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

---

## 🔮 Future Work

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
