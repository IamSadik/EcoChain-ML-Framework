EcoChain-ML

Hybrid framework for energy-efficient machine learning inference with lightweight blockchain verification in edge computing environments.

Overview

EcoChain-ML is a research-focused framework that integrates:

Energy-Aware ML Inference Scheduler: Optimizes ML model execution based on energy profiles and renewable availability.

Lightweight Blockchain Verification: Verifies inference results and energy usage using a low-overhead Proof-of-Stake (PoS) protocol.

Renewable-Aware Orchestration: Routes tasks to nodes with high renewable energy availability while balancing latency and performance.

This project targets sustainable AI deployment and provides tools to simulate and measure energy-efficient inference in edge computing setups.

Features

Simulation of edge nodes with energy-aware scheduling.

Lightweight blockchain for verifiable ML inference results.

Integration with renewable energy profiles for workload optimization.

Baseline comparisons for standard inference vs. optimized approaches.

Tech Stack

ML & Inference: Python, PyTorch/TensorFlow, ONNX Runtime, Hugging Face Transformers

Blockchain: Web3.py, Ganache, Ethereum testnet (optional)

Simulation: SimPy or custom Python simulation

Energy Measurement: CodeCarbon

Visualization: Matplotlib / Seaborn

Installation
git clone https://github.com/yourusername/ecochain-ml.git
cd ecochain-ml
pip install -r requirements.txt

Usage

Configure simulation parameters in config.py.

Run the inference scheduler simulation:

python run_simulation.py


Run blockchain verification module (optional):

python run_blockchain.py


Analyze metrics using analysis/ scripts.

Project Structure
ecochain-ml/
├── config/          # Configuration files for simulation and blockchain
├── ecochain/        # Core framework: scheduler, orchestration, blockchain
├── data/            # Renewable energy traces, workload traces
├── experiments/     # Scripts for running experiments
├── analysis/        # Scripts for plotting and metrics evaluation
├── requirements.txt # Python dependencies
└── README.md

Contributing

This is primarily a research project. Contributions for testing, simulations, or extending energy-aware ML techniques are welcome.

License

MIT License
