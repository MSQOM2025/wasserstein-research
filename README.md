# Robust Dependence Testing on Manifolds via Centered Sliced-Wasserstein Projections

This repository provides the implementation for the experiments presented in the manuscript titled "Robust Dependence Testing on Manifolds via Centered Sliced-Wasserstein Projections" (submitted for publication).

## 📄 Abstract
Traditional non-parametric independence tests suffer from a loss of statistical power in visual domains due to the pixel misalignment problem. We propose a geometry-aware framework (W-dCor) by re-interpreting images as weighted spatial measures. This implementation focuses on the Monte Carlo Sliced-Wasserstein (MCSW) estimation for manifold-valued data.

## ⚙️ Features
- W-dCor Statistic: Geometry-aware independence test implementation.
- Geometric Normalization: Implementation of the Centering Map (T) to induce translation invariance.
- Stochastic Tie-Breaking: Computational jittering for stable inverse CDF sampling.

## 🛠️ Installation
Install the necessary requirements via pip:

pip install numpy matplotlib seaborn scikit-learn scipy

🧪 Reproducing Results
To reproduce the behavioral analysis and results (e.g., Table 1 and Figure 1) using the UCI Digits dataset as described in the paper, run the following:
code

python main.py

This script evaluates the sensitivity of Euclidean dCor vs. the proposed W-dCor under:
Rigid Rotation
Random Translation (after Centering)
Geometric Expansion (Radial Dilation)
📊 Performance at a Glance (UCI Digits 8x8)
Metric	Rotation	Random Shift	Systematic Expansion
Euclidean dCor	0.98	0.45	0.69
W-dCor (Proposed)	0.80	0.63	0.87
📁 Repository Structure
main.py: Full behavioral study suite.
wdcor_lib.py: Core library containing Sliced-Wasserstein and Distance Correlation functions.
requirements.txt: List of dependencies.
✉️ Citation
Update available after publication.
For now, please refer to the submitted manuscript.
