<div align="center">

<br/>

# 📈 Synthetic Financial Data Generation
### *for Enhanced Financial Modeling*

<br/>

[![Python](https://img.shields.io/badge/Python-3.8%2B-3776AB?style=flat-square&logo=python&logoColor=white)](https://python.org)
[![Deep Learning](https://img.shields.io/badge/Deep%20Learning-TimeGAN%20%7C%20VAE-FF6F00?style=flat-square&logo=tensorflow&logoColor=white)](https://github.com/chrishounwnu/synthetic-financial-data)
[![Statistics](https://img.shields.io/badge/Statistics-ARIMA%20%7C%20GARCH-4CAF50?style=flat-square&logo=scipy&logoColor=white)](https://github.com/chrishounwnu/synthetic-financial-data)
[![License](https://img.shields.io/badge/License-Academic%20Use-9C27B0?style=flat-square)](https://github.com/chrishounwnu/synthetic-financial-data)
[![AIMS Rwanda](https://img.shields.io/badge/Institution-AIMS%20Rwanda-E91E63?style=flat-square)](https://aims.ac.rw)

<br/>

> *Generating statistically faithful synthetic financial time series using deep generative models —*
> *enabling privacy-preserving research, model development, and financial simulation.*

<br/>

</div>

---

## 🧭 Overview

This project explores the generation of **synthetic financial data** to address key challenges in quantitative finance: data privacy, dataset scarcity, and restricted access to real financial records.

We combine **classical statistical models** and **state-of-the-art deep generative architectures** to synthesize realistic financial time series, then rigorously evaluate their quality and utility across practical downstream tasks.

| Approach | Models Used |
|---|---|
| 📊 Statistical | ARIMA, GARCH |
| 🤖 Deep Learning | TimeGAN, Variational Autoencoder (VAE) |
| 📐 Evaluation | PCA Visualization, Maximum Mean Discrepancy (MMD) |
| 💼 Applications | Portfolio Optimization, Stress Testing, Backtesting |

---

## 🚀 Getting Started

### Prerequisites

Make sure you have **Python 3.8+** installed. A virtual environment is strongly recommended.

```bash
python -m venv venv
source venv/bin/activate        # On Windows: venv\Scripts\activate
```

### Installation & Run

**1. Clone the repository**
```bash
git clone https://github.com/chrishounwnu/synthetic-financial-data.git
cd synthetic-financial-data
```

**2. Install dependencies**
```bash
pip install -r requirements.txt
```

**3. Run the full pipeline**
```bash
python main.py
```

**4. Explore the notebooks**

Open any notebook inside the `notebooks/` folder for step-by-step walkthroughs of the modeling, training, and evaluation stages.

```
notebooks/
├── 01_data_preprocessing.ipynb
├── 02_statistical_models.ipynb
├── 03_timegan_training.ipynb
├── 04_vae_training.ipynb
└── 05_evaluation.ipynb
```

---

## 📊 Key Results

### 🔬 Time Series Fidelity

| Metric | Real vs TimeGAN | Real vs VAE |
|---|---|---|
| MMD Distance | **≈ 0.0044** ✅ | Higher ⚠️ |
| PCA Cluster Overlap | Strong overlap ✅ | Partial ⚠️ |

> **TimeGAN** produces synthetic sequences whose statistical distribution closely mirrors real financial data — confirmed by both visual (PCA) and quantitative (MMD) evaluation.

---

### 💼 Portfolio Optimization

- ✅ **TimeGAN** — Portfolios built on synthetic data were **realistic and comparable** to those built on real data.
- ⚠️ **VAE** — Portfolios exhibited **unrealistic allocation behavior** under stress conditions.

---

### 🌪️ Stress Testing

- ✅ **TimeGAN** — Synthetic portfolios remained **stable and coherent** under extreme market simulations.
- ❌ **VAE** — Portfolios exhibited **explosive, unrealistic growth**, attributable to random noise in the latent space.

---

### 📉 Backtesting

- ✅ **TimeGAN** — Return trajectories were **stable and plausible**, consistent with real data behavior.
- ❌ **VAE** — Highly **volatile and unreliable** for backtesting purposes.

---

### ✅ Conclusion

> Deep generative models — particularly **TimeGAN** — can produce **ethically usable, statistically valid** synthetic financial datasets suitable for research, model training, and financial simulation, without requiring access to sensitive real-world data.

---

## 👨‍🏫 Supervisor

This project was conducted under the academic supervision of:

| | |
|---|---|
| **Name** | [Prof. Dr. Yaé U. Gaba](https://github.com/gabayae/gabayae/tree/main) |
| **Institution** | [AI.Techniprenuers](https://github.com/ai-technipreneurs/ai-technipreneurs/tree/main) & [AIRINA Labs]()(https://airina-labs.github.io/AIRINA-Labs/) |
| **Email** | [yaeulrich.gaba@gmail.com](mailto:yaeulrich.gaba@gmail.com) |

---

## 📜 License

This project was developed for **academic and research purposes**.

Reuse, distribution, and modification of this work should include appropriate credit to the **author** and **supervisor**.

---

<div align="center">

<br/>

*Built with curiosity and rigor at* **AIMS Rwanda** 🌍

</div>



