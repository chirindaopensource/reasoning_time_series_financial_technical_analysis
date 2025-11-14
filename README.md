# **`README.md`**

# **Verbal Technical Analysis: A Production-Grade Implementation**

<!-- PROJECT SHIELDS -->
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Python Version](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://www.python.org/)
[![arXiv](https://img.shields.io/badge/arXiv-2511.08616-b31b1b.svg)](https://arxiv.org/abs/2511.08616)
[![Conference](https://img.shields.io/badge/Conference-ICAIF%202025-9cf)](https://ai-finance.org/)
[![Year](https://img.shields.io/badge/Year-2025-purple)](https://github.com/chirindaopensource/reasoning_time_series_financial_technical_analysis)
[![Discipline](https://img.shields.io/badge/Discipline-Quantitative%20Finance-00529B)](https://github.com/chirindaopensource/reasoning_time_series_financial_technical_analysis)
[![Data Source](https://img.shields.io/badge/Data%20Source-StockNet-003299)](https://github.com/yumoxu/stocknet-dataset)
[![Core Method](https://img.shields.io/badge/Method-Reinforcement%20Learning-orange)](https://github.com/chirindaopensource/reasoning_time_series_financial_technical_analysis)
[![Analysis](https://img.shields.io/badge/Analysis-Time--Series%20Forecasting-red)](https://github.com/chirindaopensource/reasoning_time_series_financial_technical_analysis)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Type Checking: mypy](https://img.shields.io/badge/type%20checking-mypy-blue)](http://mypy-lang.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=flat&logo=PyTorch&logoColor=white)](https://pytorch.org/)
[![Hugging Face](https://img.shields.io/badge/🤗%20Hugging%20Face-Transformers-yellow)](https://huggingface.co/transformers)
[![PEFT](https://img.shields.io/badge/PEFT-LoRA-green)](https://github.com/huggingface/peft)
[![CVXPY](https://img.shields.io/badge/CVXPY-F4B841-blue)](https://www.cvxpy.org/)
[![Pandas](https://img.shields.io/badge/pandas-%23150458.svg?style=flat&logo=pandas&logoColor=white)](https://pandas.pydata.org/)
[![NumPy](https://img.shields.io/badge/numpy-%23013243.svg?style=flat&logo=numpy&logoColor=white)](https://numpy.org/)
[![Jupyter](https://img.shields.io/badge/Jupyter-%23F37626.svg?style=flat&logo=Jupyter&logoColor=white)](https://jupyter.org/)

**Repository:** `https://github.com/chirindaopensource/reasoning_time_series_financial_technical_analysis`

**Owner:** 2025 Craig Chirinda (Open Source Projects)

This repository contains an **independent**, professional-grade Python implementation of the research methodology from the 2025 paper entitled **"Reasoning on Time-Series for Financial Technical Analysis"** by:

*   Kelvin J.L. Koa
*   Jan Chen
*   Yunshan Ma
*   Huanhuan Zheng
*   Tat-Seng Chua

The project provides a complete, end-to-end computational framework for replicating the paper's findings. It delivers a modular, auditable, and extensible pipeline that executes the entire research workflow: from rigorous data validation and cleansing to multi-stage model training, baseline comparison, and final evaluation of both forecasting accuracy and portfolio utility.

## Table of Contents

- [Introduction](#introduction)
- [Theoretical Background](#theoretical-background)
- [Features](#features)
- [Methodology Implemented](#methodology-implemented)
- [Core Components (Notebook Structure)](#core-components-notebook-structure)
- [Key Callable: `run_vta_pipeline`](#key-callable-run_vta_pipeline)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Input Data Structure](#input-data-structure)
- [Usage](#usage)
- [Output Structure](#output-structure)
- [Project Structure](#project-structure)
- [Customization](#customization)
- [Contributing](#contributing)
- [Recommended Extensions](#recommended-extensions)
- [License](#license)
- [Citation](#citation)
- [Acknowledgments](#acknowledgments)

## Introduction

This project provides a Python implementation of the Verbal Technical Analysis (VTA) framework presented in Koa et al. (2025). The core of this repository is the iPython Notebook `reasoning_time_series_financial_technical_analysis_draft.ipynb`, which contains a comprehensive suite of functions to replicate the paper's findings. The pipeline is designed as a robust and scalable system for generating dual-output stock forecasts that are both numerically accurate and accompanied by a human-readable analytical narrative.

The paper's central contribution is a novel, multi-stage training methodology that teaches a Large Language Model (LLM) to perform financial technical analysis and fuses its reasoning with a dedicated time-series forecasting model. This codebase operationalizes the paper's experimental design, allowing users to:
-   Rigorously validate and manage the entire experimental configuration via a single `config.yaml` file.
-   Execute a multi-stage data preparation pipeline to cleanse, window, and annotate time-series data with a full suite of technical indicators.
-   Train a reasoning LLM (`πθ`) using a three-stage process: cold-start Reinforcement Learning (RL), Rejection Sampling with Supervised Fine-Tuning (SFT), and final performance-tuning RL.
-   Train a bespoke, dual-branch time-series backbone (`φ`) using a cross-modal alignment objective.
-   Train a conditional fusion model (`ψ`) that learns to guide the backbone's forecast using attributes derived from the LLM's reasoning.
-   Run a complete, end-to-end inference pipeline to generate the final dual output (forecast + narrative).
-   Train and evaluate strong baseline models (DLinear, TSMixer) under identical conditions for fair comparison.
-   Perform a comprehensive evaluation of all models on both statistical error metrics (MSE, MAE) and a realistic portfolio backtest using Markowitz optimization.
-   Conduct a full suite of sensitivity analyses to test the robustness of the results to key hyperparameter choices.

## Theoretical Background

The implemented methods are grounded in principles from deep learning, reinforcement learning, and modern portfolio theory.

**1. Time-GRPO for Reasoning:**
The reasoning LLM is trained using a novel objective called Time-Series Group Relative Policy Optimization (Time-GRPO), a variant of PPO. The policy `πθ` is optimized to maximize a reward signal derived from the accuracy of its generated forecast. The core reward is the inverse Mean Squared Error:
$$
r_{\mathrm{MSE}}(\theta) = \frac{1}{\lambda \cdot \lVert \hat{y}_\theta - y \rVert^2 + \epsilon}
$$
The policy is updated using a clipped surrogate objective with a KL penalty to maintain language fluency, based on the group-relative advantage \( A_i = (r_i - \mathrm{mean}(\{r_j\})) / (\mathrm{std}(\{r_j\}) + \epsilon) \).

**2. Cross-Modal Alignment:**
The time-series backbone `φ` is trained to align the numerical time-series domain with a latent language space. This is achieved via a cross-attention mechanism where the time-series embeddings `X_time` act as the query and a set of "language prototypes" `D` (derived from a base LLM's vocabulary via PCA) act as the key and value.
$$
X_{\text{text}} = \mathrm{Softmax}\left( \frac{(X_{\text{time}} W_Q) (D W_K)^T}{\sqrt{C}} \right) (D W_V)
$$
The model is trained with a dual-loss objective that encourages consistency between the temporal and aligned-textual branches.

**3. Classifier-Free Guidance for Fusion:**
The final forecast is a blend of the unconditional prediction from the backbone `ŷ_φ(X)` and a conditional prediction `ŷ_ψ(X, c)` that is guided by attributes `c` from the LLM's reasoning. The fusion model `ψ` is trained with random dropping of the conditioning vector `c`. At inference, two forward passes are performed (one with `c`, one without) and the results are blended:
$$
\hat{y} = \hat{y}_\phi(X) + s \cdot \big( \hat{y}_\psi(X, c) - \hat{y}_\phi(X) \big)
$$
where `s` is the guidance scale.

**4. Markowitz Portfolio Optimization:**
To assess financial utility, a daily-rebalanced portfolio is constructed by solving the Markowitz mean-variance optimization problem. The model's multi-step price forecasts are used to derive the expected returns `μ` and the covariance matrix `Σ` of the assets. The optimizer finds the weights `w` that maximize the risk-adjusted return:
$$
\max_{w} \; \mu_t^\top w - \frac{\gamma}{2} w^\top \Sigma_t w \quad \text{subject to} \quad w \succeq 0, \; \mathbf{1}^\top w = 1
$$

## Features

The provided iPython Notebook (`reasoning_time_series_financial_technical_analysis_draft.ipynb`) implements the full research pipeline, including:

-   **Modular, Multi-Task Architecture:** The entire pipeline is broken down into 15 distinct, modular tasks, each with its own orchestrator function.
-   **Configuration-Driven Design:** All hyperparameters and settings are managed in an external `config.yaml` file.
-   **High-Performance Feature Engineering:** Includes a complete, from-scratch implementation of 10 technical indicators using vectorized `numpy` and JIT-compiled `numba` for C-level speed.
-   **Resumable Pipeline:** The master orchestrator implements atomic artifact management, allowing the long-running pipeline to be stopped and resumed without re-running expensive completed stages.
-   **Production-Grade Training:** Implements best practices for RL and deep learning, including a robust PPO-style loop, SFT with the `transformers` Trainer, validation-based checkpointing, and gradient clipping.
-   **Rigorous Financial Backtesting:** Implements a daily rebalancing portfolio backtest with a professional-grade `cvxpy` optimizer and covariance matrix regularization.
-   **Complete Replication and Robustness:** A single top-level function call can execute the entire study, including a comprehensive suite of sensitivity analyses.
-   **Full Provenance:** The pipeline generates a unique run directory for each experiment, containing a detailed log file, a copy of the exact configuration used, and all generated artifacts for full reproducibility.

## Methodology Implemented

The core analytical steps directly implement the methodology from the paper:

1.  **Data Preparation (Tasks 1-5):** Ingests and validates all raw inputs, cleanses the market data, creates memory-efficient sliding windows, computes a full suite of technical indicators, and assembles the final prompts.
2.  **Reasoning Model Training (Tasks 6-8):** Executes the three-stage RL and SFT pipeline to train the reasoning model `πθ`.
3.  **Backbone Model Training (Task 9):** Trains the dual-branch forecasting backbone `φ` with the cross-modal alignment objective.
4.  **Fusion Model Training (Task 10):** Trains the conditional fusion model `ψ` using classifier-free guidance.
5.  **Inference (Task 11):** Runs the complete, three-model pipeline to generate the final dual outputs (forecast + narrative).
6.  **Baseline Evaluation (Task 12):** Trains and evaluates DLinear and TSMixer under identical conditions.
7.  **Final Evaluation (Task 13):** Computes all final error metrics and portfolio performance metrics for all models and generates comparison tables.
8.  **Robustness Analysis (Task 15):** Systematically re-runs the pipeline with varied hyperparameters to test for sensitivity.

## Core Components (Notebook Structure)

The `reasoning_time_series_financial_technical_analysis_draft.ipynb` notebook is structured as a logical pipeline with modular orchestrator functions for each of the 15 major tasks. All functions are self-contained, fully documented with type hints and docstrings, and designed for professional-grade execution.

## Key Callable: `run_vta_pipeline`

The project is designed around a single, top-level user-facing interface function:

-   **`run_vta_pipeline`:** This master orchestrator function, located in the final section of the notebook, runs the entire automated research pipeline from end-to-end. A single call to this function reproduces the entire computational portion of the project.

## Prerequisites

-   Python 3.9+
-   A CUDA-enabled GPU is highly recommended for all model training stages.
-   Core dependencies: `pandas`, `numpy`, `pyyaml`, `torch`, `transformers`, `peft`, `numba`, `scikit-learn`, `cvxpy`, `exchange_calendars`, `tqdm`.

## Installation

1.  **Clone the repository:**
    ```sh
    git clone https://github.com/chirindaopensource/reasoning_time_series_financial_technical_analysis.git
    cd reasoning_time_series_financial_technical_analysis
    ```

2.  **Create and activate a virtual environment (recommended):**
    ```sh
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

3.  **Install Python dependencies:**
    ```sh
    pip install -r requirements.txt
    ```

## Input Data Structure

The pipeline requires a primary `market_data_df` with a specific schema, which is rigorously validated. A synthetic data generator is included in the notebook for a self-contained demonstration.

-   **`market_data_df`**: A `pandas.DataFrame` with a `MultiIndex` of `['date', 'ticker']`.
    -   **Index:**
        -   `date`: `datetime64[ns]`
        -   `ticker`: `object` (string)
    -   **Columns:**
        -   `Open`, `High`, `Low`, `Close`, `Adj Close`: `float64`
        -   `Volume`: `int64` or `float64`

All other parameters are controlled by the `config.yaml` file.

## Usage

The `reasoning_time_series_financial_technical_analysis_draft.ipynb` notebook provides a complete, step-by-step guide. The primary workflow is to execute the final cell of the notebook, which demonstrates how to use the top-level `main` orchestrator:

```python
# Final cell of the notebook

# This block serves as the main entry point for the entire project.
if __name__ == '__main__':
    # 1. Define paths and parameters.
    CONFIG_PATH = "./config.yaml"
    DATA_PATH = "./synthetic_market_data.csv"
    
    # 2. Load configuration from the YAML file.
    with open(CONFIG_PATH, 'r') as f:
        config = yaml.safe_load(f)
    
    # 3. Define necessary data mappings.
    TICKER_TO_MARKET_MAP = {'SYNTH_A': 'US', 'SYNTH_B': 'US', ...}
    TICKER_TO_DATASET_MAP = {'SYNTH_A': 'StockNet', 'SYNTH_B': 'StockNet', ...}
    
    # 4. Execute the entire replication study in dry-run mode for a quick test.
    #    Set dry_run=False for a full run.
    final_results = main(
        market_data_path=DATA_PATH,
        config=config,
        ticker_to_market_map=TICKER_TO_MARKET_MAP,
        ticker_to_dataset_map=TICKER_TO_DATASET_MAP,
        dataset_name="StockNet",
        base_run_id="vta_replication",
        run_sensitivity=False,
        dry_run=True
    )
    
    # 5. Inspect final results.
    print("--- PIPELINE EXECUTION SUCCEEDED ---")
```

## Output Structure

The pipeline generates a structured `results/` directory. Each call to the master orchestrator creates a unique run directory:
-   **`results/<run_id>/`**: Contains all artifacts for a specific run.
    -   `artifacts/`: Pickled Python objects for each major task's output (e.g., `task_6_outputs.pkl`).
    -   `models/`: Saved model checkpoints for each training stage (e.g., `stage3_final_lora/`).
    -   `config.yaml`: An exact copy of the configuration used for this run.
    -   `pipeline.log`: A detailed log file for the run.
-   **`results/<base_run_id>_sensitivity_analysis_summary.csv`**: If sensitivity analysis is run, this master table summarizes the results.

## Project Structure

```
reasoning_time_series_financial_technical_analysis/
│
├── reasoning_time_series_financial_technical_analysis_draft.ipynb
├── config.yaml
├── requirements.txt
│
├── results/
│   ├── vta_replication_baseline_dry_run/
│   │   ├── artifacts/
│   │   ├── models/
│   │   ├── config.yaml
│   │   └── pipeline.log
│   └── ...
│
├── LICENSE
└── README.md
```

## Customization

The pipeline is highly customizable via the `config.yaml` file. Users can modify all study parameters, including model identifiers, learning rates, architectural details, and technical indicator settings, without altering the core Python code.

## Contributing

Contributions are welcome. Please fork the repository, create a feature branch, and submit a pull request with a clear description of your changes. Adherence to PEP 8, type hinting, and comprehensive docstrings is required.

## Recommended Extensions

An ablation study was proposed but not implemented. A key extension would be to implement this analysis to quantify the contribution of each component of the VTA framework. This would involve:
-   Creating a meta-orchestrator similar to the sensitivity analysis.
-   Programmatically creating modified configurations for each ablation scenario (e.g., setting `guidance_scale = 0`, using a simplified `c` vector).
-   Running the pipeline for each ablation and comparing the final performance metrics against the full VTA model.

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.

## Citation

If you use this code or the methodology in your research, please cite the original paper:

```bibtex
@article{koa2025reasoning,
  title={Reasoning on Time-Series for Financial Technical Analysis},
  author={Koa, Kelvin J.L. and Chen, Jan and Ma, Yunshan and Zheng, Huanhuan and Chua, Tat-Seng},
  journal={arXiv preprint arXiv:2511.08616},
  year={2025}
}
```

For the implementation itself, you may cite this repository:
```
Chirinda, C. (2025). A Production-Grade Implementation of "Reasoning on Time-Series for Financial Technical Analysis".
GitHub repository: https://github.com/chirindaopensource/reasoning_time_series_financial_technical_analysis
```

## Acknowledgments

-   Credit to **Kelvin J.L. Koa, Jan Chen, Yunshan Ma, Huanhuan Zheng, and Tat-Seng Chua** for the foundational research that forms the entire basis for this computational replication.
-   This project is built upon the exceptional tools provided by the open-source community. Sincere thanks to the developers of the scientific Python ecosystem, including **PyTorch, Hugging Face (Transformers, PEFT), Pandas, NumPy, Numba, CVXPY, and Scikit-learn**.

--

*This README was generated based on the structure and content of the `reasoning_time_series_financial_technical_analysis_draft.ipynb` notebook and follows best practices for research software documentation.*
