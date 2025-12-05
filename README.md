# Multi-Objective Adversarial Perturbations for SLAM Systems using NSGA-III

[![CI](https://github.com/francescacraievich/mola-adversarial-nsga3/actions/workflows/ci.yml/badge.svg)](https://github.com/francescacraievich/mola-adversarial-nsga3/actions/workflows/ci.yml)
[![Documentation](https://github.com/francescacraievich/mola-adversarial-nsga3/actions/workflows/docs.yml/badge.svg)](https://github.com/francescacraievich/mola-adversarial-nsga3/actions/workflows/docs.yml)
[![Docs](https://img.shields.io/badge/docs-gh--pages-blue)](https://francescacraievich.github.io/mola-adversarial-nsga3/)
[![codecov](https://codecov.io/github/francescacraievich/mola-adversarial-nsga3/graph/badge.svg?token=BQX8LWJMSJ)](https://codecov.io/github/francescacraievich/mola-adversarial-nsga3)

Evolutionary multi-objective optimization of adversarial perturbations on LiDAR point clouds to evaluate the robustness of SLAM systems.

## Overview

This project uses **NSGA-III** (Non-dominated Sorting Genetic Algorithm III) to evolve adversarial perturbations on LiDAR point clouds that compromise SLAM systems. The algorithm optimizes a trade-off between:

- **Attack Effectiveness**: Maximize localization error in the SLAM system
- **Imperceptibility**: Minimize the magnitude of perturbations

## Setup

The system is integrated with **MOLA SLAM** running in **Isaac Sim** with recorded point clouds from a rover, enabling automatic fitness evaluation by comparing SLAM trajectories against ground truth from the simulator.

## Features

- Multi-objective optimization using NSGA-III
- Automated fitness evaluation against ground truth
- Pareto-optimal perturbation generation
- Comparison with baseline approaches (random perturbations, grid search)

## Installation

```bash
pip install -r requirements.txt
```

## Usage

Coming soon...

## Project Structure

```
mola-adversarial-nsga3/
├── src/              # Source code
├── mola/             # MOLA SLAM integration
├── docs/             # Documentation
└── tests/            # Test suite
```

## Deliverables

- Python implementation (perturbations + NSGA-III + evaluation)
- Comparison with baseline approaches
- Documentation and results analysis

## License

MIT

## Author

Francesca Craievich
