# SIR Model Experiments with SciML

![License](https://img.shields.io/github/license/agme2019/SciML-Experiments)
![Julia](https://img.shields.io/badge/Julia-1.8%2B-9558B2)
![SciML](https://img.shields.io/badge/SciML-Framework-blue)

## 📋 Overview

This repository contains a collection of experiments applying the Scientific Machine Learning (SciML) framework to the classic SIR (Susceptible-Infected-Recovered) epidemic model. The code demonstrates how differential equation solvers and machine learning techniques can be combined to simulate, analyze, and predict disease dynamics.

## 🦠 About SIR Models

The SIR model is a fundamental compartmental model in epidemiology that divides a population into three groups:

- **S**usceptible: Individuals who can contract the disease
- **I**nfected: Individuals who have the disease and can spread it
- **R**ecovered: Individuals who have recovered and developed immunity

The interactions between these compartments are governed by a system of ordinary differential equations (ODEs).

## 🔬 Experiments

This repository includes:

1. **Basic SIR Implementation** - Classic SIR model implementation using Julia's DifferentialEquations.jl
2. **Parameter Estimation** - Techniques for estimating transmission and recovery rates from data
3. **Stochastic Variants** - SIR models with random fluctuations to capture real-world uncertainty
4. **Neural-Enhanced Models** - Using neural networks to enhance traditional SIR models

## 🚀 Getting Started

### Prerequisites

- Julia 1.8 or higher
- SciML packages (instructions below)

### Installation

```bash
# Clone this repository
git clone https://github.com/agme2019/SciML-Experiments.git
cd SciML-Experiments/SIR\ codes/

# Install required packages
julia -e 'using Pkg; Pkg.activate("."); Pkg.instantiate()'
```

### Running the Examples

Each experiment is contained in its own Julia file. To run an experiment:

```bash
julia path/to/experiment.jl
```

## 📊 Example Results

The SIR model simulations produce time-series data showing the progression of an epidemic:

```
Time    Susceptible    Infected    Recovered
0.0     999            1           0
5.0     905            85          10
10.0    463            367         170
...
```

## 📚 Documentation

Each code file contains detailed comments explaining the mathematical models, implementation details, and experimental setup. For theoretical background on SIR models, see the references section.

## 🔗 References

- Kermack, W. O., & McKendrick, A. G. (1927). A contribution to the mathematical theory of epidemics.
- Rackauckas, C., & Nie, Q. (2017). DifferentialEquations.jl–A Performant and Feature-Rich Ecosystem for Solving Differential Equations in Julia.
- [SciML Documentation](https://sciml.ai/documentation/)

## 🤝 Contributing

Contributions are welcome! If you'd like to add your own SIR model experiments or improvements:

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-experiment`)
3. Commit your changes (`git commit -m 'Add some amazing experiment'`)
4. Push to the branch (`git push origin feature/amazing-experiment`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 📬 Contact

If you have any questions or want to discuss these experiments, please open an issue or contact the repository owner.

---

*This repository is part of ongoing research into Scientific Machine Learning applications in epidemiology.*
