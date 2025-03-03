# Optical Fiber Design with Machine Learning

## Project Overview

This repository provides a comprehensive solution for optical fiber design using both traditional optimization techniques and machine learning approaches. The project focuses on step-index fiber designs and includes tools for predicting critical fiber parameters based on desired optical properties.

## Key Features

- **Traditional Optimization**: Design step-index fibers using classical optimization methods
- **Machine Learning Models**: Train and use ML models (Random Forest, Gradient Boosting, Neural Networks) to predict optimal fiber parameters
- **Performance Comparison**: Compare ML-based designs with traditional optimization approaches
- **Batch Processing**: Process multiple design requirements efficiently
- **Visualization Tools**: Plot refractive index profiles and parameter comparisons

## Files

### `step_index_fiber.py`

This module contains the core physics-based functions for optical fiber design:

- Calculate important optical parameters (numerical aperture, V-parameter, mode field diameter)
- Estimate bend loss for fiber designs
- Generate refractive index profiles
- Design step-index fibers using direct optimization
- Generate training databases with varying parameters
- Visualize fiber designs

### `fiber_ml_model.py`

This module implements the machine learning approach to fiber design:

- Data preparation for ML model training
- Training multiple ML models (Random Forest, Gradient Boosting, Neural Networks)
- Model evaluation and comparison
- Saving and loading trained models
- Predicting fiber parameters based on desired optical properties
- Creating fiber designs from ML predictions
- Batch processing of design requirements

## Usage Examples

### Traditional Fiber Design

```python
from step_index_fiber import design_step_index_fiber, plot_index_profile

# Define target optical properties
target_attributes = {
    'numerical_aperture': 0.14,
    'mode_field_diameter': 9.2e-6,  # 9.2 µm
    'bend_radius': 10e-3,  # 10 mm bend radius
    'bend_loss': 0.05  # 0.05 dB/m max bend loss
}

# Design fiber
fiber = design_step_index_fiber(target_attributes)

# Visualize design
plot_index_profile(fiber, title="Optimized Fiber Design")
```

### ML-Based Fiber Design

```python
from fiber_ml_model import design_fiber_with_ml

# Design a fiber using the ML model
fiber_design = design_fiber_with_ml(
    target_na=0.12,
    target_mfd=10.5,  # µm
    target_bend_loss=0.03  # dB/m
)

# Access the predicted parameters
print(f"Core radius: {fiber_design['core_radius']*1e6:.2f} µm")
print(f"Core index: {fiber_design['n1']:.6f}")
print(f"Cladding index: {fiber_design['n2']:.6f}")
```

### Batch Processing

```python
from fiber_ml_model import batch_design_fibers

# Define multiple design requirements
design_requirements = [
    {'na': 0.12, 'mfd': 9.5, 'bend_loss': 0.03},
    {'na': 0.14, 'mfd': 8.2, 'bend_loss': 0.05},
    {'na': 0.11, 'mfd': 10.0, 'bend_loss': 0.02}
]

# Process all designs at once
results_df = batch_design_fibers(design_requirements, output_file="batch_results.csv")
```

## Performance

The ML approach shows excellent agreement with traditional optimization methods, as demonstrated in the comparison plots. The ML models can predict fiber parameters with high accuracy while offering significant speed advantages for batch processing.

## Getting Started

1. Clone the repository
2. Install dependencies: `pip install numpy scipy pandas matplotlib scikit-learn joblib`
3. Run the example in `fiber_ml_model.py` to generate training data and train models
4. Use the trained models for your own fiber design requirements

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Citation

If you use this code in your research, please cite:

[Citation](https://github.com/agme2019/SciML-Experiments/tree/main/Optical%20Fiber%20Sim)
