import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import joblib

# Import the fiber design functions
from step_index_fiber import (design_step_index_fiber, plot_index_profile, 
                              generate_fiber_database, generate_index_profile,
                              calculate_numerical_aperture, estimate_mode_field_diameter,
                              calculate_v_parameter, estimate_bend_loss, cutoff_wavelength)


def prepare_data(df, input_features, output_features):
    """
    Prepare data for ML model training.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame containing fiber design data
    input_features : list
        List of column names to use as input features
    output_features : list
        List of column names to use as output features
    
    Returns:
    --------
    X_train, X_test, y_train, y_test, scaler_X, scaler_y
    """
    # Drop rows with NaN values
    df = df.dropna(subset=input_features + output_features)
    
    # Extract features and targets
    X = df[input_features].values
    y = df[output_features].values
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Scale features
    scaler_X = StandardScaler()
    X_train = scaler_X.fit_transform(X_train)
    X_test = scaler_X.transform(X_test)
    
    # Scale targets
    scaler_y = StandardScaler()
    y_train = scaler_y.fit_transform(y_train)
    y_test = scaler_y.transform(y_test)
    
    return X_train, X_test, y_train, y_test, scaler_X, scaler_y


def train_models(X_train, y_train):
    """
    Train multiple ML models for multi-output regression.
    
    Returns:
    --------
    dict of trained models
    """
    # Define models with multi-output support
    models = {
        'random_forest': RandomForestRegressor(n_estimators=100, random_state=42),
        # GradientBoostingRegressor doesn't natively support multi-output regression, so we wrap it
        'gradient_boosting': MultiOutputRegressor(
            GradientBoostingRegressor(n_estimators=100, random_state=42)
        ),
        'neural_network': MLPRegressor(hidden_layer_sizes=(100, 50), max_iter=1000, 
                                       activation='relu', solver='adam', random_state=42)
    }
    
    # Train all models
    for name, model in models.items():
        print(f"Training {name}...")
        model.fit(X_train, y_train)
    
    return models


def evaluate_models(models, X_test, y_test, scaler_y):
    """
    Evaluate ML models and return performance metrics.
    """
    results = {}
    
    for name, model in models.items():
        # Make predictions
        y_pred = model.predict(X_test)
        
        # Inverse transform to get actual values
        y_test_orig = scaler_y.inverse_transform(y_test)
        y_pred_orig = scaler_y.inverse_transform(y_pred)
        
        # Calculate metrics
        mse = mean_squared_error(y_test_orig, y_pred_orig)
        r2 = r2_score(y_test_orig, y_pred_orig)
        
        results[name] = {
            'mse': mse,
            'r2': r2,
            'predictions': y_pred_orig
        }
        
        print(f"{name} - MSE: {mse:.6f}, R²: {r2:.4f}")
    
    return results


def save_models(models, scaler_X, scaler_y, feature_names, target_names, filename='fiber_ml_models.pkl'):
    """Save the trained models and associated metadata."""
    model_data = {
        'models': models,
        'scaler_X': scaler_X,
        'scaler_y': scaler_y,
        'feature_names': feature_names,
        'target_names': target_names
    }
    
    joblib.dump(model_data, filename)
    print(f"Models saved to {filename}")


def load_models(filename='fiber_ml_models.pkl'):
    """Load trained models and associated metadata."""
    model_data = joblib.load(filename)
    return (model_data['models'], model_data['scaler_X'], model_data['scaler_y'],
            model_data['feature_names'], model_data['target_names'])


def predict_fiber_parameters(model, scaler_X, scaler_y, input_values, feature_names, target_names):
    """
    Predict fiber parameters from desired optical attributes.
    
    Parameters:
    -----------
    model : trained model
        The ML model to use for prediction
    scaler_X, scaler_y : sklearn.preprocessing.StandardScaler
        Scalers for input and output features
    input_values : list or array
        Values of input features
    feature_names : list
        Names of input features
    target_names : list
        Names of output features
    
    Returns:
    --------
    dict
        Predicted fiber parameters
    """
    # Convert input to DataFrame for easier handling
    input_df = pd.DataFrame([input_values], columns=feature_names)
    
    # Scale input
    X_scaled = scaler_X.transform(input_df)
    
    # Predict
    y_pred_scaled = model.predict(X_scaled)
    
    # Ensure prediction is 2D for inverse transform
    if len(y_pred_scaled.shape) == 1:
        y_pred_scaled = y_pred_scaled.reshape(1, -1)
        
    # Inverse transform
    y_pred = scaler_y.inverse_transform(y_pred_scaled)[0]
    
    # Create results dictionary
    results = {target_names[i]: y_pred[i] for i in range(len(target_names))}
    
    return results


def generate_training_database(size=5000, wavelength=1550e-9):
    """Generate a large database for training the ML model."""
    parameter_ranges = {
        'na_range': (0.08, 0.30),
        'mfd_range': (4e-6, 12e-6),  # 4-12 µm
        'bend_radius': 10e-3,  # 10 mm
        'bend_loss_range': (0.01, 0.5)  # 0.01-0.5 dB/m
    }
    
    print(f"Generating database with {size} samples...")
    db = generate_fiber_database(parameter_ranges, num_samples=size, wavelength=wavelength)
    
    # Save database
    db.to_csv("fiber_design_database.csv", index=False)
    print(f"Database with {len(db)} designs saved to fiber_design_database.csv")
    
    return db


def plot_predictions_vs_actual(results, y_test_orig, target_names):
    """Plot predicted vs actual values for the best model."""
    best_model = max(results.items(), key=lambda x: x[1]['r2'])[0]
    predictions = results[best_model]['predictions']
    
    fig, axes = plt.subplots(1, len(target_names), figsize=(15, 5))
    
    for i, feature in enumerate(target_names):
        ax = axes[i]
        ax.scatter(y_test_orig[:, i], predictions[:, i], alpha=0.5)
        ax.plot([y_test_orig[:, i].min(), y_test_orig[:, i].max()], 
                [y_test_orig[:, i].min(), y_test_orig[:, i].max()], 
                'k--', lw=2)
        ax.set_xlabel(f'Actual {feature}')
        ax.set_ylabel(f'Predicted {feature}')
        ax.set_title(f'{feature}')
        
    plt.tight_layout()
    plt.savefig("prediction_accuracy.png")
    plt.show()


def create_profile_from_prediction(predicted_params, wavelength=1550e-9, bend_radius=10e-3):
    """
    Create a refractive index profile from ML model predictions and calculate all optical properties.
    
    Parameters:
    -----------
    predicted_params : dict
        Dictionary with predicted parameters
    wavelength : float
        Operating wavelength in meters (default: 1550 nm)
    bend_radius : float
        Bend radius in meters for bend loss calculation (default: 10 mm)
    
    Returns:
    --------
    fiber_design : dict
        Fiber design dictionary compatible with plotting functions
    """
    # Extract parameters
    n1 = predicted_params.get('n1', None)
    n2 = predicted_params.get('n2', None)
    core_radius = predicted_params.get('core_radius', None)
    
    if n1 is None or n2 is None or core_radius is None:
        raise ValueError("Prediction must include n1, n2, and core_radius")
    
    # Convert core_radius from µm to m if needed
    if core_radius < 1e-3:  # already in meters
        pass
    else:  # in microns
        core_radius = core_radius * 1e-6
    
    # Create fiber design dictionary
    fiber_design = {
        'profile_type': 'step',
        'core_radius': core_radius,
        'n1': n1,
        'n2': n2,
        'delta_n': n1 - n2,
        'calculated_attributes': {}
    }
    
    # Calculate all optical properties
    NA = calculate_numerical_aperture(n1, n2)
    mfd = estimate_mode_field_diameter(wavelength, core_radius, NA)
    v_param = calculate_v_parameter(wavelength, core_radius, NA)
    cutoff_wl = cutoff_wavelength(core_radius, NA)
    bend_loss_val = estimate_bend_loss(wavelength, core_radius, n1, n2, bend_radius)
    
    # Store calculated properties
    fiber_design['calculated_attributes']['numerical_aperture'] = NA
    fiber_design['calculated_attributes']['mode_field_diameter'] = mfd
    fiber_design['calculated_attributes']['v_parameter'] = v_param
    fiber_design['calculated_attributes']['cutoff_wavelength'] = cutoff_wl
    fiber_design['calculated_attributes']['bend_loss'] = bend_loss_val
    fiber_design['calculated_attributes']['bend_radius'] = bend_radius
    
    return fiber_design


# Example of using the ML model in a production environment
def design_fiber_with_ml(target_na, target_mfd, target_bend_loss, model_path='fiber_ml_models.pkl'):
    """
    Design a fiber using a trained ML model
    
    Parameters:
    -----------
    target_na : float
        Target numerical aperture
    target_mfd : float
        Target mode field diameter in µm
    target_bend_loss : float
        Target bend loss in dB/m
    model_path : str
        Path to the saved ML models
    
    Returns:
    --------
    dict
        Fiber design dictionary
    """
    # Load models
    models, scaler_X, scaler_y, feature_names, target_names = load_models(model_path)
    
    # Select the best model
    best_model_name = "neural_network"  # This could be configured
    
    # Prepare input values
    input_values = [target_na, target_mfd, target_bend_loss]
    
    # Get prediction
    predicted_params = predict_fiber_parameters(
        models[best_model_name], scaler_X, scaler_y,
        input_values, feature_names, target_names
    )
    
    # Create fiber design
    fiber_design = create_profile_from_prediction(predicted_params)
    
    return fiber_design


# Batch processing example
def batch_design_fibers(design_requirements, output_file='batch_designs.csv'):
    """
    Process a batch of fiber design requirements
    
    Parameters:
    -----------
    design_requirements : list of dict
        List of dictionaries, each containing 'na', 'mfd', and 'bend_loss'
    output_file : str
        Path to save the results
    
    Returns:
    --------
    pandas.DataFrame
        DataFrame containing design results
    """
    # Load models
    models, scaler_X, scaler_y, feature_names, target_names = load_models()
    
    # Select model
    best_model_name = "neural_network"
    
    results = []
    
    for i, req in enumerate(design_requirements):
        try:
            # Prepare input
            input_values = [req['na'], req['mfd'], req['bend_loss']]
            
            # Get prediction
            predicted_params = predict_fiber_parameters(
                models[best_model_name], scaler_X, scaler_y,
                input_values, feature_names, target_names
            )
            
            # Create design
            fiber_design = create_profile_from_prediction(predicted_params)
            
            # Create result record
            result = {
                'design_id': i,
                'target_na': req['na'],
                'target_mfd': req['mfd'],
                'target_bend_loss': req['bend_loss'],
                'core_radius': fiber_design['core_radius'] * 1e6,  # µm
                'n1': fiber_design['n1'],
                'n2': fiber_design['n2'],
                'delta_n': fiber_design['delta_n'],
                'actual_na': fiber_design['calculated_attributes']['numerical_aperture'],
                'actual_mfd': fiber_design['calculated_attributes']['mode_field_diameter'] * 1e6,  # µm
                'v_parameter': fiber_design['calculated_attributes']['v_parameter'],
                'cutoff_wavelength': fiber_design['calculated_attributes']['cutoff_wavelength'] * 1e9  # nm
            }
            
            results.append(result)
            
        except Exception as e:
            print(f"Error processing design requirement {i}: {e}")
    
    # Create DataFrame
    results_df = pd.DataFrame(results)
    
    # Save to CSV
    if output_file:
        results_df.to_csv(output_file, index=False)
        print(f"Results saved to {output_file}")
    
    return results_df


if __name__ == "__main__":
    # Define input and output features
    # Inputs are the target optical properties
    input_features = ['target_na', 'target_mfd', 'target_bend_loss']
    
    # Outputs are the fiber parameters to achieve those properties
    output_features = ['core_radius', 'n1', 'n2']
    
    # Check if database exists, otherwise generate it
    try:
        print("Attempting to load existing database...")
        df = pd.read_csv("fiber_design_database.csv")
        print(f"Loaded database with {len(df)} samples")
    except FileNotFoundError:
        print("No existing database found. Generating new database...")
        df = generate_training_database(size=1000)  # Smaller size for demonstration
    
    # Prepare data
    X_train, X_test, y_train, y_test, scaler_X, scaler_y = prepare_data(
        df, input_features, output_features)
    
    print(f"Training data shape: X_train={X_train.shape}, y_train={y_train.shape}")
    
    # Train models
    models = train_models(X_train, y_train)
    
    # Evaluate models
    results = evaluate_models(models, X_test, y_test, scaler_y)
    
    # Get original test values for plotting
    y_test_orig = scaler_y.inverse_transform(y_test)
    
    # Plot predictions vs actual
    plot_predictions_vs_actual(results, y_test_orig, output_features)
    
    # Save models
    save_models(models, scaler_X, scaler_y, input_features, output_features)
    
    # Example prediction
    best_model_name = max(results.items(), key=lambda x: x[1]['r2'])[0]
    print(f"\nBest model: {best_model_name}")
    
    # Example: predict fiber parameters for desired optical properties
    desired_properties = {
        'target_na': 0.12,
        'target_mfd': 10.5,  # µm
        'target_bend_loss': 0.03  # dB/m
    }
    
    input_values = [desired_properties[feature] for feature in input_features]
    
    predicted_params = predict_fiber_parameters(
        models[best_model_name], scaler_X, scaler_y, 
        input_values, input_features, output_features)
    
    print("\nPredicted fiber parameters:")
    for param, value in predicted_params.items():
        if param == 'core_radius':
            print(f"  {param}: {value:.2f} µm")
        else:
            print(f"  {param}: {value:.6f}")
    
    # Create fiber design from prediction with all optical properties calculated
    fiber_design = create_profile_from_prediction(predicted_params)
    
    # Plot the predicted profile
    plot_index_profile(fiber_design, title="ML-Predicted Fiber Design")
    plt.savefig("predicted_fiber.png")
    
    print("\nPredicted optical properties:")
    print(f"  NA: {fiber_design['calculated_attributes']['numerical_aperture']:.4f}")
    print(f"  MFD: {fiber_design['calculated_attributes']['mode_field_diameter']*1e6:.2f} µm")
    print(f"  V-parameter: {fiber_design['calculated_attributes']['v_parameter']:.4f}")
    print(f"  Cutoff wavelength: {fiber_design['calculated_attributes']['cutoff_wavelength']*1e9:.2f} nm")
    print(f"  Bend loss at {fiber_design['calculated_attributes']['bend_radius']*1000:.1f} mm radius: "
          f"{fiber_design['calculated_attributes']['bend_loss']:.4f} dB/m")
    
    # Compare with direct optimization approach
    print("\nComparing with direct optimization approach...")
    target_attributes = {
        'numerical_aperture': desired_properties['target_na'],
        'mode_field_diameter': desired_properties['target_mfd'] * 1e-6,  # convert to meters
        'bend_loss': desired_properties['target_bend_loss'],
        'bend_radius': 10e-3  # 10 mm
    }
    
    from step_index_fiber import design_step_index_fiber
    
    direct_design = design_step_index_fiber(target_attributes)
    
    print("\nDirect optimization results:")
    print(f"  core_radius: {direct_design['core_radius']*1e6:.2f} µm")
    print(f"  n1: {direct_design['n1']:.6f}")
    print(f"  n2: {direct_design['n2']:.6f}")
    print(f"  NA: {direct_design['calculated_attributes']['numerical_aperture']:.4f}")
    print(f"  MFD: {direct_design['calculated_attributes']['mode_field_diameter']*1e6:.2f} µm")
    print(f"  V-parameter: {direct_design['calculated_attributes']['v_parameter']:.4f}")
    print(f"  Cutoff wavelength: {direct_design['calculated_attributes']['cutoff_wavelength']*1e9:.2f} nm")
    if 'bend_loss' in direct_design['calculated_attributes']:
        print(f"  Bend loss at {direct_design['calculated_attributes']['bend_radius']*1000:.1f} mm radius: "
            f"{direct_design['calculated_attributes']['bend_loss']:.4f} dB/m")
    
    # Plot the direct design profile
    plot_index_profile(direct_design, title="Direct Optimization Fiber Design")
    plt.savefig("direct_fiber.png")
    
    # Compare the two designs
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # ML prediction profile
    r_ml, n_ml = generate_index_profile(fiber_design)
    ax1.plot(r_ml * 1e6, n_ml, 'b-', label='ML Prediction')
    ax1.axvline(x=fiber_design['core_radius'] * 1e6, color='b', linestyle='--')
    
    # Direct optimization profile
    r_direct, n_direct = generate_index_profile(direct_design)
    ax1.plot(r_direct * 1e6, n_direct, 'r-', label='Direct Optimization')
    ax1.axvline(x=direct_design['core_radius'] * 1e6, color='r', linestyle='--')
    
    ax1.set_xlabel('Radius (µm)')
    ax1.set_ylabel('Refractive Index')
    ax1.set_title('Profile Comparison')
    ax1.grid(True)
    ax1.legend()
    
    # Bar chart comparison of key parameters
    params = ['core_radius', 'delta_n', 'NA', 'MFD']
    ml_values = [
        fiber_design['core_radius'] * 1e6,
        fiber_design['delta_n'],
        fiber_design['calculated_attributes']['numerical_aperture'],
        fiber_design['calculated_attributes']['mode_field_diameter'] * 1e6
    ]
    direct_values = [
        direct_design['core_radius'] * 1e6,
        direct_design['delta_n'],
        direct_design['calculated_attributes']['numerical_aperture'],
        direct_design['calculated_attributes']['mode_field_diameter'] * 1e6
    ]
    
    x = np.arange(len(params))
    width = 0.35
    
    ax2.bar(x - width/2, ml_values, width, label='ML Prediction')
    ax2.bar(x + width/2, direct_values, width, label='Direct Optimization')
    
    ax2.set_xticks(x)
    ax2.set_xticklabels(params)
    ax2.set_title('Parameter Comparison')
    ax2.legend()
    
    # Add percentages to show differences
    for i in range(len(params)):
        diff_pct = (ml_values[i] - direct_values[i]) / direct_values[i] * 100
        ax2.text(i, max(ml_values[i], direct_values[i]) + 0.05 * max(ml_values[i], direct_values[i]), 
                 f"{diff_pct:.1f}%", ha='center')
    
    plt.tight_layout()
    plt.savefig("comparison.png")
    
    print("\nResults comparison saved to comparison.png")
    
    # Example batch processing
    design_requirements = [
        {'na': 0.12, 'mfd': 9.5, 'bend_loss': 0.03},
        {'na': 0.14, 'mfd': 8.2, 'bend_loss': 0.05},
        {'na': 0.11, 'mfd': 10.0, 'bend_loss': 0.02},
        {'na': 0.18, 'mfd': 7.5, 'bend_loss': 0.08},
        {'na': 0.10, 'mfd': 11.2, 'bend_loss': 0.01}
    ]
    
    print("\nProcessing batch of design requirements...")
    results_df = batch_design_fibers(design_requirements)
    
    print("\nBatch processing results:")
    print(results_df)