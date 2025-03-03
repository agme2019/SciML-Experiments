import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import pandas as pd


def calculate_numerical_aperture(n1, n2):
    """Calculate numerical aperture from core and cladding indices."""
    return np.sqrt(n1**2 - n2**2)


def calculate_v_parameter(wavelength, core_radius, numerical_aperture):
    """Calculate V-parameter (normalized frequency)."""
    return (2 * np.pi / wavelength) * core_radius * numerical_aperture


def estimate_mode_field_diameter(wavelength, core_radius, numerical_aperture):
    """
    Estimate mode field diameter using Marcuse formula for step-index fibers.
    Returns MFD in the same units as wavelength and core_radius.
    """
    V = calculate_v_parameter(wavelength, core_radius, numerical_aperture)
    
    # Marcuse formula is valid for V > 1.2
    if V > 1.2:
        w_over_a = 0.65 + 1.619/V**(3/2) + 2.879/V**6
        mfd = 2 * core_radius * w_over_a
    else:
        # For low V values, use alternative approximation
        mfd = 2 * core_radius * (1.1 + 0.996/V)
    
    return mfd


def estimate_bend_loss(wavelength, core_radius, n1, n2, bend_radius):
    """
    Estimate bend loss for step-index fiber using simplified model.
    Returns loss in dB/m.
    """
    NA = calculate_numerical_aperture(n1, n2)
    V = calculate_v_parameter(wavelength, core_radius, NA)
    
    # Simplified bend loss formula
    # This is an approximation - detailed bend loss models are more complex
    k0 = 2 * np.pi / wavelength
    delta = (n1 - n2) / n1
    
    # Calculate normalized transverse attenuation parameter
    gamma = np.sqrt((n1 * k0)**2 - (n1 * k0 * np.cos(np.arcsin(NA/n1)))**2)
    
    # Simple approximation of bend loss coefficient (dB/m)
    # This is a simplified formula - actual bend loss is more complex
    if V > 2.405:  # Single-mode cutoff
        factor = np.exp(-4/3 * gamma * core_radius**3 / bend_radius)
        bend_loss_coefficient = 10 * np.log10(np.e) * factor * (V**2 / (delta * core_radius**3))
    else:
        # For V below cutoff, bend loss increases dramatically
        factor = np.exp(-4/3 * gamma * core_radius**3 / bend_radius)
        bend_loss_coefficient = 10 * np.log10(np.e) * factor * (4 / (delta * core_radius**3))
    
    return bend_loss_coefficient


def cutoff_wavelength(core_radius, NA):
    """Calculate the cutoff wavelength for single-mode operation."""
    # V = 2.405 at cutoff
    return (2 * np.pi * core_radius * NA) / 2.405


def design_step_index_fiber(target_attributes, wavelength=1550e-9):
    """
    Design a step-index fiber based on target optical attributes.
    
    Parameters:
    -----------
    target_attributes : dict
        Dictionary containing target values for optical attributes:
        - 'numerical_aperture': Target numerical aperture
        - 'mode_field_diameter': Target mode field diameter (in m)
        - 'bend_loss': Target bend loss at specified bend radius (dB/m)
        - 'bend_radius': Bend radius for bend loss calculation (m)
    
    wavelength : float
        Operating wavelength in meters (default: 1550 nm)
    
    Returns:
    --------
    dict
        Optimized fiber parameters:
        - 'core_radius': Core radius (m)
        - 'n1': Core refractive index
        - 'n2': Cladding refractive index
        - 'profile_type': Type of index profile ('step')
        - 'calculated_attributes': dict of calculated optical attributes
    """
    # Set reference cladding index for silica
    n2_ref = 1.444  # Typical silica cladding at 1550 nm
    
    def objective_function(params):
        core_radius, delta_n = params
        
        # Calculate refractive indices
        n1 = n2_ref + delta_n
        n2 = n2_ref
        
        # Calculate optical attributes
        NA = calculate_numerical_aperture(n1, n2)
        mfd = estimate_mode_field_diameter(wavelength, core_radius, NA)
        
        if 'bend_radius' in target_attributes and 'bend_loss' in target_attributes:
            bend_loss = estimate_bend_loss(wavelength, core_radius, n1, n2, 
                                          target_attributes['bend_radius'])
        else:
            bend_loss = 0
        
        # Calculate error terms (weighted squared differences)
        errors = []
        
        if 'numerical_aperture' in target_attributes:
            na_error = ((NA - target_attributes['numerical_aperture']) / 
                        target_attributes['numerical_aperture'])**2
            errors.append(5 * na_error)  # Higher weight for NA
            
        if 'mode_field_diameter' in target_attributes:
            mfd_error = ((mfd - target_attributes['mode_field_diameter']) / 
                         target_attributes['mode_field_diameter'])**2
            errors.append(3 * mfd_error)
            
        if 'bend_loss' in target_attributes and 'bend_radius' in target_attributes:
            # For bend loss, we want to be below target, so penalize only if higher
            if bend_loss > target_attributes['bend_loss']:
                bl_error = ((bend_loss - target_attributes['bend_loss']) / 
                           target_attributes['bend_loss'])**2
                errors.append(2 * bl_error)
        
        # Single-mode condition check - add penalty if not single-mode
        v_param = calculate_v_parameter(wavelength, core_radius, NA)
        if v_param > 2.405:  # V > 2.405 means multimode
            errors.append(10 * (v_param - 2.405)**2)  # Strong penalty
            
        # Return sum of weighted errors
        return sum(errors)
    
    # Initial guess for optimization
    if 'numerical_aperture' in target_attributes:
        initial_delta_n = target_attributes['numerical_aperture']**2 / (2 * n2_ref)
    else:
        initial_delta_n = 0.005  # Default delta_n guess
    
    if 'mode_field_diameter' in target_attributes:
        initial_core_radius = target_attributes['mode_field_diameter'] / 2 / 1.1
    else:
        initial_core_radius = 4.1e-6  # Default core radius guess
    
    initial_guess = [initial_core_radius, initial_delta_n]
    
    # Set bounds for parameters
    bounds = [(2e-6, 10e-6),       # Core radius between 2 and 10 µm
              (0.001, 0.03)]       # Delta n between 0.001 and 0.03
    
    # Run optimization
    result = minimize(objective_function, initial_guess, bounds=bounds, method='L-BFGS-B')
    
    # Extract optimized parameters
    core_radius, delta_n = result.x
    n1 = n2_ref + delta_n
    n2 = n2_ref
    
    # Calculate resulting optical attributes
    NA = calculate_numerical_aperture(n1, n2)
    mfd = estimate_mode_field_diameter(wavelength, core_radius, NA)
    v_param = calculate_v_parameter(wavelength, core_radius, NA)
    
    if 'bend_radius' in target_attributes:
        bend_loss = estimate_bend_loss(wavelength, core_radius, n1, n2, 
                                     target_attributes['bend_radius'])
    else:
        bend_loss = None
    
    cutoff_wl = cutoff_wavelength(core_radius, NA)
    
    # Store profile and calculated attributes
    fiber_design = {
        'profile_type': 'step',
        'core_radius': core_radius,
        'n1': n1,
        'n2': n2,
        'delta_n': delta_n,
        'calculated_attributes': {
            'numerical_aperture': NA,
            'mode_field_diameter': mfd,
            'v_parameter': v_param,
            'cutoff_wavelength': cutoff_wl
        }
    }
    
    if bend_loss is not None:
        fiber_design['calculated_attributes']['bend_loss'] = bend_loss
        fiber_design['calculated_attributes']['bend_radius'] = target_attributes['bend_radius']
    
    return fiber_design


def generate_index_profile(fiber_design, num_points=1000):
    """
    Generate refractive index profile data points for plotting.
    
    Parameters:
    -----------
    fiber_design : dict
        Fiber design parameters from design_step_index_fiber
    num_points : int
        Number of points to generate for plotting
    
    Returns:
    --------
    r : ndarray
        Radial position array
    n : ndarray
        Refractive index array
    """
    # Extract parameters
    core_radius = fiber_design['core_radius']
    n1 = fiber_design['n1']
    n2 = fiber_design['n2']
    
    # Generate radial positions (finer sampling near the core-cladding boundary)
    r = np.linspace(0, 3 * core_radius, num_points)
    
    # Generate index profile
    n = np.where(r <= core_radius, n1, n2)
    
    return r, n


def plot_index_profile(fiber_design, title=None):
    """Plot the refractive index profile."""
    r, n = generate_index_profile(fiber_design)
    
    plt.figure(figsize=(10, 6))
    plt.plot(r * 1e6, n)  # Convert to microns for better readability
    plt.axvline(x=fiber_design['core_radius'] * 1e6, color='r', linestyle='--', 
                label=f'Core radius: {fiber_design["core_radius"]*1e6:.2f} µm')
    
    plt.xlabel('Radius (µm)')
    plt.ylabel('Refractive Index')
    
    if title:
        plt.title(title)
    else:
        plt.title('Step-Index Fiber Profile')
    
    plt.grid(True)
    plt.legend()
    
    # Add fiber parameters as text
    props = {
        'Core radius': f"{fiber_design['core_radius']*1e6:.2f} µm",
        'Core index (n₁)': f"{fiber_design['n1']:.6f}",
        'Cladding index (n₂)': f"{fiber_design['n2']:.6f}",
        'Index difference (Δn)': f"{fiber_design['n1']-fiber_design['n2']:.6f}",
        'NA': f"{fiber_design['calculated_attributes']['numerical_aperture']:.4f}",
        'MFD': f"{fiber_design['calculated_attributes']['mode_field_diameter']*1e6:.2f} µm",
        'V-parameter': f"{fiber_design['calculated_attributes']['v_parameter']:.2f}"
    }
    
    info_text = '\n'.join([f"{k}: {v}" for k, v in props.items()])
    plt.figtext(0.02, 0.02, info_text, fontsize=10)
    
    plt.tight_layout()
    return plt.gcf()


def generate_fiber_database(parameter_ranges, num_samples=1000, wavelength=1550e-9):
    """
    Generate a database of fiber designs with varying parameters.
    
    Parameters:
    -----------
    parameter_ranges : dict
        Dictionary containing parameter ranges:
        - 'na_range': (min_na, max_na)
        - 'mfd_range': (min_mfd, max_mfd) in meters
        - 'bend_radius': bend radius in meters
        - 'bend_loss_range': (min_loss, max_loss) in dB/m
    
    num_samples : int
        Number of samples to generate
    
    wavelength : float
        Operating wavelength in meters
    
    Returns:
    --------
    pandas.DataFrame
        Database of fiber designs with input parameters and resulting attributes
    """
    database = []
    
    for _ in range(num_samples):
        # Randomly sample target parameters from ranges
        target_attributes = {}
        
        if 'na_range' in parameter_ranges:
            min_na, max_na = parameter_ranges['na_range']
            target_attributes['numerical_aperture'] = np.random.uniform(min_na, max_na)
        
        if 'mfd_range' in parameter_ranges:
            min_mfd, max_mfd = parameter_ranges['mfd_range']
            target_attributes['mode_field_diameter'] = np.random.uniform(min_mfd, max_mfd)
        
        if 'bend_radius' in parameter_ranges and 'bend_loss_range' in parameter_ranges:
            target_attributes['bend_radius'] = parameter_ranges['bend_radius']
            min_loss, max_loss = parameter_ranges['bend_loss_range']
            target_attributes['bend_loss'] = np.random.uniform(min_loss, max_loss)
        
        # Design fiber
        try:
            fiber_design = design_step_index_fiber(target_attributes, wavelength)
            
            # Prepare row for database
            row = {
                'target_na': target_attributes.get('numerical_aperture', None),
                'target_mfd': target_attributes.get('mode_field_diameter', None) * 1e6,  # µm
                'target_bend_loss': target_attributes.get('bend_loss', None),
                'core_radius': fiber_design['core_radius'] * 1e6,  # µm
                'n1': fiber_design['n1'],
                'n2': fiber_design['n2'],
                'delta_n': fiber_design['n1'] - fiber_design['n2'],
                'actual_na': fiber_design['calculated_attributes']['numerical_aperture'],
                'actual_mfd': fiber_design['calculated_attributes']['mode_field_diameter'] * 1e6,  # µm
                'v_parameter': fiber_design['calculated_attributes']['v_parameter'],
                'cutoff_wavelength': fiber_design['calculated_attributes']['cutoff_wavelength'] * 1e9  # nm
            }
            
            if 'bend_loss' in fiber_design['calculated_attributes']:
                row['actual_bend_loss'] = fiber_design['calculated_attributes']['bend_loss']
            
            database.append(row)
            
            if len(database) % 100 == 0:
                print(f"Generated {len(database)} designs")
                
        except Exception as e:
            print(f"Error generating fiber design: {e}")
            continue
    
    return pd.DataFrame(database)


# Example usage
if __name__ == "__main__":
    # Example 1: Design a fiber with specific NA and MFD
    target_attributes = {
        'numerical_aperture': 0.14,
        'mode_field_diameter': 9.2e-6,  # 9.2 µm
        'bend_radius': 10e-3,  # 10 mm bend radius
        'bend_loss': 0.05  # 0.05 dB/m max bend loss
    }
    
    fiber = design_step_index_fiber(target_attributes)
    plot_index_profile(fiber, title="Optimized Fiber Design")
    plt.savefig("optimized_fiber.png")
    
    print(f"Core radius: {fiber['core_radius']*1e6:.2f} µm")
    print(f"Core index (n₁): {fiber['n1']:.6f}")
    print(f"Cladding index (n₂): {fiber['n2']:.6f}")
    print(f"NA: {fiber['calculated_attributes']['numerical_aperture']:.4f}")
    print(f"MFD: {fiber['calculated_attributes']['mode_field_diameter']*1e6:.2f} µm")
    print(f"V-parameter: {fiber['calculated_attributes']['v_parameter']:.4f}")
    print(f"Cutoff wavelength: {fiber['calculated_attributes']['cutoff_wavelength']*1e9:.2f} nm")
    
    if 'bend_loss' in fiber['calculated_attributes']:
        print(f"Bend loss at {fiber['calculated_attributes']['bend_radius']*1000:.1f} mm radius: "
              f"{fiber['calculated_attributes']['bend_loss']:.4f} dB/m")
    
    # Example 2: Generate a small database of fiber designs
    parameter_ranges = {
        'na_range': (0.10, 0.20),
        'mfd_range': (8e-6, 11e-6),  # 8-11 µm
        'bend_radius': 15e-3,  # 15 mm
        'bend_loss_range': (0.01, 0.1)  # 0.01-0.1 dB/m
    }
    
    # Generate just a few samples for demonstration
    db = generate_fiber_database(parameter_ranges, num_samples=10)
    print("\nSample from generated database:")
    print(db.head())
    
    # Save database to CSV
    db.to_csv("fiber_designs.csv", index=False)
    print("Database saved to fiber_designs.csv")
