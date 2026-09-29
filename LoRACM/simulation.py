import os
import numpy as np
import pandas as pd
import jax
import jax.numpy as jnp
from jax import random
from typing import Dict, Any, Tuple, List, Optional
from functools import partial

# Enable 64-bit precision for exponential accuracy
jax.config.update("jax_enable_x64", True)

# ==========================================
# 1. STATISTICAL DISTRIBUTIONS (PRIORS)
# ==========================================

class PriorSampler:
    """
    Handles sampling from various distributions using JAX.
    Updated to support 'fixed' values.
    """
    @staticmethod
    def sample(key, config: Dict[str, Any], shape: Tuple[int]):
        dist_type = config.get('type', 'uniform')
        
        if dist_type == 'fixed':
            # Returns a constant value for all samples
            return jnp.full(shape, config['value'])

        elif dist_type == 'uniform':
            return random.uniform(key, shape, minval=config['min'], maxval=config['max'])
        
        elif dist_type == 'normal':
            return config['mu'] + config['sigma'] * random.normal(key, shape)
        
        elif dist_type == 'lognormal':
            # exp(Normal(mu, sigma))
            z = config['mu'] + config['sigma'] * random.normal(key, shape)
            val = jnp.exp(z)
            if 'min' in config: val = jnp.maximum(val, config['min'])
            if 'max' in config: val = jnp.minimum(val, config['max'])
            return val
            
        elif dist_type == 'logstudentt':
            # exp(StudentT(nu) * sigma + mu)
            nu = config['nu']
            mu = config['mu']
            sigma = config['sigma']
            t_dist = random.t(key, nu, shape)
            val = jnp.exp(mu + sigma * t_dist)
            if 'min' in config: val = jnp.maximum(val, config['min'])
            if 'max' in config: val = jnp.minimum(val, config['max'])
            return val
        
        else:
            raise ValueError(f"Unknown distribution type: {dist_type}")

# ==========================================
# 2. INPUT FUNCTION MODELS (FENG)
# ==========================================

def feng_model_jax(t, params):
    """
    Standard Feng Model (Fallback if no external input).
    params: [beta1, beta2, beta3, kappa1, kappa2, kappa3, t0]
    """
    b1, b2, b3, k1, k2, k3, t0 = params
    t_shifted = t - t0
    
    term1 = (b1 * t_shifted - b2 - b3) * jnp.exp(-k1 * t_shifted)
    term2 = b2 * jnp.exp(-k2 * t_shifted)
    term3 = b3 * jnp.exp(-k3 * t_shifted)
    
    val = term1 + term2 + term3
    return jnp.where(t >= t0, val, 0.0)

# ==========================================
# 3. KINETIC MODEL DEFINITIONS
# ==========================================

class KineticModel:
    """Base class for Kinetic Models."""
    def get_num_params(self): raise NotImplementedError
    def step_fn(self): raise NotImplementedError
    def output_fn(self): raise NotImplementedError

class TwoTissueCompartment(KineticModel):
    """
    Standard 2TC Model.
    Params: K1, k2, k3, k4, Vb
    """
    def get_num_params(self): return 5 
    
    def step_fn(self):
        def step(carry, inp):
            Cf, Cp = carry 
            Cp_blood, dt = inp
            K1, k2, k3, k4, Vb = self.params
            
            dCf = -(k2 + k3) * Cf + k4 * Cp + K1 * Cp_blood
            dCp = k3 * Cf - k4 * Cp
            
            Cf_new = Cf + dCf * dt
            Cp_new = Cp + dCp * dt
            return (Cf_new, Cp_new), (Cf_new, Cp_new)
        return step

    def output_fn(self, state, Cp_blood, params):
        Cf, Cp = state
        Vb = params[4]
        return Vb * Cp_blood + (1.0 - Vb) * (Cf + Cp)

class TwoTissueDualInput(KineticModel):
    """
    2TCM with Dual Input (Arterial + Venous).
    Params: K1, k2, k3, k4, Vb, alpha
    """
    def get_num_params(self): return 6 
    
    def step_fn(self):
        def step(carry, inp):
            Cf, Cp = carry
            (Ca, Cv), dt = inp 
            K1, k2, k3, k4, Vb, alpha = self.params
            
            # Mix inputs: alpha=0 -> Pure Artery
            Cin = (1.0 - alpha) * Ca + alpha * Cv
            
            dCf = -(k2 + k3) * Cf + k4 * Cp + K1 * Cin
            dCp = k3 * Cf - k4 * Cp
            
            Cf_new = Cf + dCf * dt
            Cp_new = Cp + dCp * dt
            return (Cf_new, Cp_new), (Cf_new, Cp_new)
        return step

    def output_fn(self, state, inputs, params):
        Ca, Cv = inputs
        Cf, Cp = state
        Vb = params[4]
        alpha = params[5]
        
        Cin = (1.0 - alpha) * Ca + alpha * Cv
        return Vb * Cin + (1.0 - Vb) * (Cf + Cp)

class SRTM(KineticModel):
    """
    Simplified Reference Tissue Model (SRTM).
    Params: R1, k2, BPnd
    Input: Reference Region Curve (C_ref)
    
    Note: 'k2' here represents the target tissue clearance.
    If you fix this value in priors, you assume constant target washout.
    """
    def get_num_params(self): return 3
    
    def step_fn(self):
        def step(carry, inp):
            X = carry # Convolution state
            C_ref, dt = inp
            R1, k2, BPnd = self.params
            
            k2a = k2 / (1.0 + BPnd)
            dX = C_ref - k2a * X
            
            X_new = X + dX * dt
            return X_new, X_new
        return step
        
    def output_fn(self, state, C_ref, params):
        X = state
        R1, k2, BPnd = params
        k2a = k2 / (1.0 + BPnd)
        return R1 * C_ref + (k2 - R1 * k2a) * X

# ==========================================
# 4. SIMULATOR
# ==========================================

def solve_ode(model_cls, params, input_funcs, dt_dense, meas_idx):
    """
    Generic ODE solver using jax.lax.scan
    """
    model = model_cls()
    model.params = params
    
    if isinstance(input_funcs, tuple):
        scan_inputs = (input_funcs[0], input_funcs[1], dt_dense)
    else:
        scan_inputs = (input_funcs, dt_dense)
        
    if isinstance(model, SRTM):
        init_state = 0.0
    else:
        init_state = (0.0, 0.0)
        
    _, states_dense = jax.lax.scan(model.step_fn(), init_state, scan_inputs)
    
    def get_obs(s, i):
        if isinstance(input_funcs, tuple):
            inp_val = (i[0], i[1]) 
        else:
            inp_val = i[0] 
        return model.output_fn(s, inp_val, params)
    
    y_dense = jax.vmap(get_obs)(states_dense, scan_inputs)
    return y_dense[meas_idx]

@partial(jax.jit, static_argnames=['model_cls', 'is_dual_input', 'use_custom_input'])
def simulate_batch(key, param_matrix, feng_params, custom_inputs_dense, t_dense, dt_dense, meas_idx, model_cls, is_dual_input, use_custom_input):
    
    def single_sim(p_kin, p_feng, c_custom):
        # 1. Determine Input Function(s)
        if use_custom_input:
            if is_dual_input:
                inputs = (c_custom[0], c_custom[1])
            else:
                inputs = c_custom
        
        elif is_dual_input:
            p_feng_a = p_feng[:7]
            p_feng_v = p_feng[7:]
            Ca_dense = feng_model_jax(t_dense, p_feng_a)
            Cv_dense = feng_model_jax(t_dense, p_feng_v)
            inputs = (Ca_dense, Cv_dense)
            
        else:
            C_dense = feng_model_jax(t_dense, p_feng)
            inputs = C_dense
            
        return solve_ode(model_cls, p_kin, inputs, dt_dense, meas_idx)

    # Handling Map Shapes
    if custom_inputs_dense is None:
        batch_size = param_matrix.shape[0]
        if is_dual_input:
            custom_inputs_dense = jnp.zeros((batch_size, 2, t_dense.shape[0]))
        else:
            custom_inputs_dense = jnp.zeros((batch_size, t_dense.shape[0]))
        
    if feng_params is None:
        batch_size = param_matrix.shape[0]
        n_feng = 14 if is_dual_input else 7
        feng_params = jnp.zeros((batch_size, n_feng)) 

    return jax.vmap(single_sim)(param_matrix, feng_params, custom_inputs_dense)

@partial(jax.jit, static_argnames=['is_dual_input'])
def add_noise(key, y_clean, l1, t_meas, half_life, is_dual_input):
    lamb = jnp.log(2) / half_life
    delta_t = jnp.concatenate([jnp.array([t_meas[0]]), jnp.diff(t_meas)])
    exp_neg = jnp.exp(-lamb * t_meas)
    exp_pos = jnp.exp( lamb * t_meas)
    
    y_safe = jnp.maximum(y_clean, 1e-8)
    sigma_t = jnp.sqrt(y_safe * exp_neg / delta_t) * exp_pos
    sigma = l1 * sigma_t
    
    eps = random.normal(key, shape=y_clean.shape)
    return y_clean + sigma * eps

# ==========================================
# 5. DATA GENERATION ORCHESTRATOR
# ==========================================

def generate_dataset(
    n_samples, 
    model_class, 
    kinetic_priors, 
    feng_priors, 
    noise_prior,
    t_meas, 
    key, 
    half_life,
    is_dual_input=False,
    external_input_curves: Optional[np.ndarray] = None,
    return_clean=True
):
    # Time Setup
    max_dt = 0.01
    t_dense = np.arange(t_meas[0], t_meas[-1] + 1e-8, max_dt)
    if t_dense[-1] < t_meas[-1]: t_dense = np.concatenate([t_dense, [t_meas[-1]]])
    dt_dense = np.concatenate([[t_dense[0]], np.diff(t_dense)])
    meas_idx = np.searchsorted(t_dense, t_meas)
    
    t_dense_j = jnp.array(t_dense)
    dt_dense_j = jnp.array(dt_dense)
    meas_idx_j = jnp.array(meas_idx)
    
    keys = random.split(key, 10)
    
    # 1. Sample Kinetic Parameters (PRESERVES ORDER FROM DICT)
    k_params_list = []
    for k_name in kinetic_priors:
        p_cfg = kinetic_priors[k_name]
        col = PriorSampler.sample(keys[0], p_cfg, (n_samples, 1))
        keys = random.split(keys[0], 10)
        k_params_list.append(col)
    
    x_kinetic = jnp.concatenate(k_params_list, axis=1)
    
    # 2. Handle Inputs
    x_feng = None
    custom_inputs_dense_j = None
    y_input_to_save = None
    
    use_custom = (external_input_curves is not None)
    
    if use_custom:
        if is_dual_input:
            if external_input_curves.shape != (n_samples, 2, len(t_meas)):
                raise ValueError(f"Dual Input requires (N, 2, T), got {external_input_curves.shape}")
            ext_dense_list = []
            for i in range(n_samples):
                ca_dense = np.interp(t_dense, t_meas, external_input_curves[i, 0, :])
                cv_dense = np.interp(t_dense, t_meas, external_input_curves[i, 1, :])
                ext_dense_list.append(np.stack([ca_dense, cv_dense]))
            custom_inputs_dense_j = jnp.array(np.stack(ext_dense_list))
            y_input_to_save = jnp.array(external_input_curves.reshape(n_samples, -1))
        else:
            if external_input_curves.shape != (n_samples, len(t_meas)):
                raise ValueError(f"Single Input requires (N, T), got {external_input_curves.shape}")
            ext_dense_list = []
            for i in range(n_samples):
                d_curve = np.interp(t_dense, t_meas, external_input_curves[i])
                ext_dense_list.append(d_curve)
            custom_inputs_dense_j = jnp.array(np.stack(ext_dense_list))
            y_input_to_save = jnp.array(external_input_curves)
        
        n_feng = 14 if is_dual_input else 7
        x_feng = jnp.zeros((n_samples, n_feng))

    else:
        # Fallback to Feng
        f_list = []
        for f_name in ["beta1", "beta2", "beta3", "kappa1", "kappa2", "kappa3", "t0"]:
            col = PriorSampler.sample(keys[1], feng_priors[f_name], (n_samples, 1))
            f_list.append(col)
        x_feng = jnp.concatenate(f_list, axis=1)
        
        if is_dual_input:
            f_list_2 = []
            k_split = random.split(keys[1], 8)
            for i, f_name in enumerate(["beta1", "beta2", "beta3", "kappa1", "kappa2", "kappa3", "t0"]):
                col = PriorSampler.sample(k_split[i], feng_priors[f_name], (n_samples, 1))
                f_list_2.append(col)
            x_feng_2 = jnp.concatenate(f_list_2, axis=1)
            x_feng = jnp.concatenate([x_feng, x_feng_2], axis=1)

        if is_dual_input:
            inp_a = jax.vmap(lambda p: feng_model_jax(t_meas, p[:7]))(x_feng)
            inp_v = jax.vmap(lambda p: feng_model_jax(t_meas, p[7:]))(x_feng)
            y_input_to_save = jnp.concatenate([inp_a, inp_v], axis=1)
        else:
            y_input_to_save = jax.vmap(lambda p: feng_model_jax(t_meas, p))(x_feng)

    l1 = PriorSampler.sample(keys[2], noise_prior, (n_samples, 1))
    
    print(f"  Simulating ODEs for {n_samples} samples...")
    CHUNK_SIZE = 100000  # Adjust this depending on your GPU/RAM
    y_tissue_list = []
    
    for i in range(0, n_samples, CHUNK_SIZE):
        end_idx = min(i + CHUNK_SIZE, n_samples)
        
        # Slice inputs for the current chunk
        chunk_x_kinetic = x_kinetic[i:end_idx]
        chunk_x_feng = x_feng[i:end_idx] if x_feng is not None else None
        chunk_custom_inputs = custom_inputs_dense_j[i:end_idx] if custom_inputs_dense_j is not None else None

        # Run simulation for the chunk
        chunk_y_tissue = simulate_batch(
            keys[3], 
            chunk_x_kinetic, 
            chunk_x_feng, 
            chunk_custom_inputs, 
            t_dense_j, 
            dt_dense_j, 
            meas_idx_j, 
            model_class, 
            is_dual_input, 
            use_custom
        )
        y_tissue_list.append(chunk_y_tissue)
        
    # Recombine the chunks back into a single array
    y_tissue = jnp.concatenate(y_tissue_list, axis=0)

    # y_tissue = simulate_batch(
    #     keys[3], x_kinetic, x_feng, custom_inputs_dense_j, 
    #     t_dense_j, dt_dense_j, meas_idx_j, model_class, 
    #     is_dual_input, use_custom
    # )
    
    if return_clean:
        y_final_tissue = y_tissue
    else:
        y_final_tissue = add_noise(keys[4], y_tissue, l1, t_meas, half_life, is_dual_input)
        
    y_combined = jnp.concatenate([y_final_tissue, y_input_to_save], axis=1)
    x_combined = jnp.concatenate([x_kinetic, l1], axis=1)
    
    return np.array(y_combined), np.array(x_combined)

import copy

def get_central_priors(priors_dict, scale=0.5):
    """
    Returns a deep copy of the priors with tightened distributions.
    scale: Fraction of the original width/variance to retain (e.g., 0.5 = 50%).
    """
    central_priors = copy.deepcopy(priors_dict)
    
    for key, config in central_priors.items():
        dist_type = config.get('type', 'uniform')
        
        if dist_type == 'fixed':
            continue
            
        elif dist_type == 'uniform':
            # Squeeze the min and max towards the midpoint
            midpoint = (config['max'] + config['min']) / 2.0
            half_width = (config['max'] - config['min']) / 2.0
            config['min'] = midpoint - (half_width * scale)
            config['max'] = midpoint + (half_width * scale)
            
        elif dist_type in ['normal', 'lognormal', 'logstudentt']:
            # Reduce the standard deviation (sigma) to group samples closer to mu
            if 'sigma' in config:
                config['sigma'] = config['sigma'] * scale
                
    return central_priors

# ==========================================
# 6. CONFIGURATION & MAIN
# ==========================================

