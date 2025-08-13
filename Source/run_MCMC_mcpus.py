import os
# Optional: pin JAX to CPU explicitly, but it’ll pick CPU if no GPU is installed
os.environ["JAX_PLATFORMS"] = "cpu"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import tempfile, uuid
import numpy as np
import pandas as pd
from tqdm import tqdm
import jax
import jax.numpy as jnp
import arviz as az
import multiprocessing as mp
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

jax.config.update("jax_enable_x64", True)

def _init_worker(t_meas, Cb_meas, max_dt):
    tmp = tempfile.mkdtemp(prefix=f"pyt_{uuid.uuid4().hex}_")
    # Give EACH worker its own compiledir to avoid lock contention
    os.environ["PYTENSOR_FLAGS"] = f"mode=JAX,compiledir={tmp},cxx="
    # Import PyTensor only AFTER flags are set (per-process)
    global Op, pt, jax_funcify
    from pytensor.graph.op import Op
    import pytensor.tensor as pt
    from pytensor.link.jax.dispatch import jax_funcify

    global delta_t_pt, exp_neg_pt, exp_pos_pt, two_tc_op
    global t_dense_jnp, Cb_dense_jnp, dt_dense_jnp, meas_idx_jnp

    delta_t_pt, exp_neg_pt, exp_pos_pt = noise_model_variables(t_meas)  # defined at L28-L37
    (t_dense_jnp,
     Cb_dense_jnp,
     dt_dense_jnp,
     meas_idx_jnp) = kinetic_model_variables(max_dt, t_meas, Cb_meas)
    
    @jax.jit
    def get_curve_jax_2tc_precomp(thetas, Cb_d, dt_d):
        k1, k2, k3, k4, Vb = thetas
        def step(carry, inp):
            Cf, Cp = carry
            Cb_i, dt_i = inp
            dCf = -(k2 + k3) * Cf + k4 * Cp + k1 * Cb_i
            dCp =  k3 * Cf - k4 * Cp
            Cf = Cf + dCf * dt_i
            Cp = Cp + dCp * dt_i
            y_i = Vb * Cb_i + (1 - Vb) * (Cf + Cp)
            return (Cf, Cp), y_i

        (_ , _), y_dense = jax.lax.scan(step,
                                    (0.0, 0.0),
                                    (Cb_d, dt_d))
        return y_dense

    @jax.jit
    def predict_tac(thetas):
        y_dense = get_curve_jax_2tc_precomp(thetas,
                                            Cb_dense_jnp,
                                            dt_dense_jnp)
        return y_dense[meas_idx_jnp]

    class TwoTCPrecompOp(Op):
        itypes = [pt.dvector]
        otypes = [pt.dvector]

        @staticmethod
        def do_constant_folding(fgraph, node):
            # Prevent PyTensor from trying to fold this Op into a constant
            return False

        def perform(self, node, inputs, outputs):
            thetas, = inputs
            yj = predict_tac(jax.numpy.array(thetas))
            outputs[0][0] = np.array(yj, dtype=np.float64)

        def grad(self, inputs, output_gradients):
            thetas, = inputs
            gz,      = output_gradients
            _, pullback = jax.vjp(predict_tac, jax.numpy.array(thetas))
            grad_thetas, = pullback(jax.numpy.array(gz))
            return [np.array(grad_thetas, dtype=np.float64)]

    @jax_funcify.register(TwoTCPrecompOp)
    def jax_funcify_TwoTCPrecompOp(op, **kwargs):
        return lambda thetas: predict_tac(thetas)

    two_tc_op = TwoTCPrecompOp()

def noise_model_variables(t_meas):
    delta_t = np.concatenate([[t_meas[0]], np.diff(t_meas)])
    lamb    = np.log(2) / 109.8
    exp_neg = np.exp(-lamb * t_meas)
    exp_pos = np.exp( lamb * t_meas)
    delta_t_pt = pt.constant(delta_t)
    exp_neg_pt = pt.constant(exp_neg)
    exp_pos_pt = pt.constant(exp_pos)

    return delta_t_pt, exp_neg_pt, exp_pos_pt

def kinetic_model_variables(max_dt, t_meas, Cb_meas):
    t_dense = np.arange(t_meas[0], t_meas[-1] + 1e-8, max_dt)
    if t_dense[-1] < t_meas[-1]:
        t_dense = np.concatenate([t_dense, [t_meas[-1]]])

    Cb_dense = np.interp(t_dense, t_meas, Cb_meas)
    dt_dense = np.concatenate([[t_dense[0]], np.diff(t_dense)])
    meas_idx = np.searchsorted(t_dense, t_meas)

    t_dense_jnp   = jnp.array(t_dense,   dtype=jnp.float64)
    Cb_dense_jnp  = jnp.array(Cb_dense,  dtype=jnp.float64)
    dt_dense_jnp  = jnp.array(dt_dense,  dtype=jnp.float64)
    meas_idx_jnp  = jnp.array(meas_idx,  dtype=jnp.int32)

    return t_dense_jnp, Cb_dense_jnp, dt_dense_jnp, meas_idx_jnp

def run_inference(Ct_meas):
    import pymc as pm
    import pymc.sampling.jax as pmjax
    with pm.Model() as model:
        # 5 parameters + blood fraction
        k1  = pm.Uniform("k1",  0.01, 1.0)
        k2  = pm.Uniform("k2",  0.01, 2.0)
        k3  = pm.Uniform("k3",  0.01, 0.5)
        k4  = pm.Uniform("k4",  -0.1, 0.1)
        Vb  = pm.Uniform("Vb",  0.03, 0.2)
        l1  = pm.Uniform("l1",  2.9, 7.1)

        thetas = pt.stack([k1, k2, k3, k4, Vb], axis=0)

        Ct_pred = two_tc_op(thetas)
        Ct_pred = pt.maximum(Ct_pred, 1e-8)

        sigma_t = pt.sqrt(Ct_pred * exp_neg_pt / delta_t_pt) * exp_pos_pt
        sigma   = l1 * sigma_t
        
        pm.Normal("obs", mu=Ct_pred, sigma=sigma, observed=Ct_meas)

        trace = pmjax.sample_numpyro_nuts(
            draws=2000,
            tune=2000,
            chains=1,
            target_accept=0.9,
            progressbar=False,
            compute_convergence_checks=False
        )
    
    posterior = trace.posterior
    param_names = ["k1","k2","k3","k4","Vb"]
    flattened = [posterior[var].values.reshape(-1) for var in param_names]
    samples_array = np.stack(flattened, axis=1)
    return samples_array

def _run_indexed(ir):
    i, arg = ir
    return i, run_inference(arg)

if __name__ == "__main__":
    # Load data
    y_data = pd.read_hdf('../Yun_realdata/P001_PET_coronal_116_denoised_adjusted.h5')
    Cb_meas    = y_data.iloc[:,2].values

    # Sampling time points
    t_meas  = np.array([5.0, 10.0, 15.0, 20.0, 25.0, 30.0, 35.0,
                    40.0, 45.0, 50.0, 55.0, 60.0, 75.0, 90.0,
                    105.0,120.0,180.0,240.0,300.0,360.0,420.0,
                    480.0,600.0,720.0,840.0,960.0,1080.0,1260.0,
                    1440.0,1620.0,1800.0,2100.0,2400.0,2700.0,3000.0]) / 60.0
    
    # Discretization time step size
    max_dt = 0.01  # minutes

    args = y_data.iloc[:,3:].to_numpy().T
    n_voxels = len(args)

    ncpu = 112
    print('I will use {} cpus...'.format(ncpu))
    mp.set_start_method("spawn", force=True)
    estimates = [None] * n_voxels
    with mp.Pool(processes=ncpu, initializer=_init_worker, initargs=(t_meas, Cb_meas, max_dt)) as pool:
        for i, res in tqdm(
            pool.imap_unordered(_run_indexed, enumerate(args)),
            total=n_voxels, desc="voxels"
        ):
            estimates[i] = res

    all_samples = np.stack(estimates, axis=0)
    np.savez_compressed('../Parametric_Imaging/P001_slice116.npz', arr=all_samples)

