"""Original Create_dataset.py split keys, parameter/noise draws and AIF order.

Draw full split arrays before chunking the Euler calculations. Folding keys per
chunk or changing the random-array shapes would change the reference sequence.
"""
import numpy as np


class HPCSampler:
    def __init__(self, cfg, input_count):
        try:
            import jax
            import jaxlib
        except ImportError as error:
            raise RuntimeError('Dataset generation requires JAX. Install the updated requirements in your PET-TRACER environment: python -m pip install -r requirements-gui.txt') from error
        jax.config.update('jax_enable_x64', True)
        self.jax=jax
        self.device=jax.devices('cpu')[0]
        self.cfg=cfg
        self.counts=[cfg['n_'+split] for split in ('train','val','test')]
        # Matches np.random.seed(seed); np.random.choice(...) without changing
        # process-global NumPy state.
        all_indices=np.random.RandomState(cfg['seed']).choice(input_count,size=sum(self.counts),replace=True)
        self.indices=dict(zip(('train','val','test'),np.split(all_indices,np.cumsum(self.counts)[:-1])))
        with jax.default_device(self.device):
            self.keys=dict(zip(('train','val','test'),jax.random.split(jax.random.PRNGKey(cfg['seed']),3)))
        self.metadata=dict(engine='Original JAX split-key sequence and NumPy RandomState AIF assignment',
                           jax_version=jax.__version__,jaxlib_version=jaxlib.__version__,
                           jax_enable_x64=True,jax_default_prng_impl=str(jax.config.jax_default_prng_impl),
                           jax_threefry_partitionable=bool(jax.config.jax_threefry_partitionable),
                           sampling_device='cpu',chunk_invariant=True)

    def draw(self, split, priors):
        jax=self.jax
        count=self.cfg['n_'+split]
        with jax.default_device(self.device):
            keys=jax.random.split(self.keys[split],10)
            columns=[]
            for prior in priors:
                columns.append(np.asarray(jax.random.uniform(keys[0],(count,1),minval=prior['min'],maxval=prior['max'])))
                keys=jax.random.split(keys[0],10)
            noise=self.cfg['noise_prior']
            amplitude=np.asarray(jax.random.uniform(keys[2],(count,1),minval=noise['min'],maxval=noise['max']))
            epsilon=np.asarray(jax.random.normal(keys[4],shape=(count,len(self.cfg['t_meas']))))
        return np.concatenate(columns,axis=1),amplitude,epsilon,self.indices[split]
