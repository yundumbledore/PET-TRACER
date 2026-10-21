"""Forward models ported unchanged from InteractiveViewer.py."""
import numpy as np

class KineticModel:
    def get_num_params(self):
        raise NotImplementedError

    def step(self, state, inp, dt):
        raise NotImplementedError

    def output(self, state, inp):
        raise NotImplementedError

class ZeroTissueCompartment(KineticModel):
    def __init__(self, params):
        self.params = params  # [Vb]

    def get_num_params(self):
        return 1

    def init_state(self):
        return 0.0

    def step(self, state, Cp_blood, dt):
        return state

    def output(self, state, Cp_blood):
        return self.params[0] * Cp_blood

class OneTissueCompartment(KineticModel):
    def __init__(self, params):
        self.params = params  # [K1, k2, Vb]

    def get_num_params(self):
        return 3

    def init_state(self):
        return 0.0

    def step(self, CT, Cp_blood, dt):
        K1, k2, _ = self.params
        dCT = K1 * Cp_blood - k2 * CT
        return CT + dCT * dt

    def output(self, CT, Cp_blood):
        Vb = self.params[2]
        return Vb * Cp_blood + (1.0 - Vb) * CT

class OneTissueDualInput(KineticModel):
    def __init__(self, params):
        self.params = params  # [K1, k2, Vb, ka, fA]

    def get_num_params(self):
        return 5

    def init_state(self):
        return (0.0, 0.0)

    def step(self, state, Ca, dt):
        CT, Cpv = state
        K1, k2, _, ka, fA = self.params
        dCpv = ka * Ca - ka * Cpv
        Cp = (1.0 - fA) * Cpv + fA * Ca
        dCT = K1 * Cp - k2 * CT
        return (CT + dCT * dt, Cpv + dCpv * dt)

    def output(self, state, Ca):
        CT, Cpv = state
        Vb, _, fA = self.params[2], self.params[3], self.params[4]
        Cp = (1.0 - fA) * Cpv + fA * Ca
        return (1.0 - Vb) * CT + Vb * Cp

class TwoTissueCompartment(KineticModel):
    def __init__(self, params):
        self.params = params  # [K1, k2, k3, k4, Vb]

    def get_num_params(self):
        return 5

    def init_state(self):
        return (0.0, 0.0)

    def step(self, state, Cp_blood, dt):
        Cf, Cp = state
        K1, k2, k3, k4, _ = self.params
        dCf = -(k2 + k3) * Cf + k4 * Cp + K1 * Cp_blood
        dCp = k3 * Cf - k4 * Cp
        return (Cf + dCf * dt, Cp + dCp * dt)

    def output(self, state, Cp_blood):
        Cf, Cp = state
        Vb = self.params[4]
        return Vb * Cp_blood + (1.0 - Vb) * (Cf + Cp)

class TwoTissueDualInput(KineticModel):
    def __init__(self, params):
        self.params = params  # [K1, k2, k3, k4, Vb, ka, fA]

    def get_num_params(self):
        return 7

    def init_state(self):
        return (0.0, 0.0, 0.0)

    def step(self, state, Ca, dt):
        Cf, Cm, Cpv = state
        K1, k2, k3, k4, _, ka, fA = self.params
        dCpv = ka * Ca - ka * Cpv
        Cp = (1.0 - fA) * Cpv + fA * Ca
        dCf = K1 * Cp - (k2 + k3) * Cf + k4 * Cm
        dCm = k3 * Cf - k4 * Cm
        return (Cf + dCf * dt, Cm + dCm * dt, Cpv + dCpv * dt)

    def output(self, state, Ca):
        Cf, Cm, Cpv = state
        Vb, _, fA = self.params[4], self.params[5], self.params[6]
        Cp = (1.0 - fA) * Cpv + fA * Ca
        return (1.0 - Vb) * (Cf + Cm) + Vb * Cp

class ThreeTissueCompartment(KineticModel):
    def __init__(self, params):
        self.params = params  # [K1, k2, k3, k4, k5, k6, Vb]

    def get_num_params(self):
        return 7

    def init_state(self):
        return (0.0, 0.0, 0.0)

    def step(self, state, Cp_blood, dt):
        Cf, Cb, Cns = state
        K1, k2, k3, k4, k5, k6, _ = self.params
        dCf = K1 * Cp_blood - (k2 + k3 + k5) * Cf + k4 * Cb + k6 * Cns
        dCb = k3 * Cf - k4 * Cb
        dCns = k5 * Cf - k6 * Cns
        return (Cf + dCf * dt, Cb + dCb * dt, Cns + dCns * dt)

    def output(self, state, Cp_blood):
        Cf, Cb, Cns = state
        Vb = self.params[6]
        return Vb * Cp_blood + (1.0 - Vb) * (Cf + Cb + Cns)

class ThreeTissueDualInput(KineticModel):
    def __init__(self, params):
        self.params = params  # [K1, k2, k3, k4, k5, k6, Vb, ka, fA]

    def get_num_params(self):
        return 9

    def init_state(self):
        return (0.0, 0.0, 0.0, 0.0)

    def step(self, state, Ca, dt):
        Cf, Cb, Cns, Cpv = state
        K1, k2, k3, k4, k5, k6, _, ka, fA = self.params
        dCpv = ka * Ca - ka * Cpv
        Cp = (1.0 - fA) * Cpv + fA * Ca
        dCf = K1 * Cp - (k2 + k3 + k5) * Cf + k4 * Cb + k6 * Cns
        dCb = k3 * Cf - k4 * Cb
        dCns = k5 * Cf - k6 * Cns
        return (Cf + dCf * dt, Cb + dCb * dt, Cns + dCns * dt, Cpv + dCpv * dt)

    def output(self, state, Ca):
        Cf, Cb, Cns, Cpv = state
        Vb, _, fA = self.params[6], self.params[7], self.params[8]
        Cp = (1.0 - fA) * Cpv + fA * Ca
        return (1.0 - Vb) * (Cf + Cb + Cns) + Vb * Cp

MODEL_CLASSES = {0: ZeroTissueCompartment, 1: OneTissueCompartment, 2: OneTissueDualInput, 3: TwoTissueCompartment, 4: TwoTissueDualInput, 5: ThreeTissueCompartment, 6: ThreeTissueDualInput}

def simulate_kinetic_model(model_idx, params, input_dense, dt_dense, meas_idx):
    """Run forward Euler integration on the dense time grid and sample at meas_idx."""
    if model_idx == 0:
        return params[0] * input_dense[meas_idx]

    model_cls = MODEL_CLASSES[model_idx]
    model = model_cls(params)
    state = model.init_state()

    y_dense = np.empty(np.shape(input_dense), dtype=float)
    step_fn = model.step
    out_fn = model.output
    for t_i in range(len(input_dense)):
        inp_val = input_dense[t_i]
        state = step_fn(state, inp_val, dt_dense[t_i])
        y_dense[t_i] = out_fn(state, inp_val)

    return y_dense[meas_idx]
