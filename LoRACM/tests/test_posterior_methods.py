"""Equation-level checks for posterior objectives and sampling schedules."""
import math
from pathlib import Path
import sys
import unittest

import numpy as np
import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from posterior_methods import imf_loss, imf_sample, sampling_schedule, scm_loss, scm_sample


class AffineIMF(nn.Module):
    def __init__(self):
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(.3, dtype=torch.double))

    def forward(self, state, condition, t, r):
        return self.scale * state + .2 * t[:, None, None] + .4 * r[:, None, None] + condition[:, None, :]


class AffineSCM(nn.Module):
    sigma_data = 1.7

    def __init__(self):
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(.3, dtype=torch.double))
        self.log_weight = nn.Parameter(torch.tensor(.2, dtype=torch.double))

    def forward(self, state, condition, t):
        return self.scale * state + .2 * t[:, None, None] + condition[:, None, :]

    def scm_log_weight(self, t):
        return self.log_weight.expand_as(t)


class PosteriorMethodTests(unittest.TestCase):
    def setUp(self):
        self.x = torch.tensor([[[.2, .8]], [[-.5, 1.1]]], dtype=torch.double)
        self.noise = torch.tensor([[[.3, -.6]], [[1.2, .5]]], dtype=torch.double)
        self.y = torch.tensor([[1.2, .2], [.4, -.3]], dtype=torch.double)
        self.t = torch.tensor([.6, .9], dtype=torch.double)
        self.r = torch.tensor([.2, .3], dtype=torch.double)
        self.weights = torch.tensor([[[.75, 1.5]]], dtype=torch.double)

    def test_imf_interval_matches_mean_velocity_identity_and_detaches_jvp(self):
        model = AffineIMF()
        t, r = self.t[:, None, None], self.r[:, None, None]
        z = (1-t) * self.x + t * self.noise
        with torch.no_grad():
            boundary = .3 * z + .6 * t + self.y[:, None, :]
            derivative = .2 + .3 * boundary
            u = .3 * z + .2 * t + .4 * r + self.y[:, None, :]
            error = u + (t-r) * derivative - (self.noise - self.x)
            expected = (self.weights * error.square()).mean()
            expected_gradient = (2 * self.weights * error * z).mean()

        loss = imf_loss(model, self.x, self.y, self.weights, noise=self.noise, t=self.t, r=self.r, adaptive=False)
        loss.backward()
        torch.testing.assert_close(loss, expected)
        torch.testing.assert_close(model.scale.grad, expected_gradient)
        self.assertNotEqual(model.scale.grad.item(), 0.)

    def test_imf_diagonal_matches_regular_flow_matching(self):
        model = AffineIMF()
        with torch.no_grad():
            t = self.t[:, None, None]
            z = (1-t) * self.x + t * self.noise
            expected = (model(z, self.y, self.t, self.t) - self.noise + self.x).square().mean()
            loss = imf_loss(model, self.x, self.y, noise=self.noise, t=self.t, r=self.t, adaptive=False)
        torch.testing.assert_close(loss, expected)

    def test_scm_learned_weight_and_detached_normalized_tangent(self):
        model = AffineSCM()
        sigma = model.sigma_data
        sin_t, cos_t = self.t.sin()[:, None, None], self.t.cos()[:, None, None]
        state = cos_t * self.x + sin_t * sigma * self.noise
        state_prime = cos_t * sigma * self.noise - sin_t * self.x
        with torch.no_grad():
            raw = model(state/sigma, self.y, self.t)
            raw_prime = .3 * state_prime / sigma + .2
            # Independently differentiate the endpoint f = cos(t)*state - sigma*sin(t)*raw.
            endpoint_prime = cos_t * state_prime - sin_t * state - sigma * (cos_t * raw + sin_t * raw_prime)
            tangent = cos_t * endpoint_prime
            target_change = tangent / (torch.linalg.vector_norm(tangent.flatten(1), dim=1)[:, None, None]+.1)
            error = (self.weights * target_change.square()).mean()
            expected = math.exp(.2) * error - .2
            expected_scale_gradient = -2 * math.exp(.2) * (self.weights * target_change * state/sigma).mean()
            expected_weight_gradient = math.exp(.2) * error - 1
            score = scm_loss(model, self.x, self.y, self.weights, noise=self.noise, t=self.t, adaptive=False)
            torch.testing.assert_close(score, (self.weights * tangent.square()).mean())

        loss = scm_loss(model, self.x, self.y, self.weights, noise=self.noise, t=self.t)
        loss.backward()
        torch.testing.assert_close(loss, expected)
        torch.testing.assert_close(model.scale.grad, expected_scale_gradient)
        torch.testing.assert_close(model.log_weight.grad, expected_weight_gradient)
        self.assertNotEqual(model.scale.grad.item(), 0.)

    def test_imf_sampler_composes_state_dependent_jumps_without_renoising(self):
        model = AffineIMF()
        times = sampling_schedule('imf', 2)
        draws = iter([self.noise.clone()])
        with torch.no_grad():
            state_half = self.noise - .5*(.3*self.noise + .2 + .2 + self.y[:, None, :])
            expected = state_half - .5*(.3*state_half + .1 + self.y[:, None, :])
            sample = imf_sample(model, self.y, lambda: next(draws), times)
        torch.testing.assert_close(sample, expected)
        np.testing.assert_array_equal(sampling_schedule('cm', 3), np.linspace(1., 0., 3, endpoint=False))

    def test_scm_sampler_uses_scaled_gaussian_and_trigflow_renoising(self):
        model = AffineSCM()
        times = sampling_schedule('scm', 2, sigma_data=model.sigma_data)
        np.testing.assert_array_equal(times, [math.pi/2, 1.1])
        sigma, angle = model.sigma_data, times[1]
        with torch.no_grad():
            first = -sigma * (.3*self.noise + .2*(math.pi/2) + self.y[:, None, :])
            fresh_noise = self.noise.flip(0)
            state = math.cos(angle)*first + sigma*math.sin(angle)*fresh_noise
            expected = math.cos(angle)*state - sigma*math.sin(angle)*(.3*state/sigma + .2*angle + self.y[:, None, :])
            draws = iter([self.noise.clone(), fresh_noise])
            sample = scm_sample(model, self.y, lambda: next(draws), times)
        torch.testing.assert_close(sample, expected)


if __name__ == '__main__':
    unittest.main()
