import torch
import torch.nn as nn
import math
import numpy as np

# Compute N(k)
def compute_Nk(k, K, s0, s1):
    term = (k / K) * ((s1 + 1)**2 - s0**2) + s0**2
    return math.ceil(math.sqrt(term) - 1)

# Compute µ(k)
def compute_mu(Nk, s0, mu0):
    return math.exp((s0 * math.log(mu0)) / Nk)

# Compute t
def timestep2t(T, N, i):
    rho = 7
    epsilon = 0.002
    return (epsilon**(1/rho) + ((i-1)/(N-1))*(T**(1/rho) - epsilon**(1/rho)))**rho

class Conv1DEmbedding(nn.Module):
    def __init__(self, input_size, embed_dim):
        super(Conv1DEmbedding, self).__init__()
        self.input_size = input_size
        self.conv1d = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),  # Output a fixed-size embedding
        )
        self.fc = nn.Linear(64, embed_dim)

    def forward(self, x):
        x = x.unsqueeze(-1) if x.ndim == 2 else x  # (batch_size, seq_len, 1)
        x = x.transpose(1, 2)  # (batch_size, 1, seq_len)
        x = self.conv1d(x)  # (batch_size, 32, 1)
        x = x.squeeze(-1)  # (batch_size, 32)
        x = self.fc(x)  # (batch_size, embed_dim)
        return x
    
class BiGRUEmbedding(nn.Module):
    def __init__(self, input_size, embed_dim):
        super(BiGRUEmbedding, self).__init__()
        self.input_size = input_size
        self.bi_gru = nn.GRU(input_size=self.input_size, hidden_size=embed_dim // 2, num_layers=1, batch_first=True, bidirectional=True)

    def forward(self, x):
        # Call flatten_parameters to ensure weights are contiguous
        self.bi_gru.flatten_parameters()

        x = x.unsqueeze(-1) if x.ndim == 2 else x  # (batch_size, seq_len, 1)
        x, _ = self.bi_gru(x)  # (batch_size, seq_len, embed_dim)
        x = x[:, -1, :]  # Take the last time step's output (batch_size, embed_dim)
        return x
    
class GRUEmbedding(nn.Module):
    def __init__(self, input_size, embed_dim):
        super(GRUEmbedding, self).__init__()
        self.input_size = input_size
        self.gru = nn.GRU(input_size=1, hidden_size=embed_dim, num_layers=1, batch_first=True, bidirectional=False)

    def forward(self, x):
        # Call flatten_parameters to ensure weights are contiguous
        self.gru.flatten_parameters()

        x = x.unsqueeze(-1) if x.ndim == 2 else x  # (batch_size, seq_len, 1)
        x, _ = self.gru(x)  # (batch_size, seq_len, embed_dim)
        x = x[:, -1, :]  # Take the last time step's output (batch_size, embed_dim)
        return x
    
class LSTMEmbedding(nn.Module):
    def __init__(self, input_size, embed_dim):
        super(LSTMEmbedding, self).__init__()
        self.input_size = input_size
        self.lstm = nn.LSTM(input_size=1, hidden_size=embed_dim, num_layers=1, batch_first=True)

    def forward(self, x):
        self.lstm.flatten_parameters()  # Compact the weights for better performance
        x = x.unsqueeze(-1) if x.ndim == 2 else x  # (batch_size, seq_len, 1)
        _, (h, _) = self.lstm(x)  # h: (num_layers * num_directions, batch_size, embed_dim)
        x = h[-1]  # Take the last layer's output (batch_size, embed_dim)
        return x
    
class Dense(nn.Module):
  # A fully connected layer that reshapes outputs to feature maps
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.dense = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        return self.dense(x)[..., None]
    
class GaussianFourierProjection(nn.Module):
  # Gaussian random features for encoding time steps
    def __init__(self, embed_dim, scale=30.):
        super().__init__()
        self.W = nn.Parameter(torch.randn(embed_dim // 2) * scale, requires_grad=False)

    def forward(self, x):
        x_proj = x[:, None] * self.W[None, :] * 2 * np.pi
        output = torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)
        return output

# def loss_fn(predictions, parameters):
#     # prediction error
#     mse_loss = nn.MSELoss()
#     prediction_error = mse_loss(predictions, parameters)

#     # bounds penalty
#     device = predictions.device
#     params_lower_bd = torch.log(torch.tensor([1e-8, 1e-8, 1e-8, 1e-8, 1e-8], dtype=torch.float32, device=device))
#     params_upper_bd = torch.log(torch.tensor([0.6, 1.0, 0.8, 0.1, 0.1], dtype=torch.float32, device=device))

#     bounds_penalty = (
#     nn.functional.relu(predictions - params_upper_bd).sum(dim=1) +  # Penalize values above upper bound (per sample)
#     nn.functional.relu(params_lower_bd - predictions).sum(dim=1)   # Penalize values below lower bound (per sample)
#     )
#     bounds_penalty = torch.mean(bounds_penalty, dim=0)
#     lambda1 = 1

#     return prediction_error + lambda1 * bounds_penalty

# # Add perturbation to data
# def perturb_data(x, t):
#     device = x.device
#     noise = torch.randn_like(x).to(torch.float).to(device)  # Gaussian noise
#     x_t = x + t * noise  # Perturbation proportional to time t
#     return x_t, noise

# # Function to approximate the posterior of x given y
# def generate_posterior_samples(model, y, x_dim, num_samples=100):
#     t_samples = torch.ones(num_samples, 1)  # Random time steps
#     y_expanded = y.repeat(num_samples, 1)  # Expand y for each noise sample

#     # Initialize x_t from noise and y
#     x_t_samples = torch.normal(mean=0, std=1, size=(num_samples, x_dim))  # Start from a noisy distribution

#     # Generate samples
#     with torch.no_grad():
#         x_samples = model(x_t_samples, y_expanded, t_samples)
#     return x_samples

# def generate_posterior_samples_multistep(model, y, x_dim, num_samples=100, num_steps=10):
#     # Initialize with Gaussian noise
#     x_t_samples = torch.randn(num_samples, x_dim)  # Start from pure noise
#     y_expanded = y.repeat(num_samples, 1)  # Expand y for all noise samples

#     # Define time schedule (e.g., linear)
#     t_schedule = torch.linspace(1.0, 0.0, steps=num_steps)

#     # Iterative refinement
#     with torch.no_grad():
#         for i in range(num_steps - 1):
#             t = t_schedule[i]  # Current time step
#             next_t = t_schedule[i + 1]  # Next time step

#             # Predict x_0 using the current x_t and t
#             t_batch = torch.full((num_samples, 1), t)  # Current t
#             predicted_x0 = model(x_t_samples, y_expanded, t_batch)

#             # Update x_t for the next step
#             noise_scale = (next_t - t).abs().sqrt()  # Noise adjustment based on time step
#             noise = torch.randn_like(x_t_samples) if next_t > 0 else 0.0  # Add noise if not the final step
#             x_t_samples = predicted_x0 + noise_scale * noise

#     return x_t_samples
