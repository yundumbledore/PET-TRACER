from model import *
from utility import *
import torch
import numpy as np
import json
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns

def Sampler(model, y, num_posterior_samples, t=[1.0, 0.75, 0.50, 0.25]):

    batch_size, y_dim = y.shape
    y_repeated = y.unsqueeze(1).repeat(1, num_posterior_samples, 1)  # Shape: [batch_size, num_posterior_samples, y_dim]
    y_repeated = y_repeated.view(-1, y_dim)  # Flatten to [batch_size * num_posterior_samples, y_dim]
    
    x_noisy = torch.randn(batch_size * num_posterior_samples, 1, 5, device=y.device)  # Initialize with pure noise (Gaussian)
    
    for j in range(len(t)):
        t_tensor = torch.full((batch_size * num_posterior_samples,), t[j], device=y.device)  # Set the noise level t for all samples
        
        if j > 0:
            z = torch.randn(batch_size * num_posterior_samples, 1, 5, device=y.device)
            x_noisy = x_noisy + t[j] * z
            
        with torch.no_grad():
            x_noisy = model(x_noisy, y_repeated, t_tensor)  # Apply the trained model
    
    x_noisy = x_noisy.view(batch_size, num_posterior_samples, 5)
    
    return x_noisy

def computeKi(K1, k2, k3):
    return (K1 * k3)/(k2 + k3)

def get_model(device):
    # Load train_params
    with open("train_config.json", "r") as f:
        train_config = json.load(f)

    # Model settings
    x_dim = train_config["x_dim"]
    y_dim = train_config["y_dim"]
    hidden_dim = train_config["hidden_dim"]
    channels = train_config["channels"]
    embedy = train_config["embedy"]
    model = OneDUnet(x_dim, y_dim, hidden_dim, channels, embedy)

    # Load model
    model.load_state_dict(torch.load('CM_model_weights.pth', map_location=device))
    model.to(device)
    model.eval()
    return model

def demo_single_TAC(model, device, data_path, sample_size = 10000, num_timesteps = 3):
    with open("scaling_params.json", "r") as f:
        scaling_params = json.load(f)

    x_mean = np.array(scaling_params["x_mean"])[:5]
    x_std = np.array(scaling_params["x_std"])[:5]
    x_mean = torch.tensor(x_mean, dtype=torch.float32).to(device)
    x_std = torch.tensor(x_std, dtype=torch.float32).to(device)
    y_mean = scaling_params["y_mean"]
    y_std = scaling_params["y_std"]

    t = np.linspace(1.0, 0.0, num=num_timesteps, endpoint=False)
    y_data = pd.read_hdf(data_path)
    AIF = y_data.iloc[:,2].values
    y_data = y_data.iloc[:,3:].to_numpy().T
    AIFs = np.tile(AIF, (y_data.shape[0], 1))
    y_data = np.hstack((y_data, AIFs))
    y_data = (y_data - y_mean) / y_std

    obs = torch.tensor(y_data, dtype=torch.float32, device=device)
    # Inference
    theta_hat = Sampler(model, obs, sample_size, t)
    # Denormalize and reverse log transformation
    x_pred = theta_hat * x_std + x_mean
    samples = x_pred.cpu().detach().numpy()

    Ki = computeKi(samples[:,:,0], samples[:,:,1], samples[:,:,2]).squeeze()
    return Ki

def plot_posterior(title, parameter_name, cm, abc=None, xl = None, xr = None):
    custom_colors = ['#FF0000', '#008000', '#01153E', '#7E1E9C', '#F97306', '#7BC8F6', '#0000FF']

    fig = plt.figure(figsize=(3, 2))
    if abc is not None:
        sns.kdeplot(abc, color=custom_colors[1], linewidth=2, label='ABC')
    sns.kdeplot(cm, color=custom_colors[6],linewidth=2, label='CM')

    plt.xlabel(parameter_name,fontsize=14, fontweight='bold', labelpad=10)
    plt.ylabel('Density', fontsize=14, fontweight='bold', labelpad=10)
    plt.tick_params(axis='both', labelsize=10)
    plt.legend(fontsize=10, loc='best')
    plt.title(title, fontsize=14)
    if xl is not None: 
        plt.xlim([xl,xr])
    plt.show()

def plot_TAC(data_path):
    df = pd.read_hdf(data_path)
    tspan = np.array([5.0, 10.0, 15.0, 20.0, 25.0, 30.0, 35.0,
                      40.0, 45.0, 50.0, 55.0, 60.0, 75.0, 90.0,
                      105.0, 120.0, 180.0, 240.0, 300.0, 360.0, 420.0,
                      480.0, 600.0, 720.0, 840.0, 960.0, 1080.0, 1260.0,
                      1440.0, 1620.0, 1800.0, 2100.0, 2400.0, 2700.0, 3000.0])/60
    fig = plt.figure(figsize=(5,3.5))
    plt.plot(tspan, df.iloc[:,2],'--', label='Arterial input function', color='grey')
    plt.plot(tspan, df.iloc[:,3], 'k',label='Time-activity curve')

    plt.xlabel('Time (min)',fontweight='bold')
    plt.ylabel('Concentration (Bq/ml)',fontweight='bold')
    plt.rcParams['font.size'] = 14
    plt.legend(fontsize=18)

