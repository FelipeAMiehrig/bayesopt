import matplotlib.pyplot as plt
import numpy as np 
import pandas as pd 
import seaborn as sns
import torch

def plot_samples_before_after(samples_before, samples_after, likelihood):
    # Adjusting the plot: original points as large circles, noisy points as crosses
    likelihood = likelihood.detach().numpy()
    len_before, len_after = np.squeeze(samples_before['lengthscale'][:,0].detach().numpy()), np.squeeze(samples_after['lengthscale'][:,:,0].detach().numpy())
    out_before, out_after = np.squeeze(samples_before['outputscale'].detach().numpy()), np.squeeze(samples_after['outputscale'].detach().numpy())
    len_before, len_after = np.log(len_before), np.log(len_after)
    out_before, out_after =  np.log(out_before), np.log(out_after)

    fig = plt.figure(figsize=(10, 12))  # Adjusted figure size for layout
    gs = fig.add_gridspec(4, 2, width_ratios=(5, 1), height_ratios=(1, 5, 0.5, 0.1), hspace=0.05, wspace=0.05)

    # Main scatter plot with density background
    ax_main = fig.add_subplot(gs[1, 0])
    sns.kdeplot(x=out_after, y=len_after, ax=ax_main, cmap='viridis', fill=True, bw_adjust=0.5)
    scatter = ax_main.scatter(out_after, len_after, c=likelihood, cmap='viridis', s=100, edgecolors='black', label='SGD optimized', marker='o')
    ax_main.scatter(out_before, len_before, c=likelihood, cmap='viridis', alpha=0.5, s=100, label='prior sample', marker='x')

    # KDE plot for X-axis (original points only) colored with the same palette
    ax_kdex = fig.add_subplot(gs[0, 0], sharex=ax_main)
    sns.kdeplot(x=out_after, ax=ax_kdex, fill=True, bw_adjust=0.5, cmap='viridis')
    ax_kdex.get_xaxis().set_visible(False)
    ax_kdex.get_yaxis().set_visible(False)

    # KDE plot for Y-axis (original points only) colored with the same palette
    ax_kdey = fig.add_subplot(gs[1, 1], sharey=ax_main)
    sns.kdeplot(y=len_after, ax=ax_kdey, fill=True, bw_adjust=0.5, cmap='viridis')
    ax_kdey.get_xaxis().set_visible(False)
    ax_kdey.get_yaxis().set_visible(False)
    ax_main.set_xlim([-15,15])
    ax_main.set_ylim([-15,15])   
    # Formatting
    ax_main.grid(False)
    ax_main.set_xlabel('outputscale')
    ax_main.set_ylabel('lenghtscale')
    ax_main.legend()

    # Colorbar below the main plot
    ax_colorbar = fig.add_subplot(gs[3, 0])  # Positioned closer to the main plot
    plt.colorbar(scatter, cax=ax_colorbar, orientation='horizontal', label='mll')


    plt.show()


def get_yhat(gp, test_X, tkwargs, batch_size = 100):

    total_batches = test_X.size(0) // batch_size
    Y_full = torch.Tensor().to(**tkwargs)
    for i in range(total_batches):
        start_idx = i * batch_size
        end_idx = start_idx + batch_size
        batch_X = test_X[start_idx:end_idx]
        posterior = gp.posterior(batch_X)
        Y_hat = posterior.mean
        Y_full = torch.cat((Y_full, Y_hat),0)
    return Y_full

""" def plot_gps(test_X, std, test_Y, Y_hat):

        # Define the sine function
    x = test_X.to_numpy()
    y = test_Y.to_numpy()

    # Generate noisy versions of y
    
    gps_y = Y_hat.to_numpy()

# Plot th
    # Calculate the 96% confidence interval for the original sine function
    # Since this is a mock example and there's no "real" data variability around the mean (the sine function),
    # we simulate a confidence interval by assuming a hypothetical variability
    ci = 1.96 * std

    # Upper and lower bounds of the confidence interval
    y_lower = y - ci
    y_upper = y + ci

    # Re-plot the original plot with the confidence interval added
    plt.figure(figsize=(10, 6))
    plt.plot(x, y, label='true function', color='red')
    plt.fill_between(x, y_lower, y_upper, color='red', alpha=0.3, label='95% CI')
    #colors = plt.cm.viridis(np.linspace(0, 1, len(noisy_ys)))
    # Plot each noisy version with a different color according to its noise level
    for gp_y in gps_y:
        plt.plot(x, gp_y)

    plt.legend()
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Sine Function with Various Levels of Noise and 96% CI for the Original')
    plt.show() """