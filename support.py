import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import beta

def visualize_beta_distribution(a,b):
    # Parameters for the Beta distribution
    # alpha, beta_param = a, b  # Shape parameters

    # Generate x values
    x = np.linspace(0, 1, 1000)

    # Compute the Beta distribution's probability density function (PDF)
    pdf = beta.pdf(x, a, b)

    # Plot the Beta distribution
    plt.figure(figsize=(8, 5))
    plt.plot(x, pdf, label=f'Beta(α={a}, β={b})', color='blue')
    plt.title('Beta Distribution', fontsize=14)
    plt.xlabel('x', fontsize=12)
    plt.ylabel('Probability Density', fontsize=12)
    plt.legend(fontsize=12)
    plt.grid(alpha=0.3)
    plt.show()

visualize_beta_distribution(0.5,5)
