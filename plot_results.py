import pandas as pd
import matplotlib.pyplot as plt

def plot_degradation_curves():
    # Load results
    df = pd.read_csv('robustness_results.csv')

    plt.figure(figsize=(12, 7))

    # Group by noise type and plot accuracy vs severity
    for noise_name, group in df.groupby('Noise_Type'):
        plt.plot(group['Severity'], group['Accuracy'], marker='o', label=noise_name.replace('_', ' ').title())

    plt.title('Model Accuracy vs. Corruption Severity Across 14 Noise Types', fontsize=14)
    plt.xlabel('Severity Level (1 to 5)', fontsize=12)
    plt.ylabel('Test Accuracy (%)', fontsize=12)
    plt.xticks([1, 2, 3, 4, 5])
    plt.ylim(0, 100)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9)
    plt.tight_layout()

    # Save as PDF per project guidelines
    plt.savefig('robustness_degradation.pdf', format='pdf')
    print("Summary plot saved to 'robustness_degradation.pdf'.")

if __name__ == "__main__":
    plot_degradation_curves()