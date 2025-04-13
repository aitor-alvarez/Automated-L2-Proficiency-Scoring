import pandas as pd
import matplotlib.pyplot as plt

# Create the dataset
data = {
    'Model': ['XGBoost'] * 11 + ['LGBLight'] * 11,
    'width': [
        0.128, 0.133, 0.135, 0.136, 0.133, 0.134, 0.133, 0.135, 0.133, 0.136, 0.136,
        0.172, 0.176, 0.178, 0.175, 0.181, 0.182, 0.184, 0.180, 0.179, 0.176, 0.182
    ],
    'sample size (unlabeled)': [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 1100] * 2
}

# Convert to DataFrame
df = pd.DataFrame(data)

# Pivot for plotting
pivot_df = df.pivot(index='sample size (unlabeled)', columns='Model', values='width')

# Plot
pivot_df.plot(marker='o')
plt.title('Width vs. Sample Size for Each Model')
plt.xlabel('Sample Size (unlabeled)')
plt.ylabel('Width')
plt.grid(True)
plt.legend(title='Model')
plt.tight_layout()
plt.show()