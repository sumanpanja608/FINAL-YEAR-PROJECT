import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
file_path = r'D:\Final Year Project\Independent_expanded_with_Actual_Bandgap.csv'
data = pd.read_csv(file_path)

# Compute correlation matrix
corr_matrix = data.corr()

# Set sans-serif font
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial']  # Use 'DejaVu Sans' if Arial not available

# Set figure dimensions in inches for 174 mm width
fig_width_in = 174 / 25.4
fig_height_in = fig_width_in * 0.8
plt.figure(figsize=(fig_width_in, fig_height_in))

# Plot heatmap
ax = sns.heatmap(
    corr_matrix,
    annot=True,
    cmap='coolwarm',
    annot_kws={"size": 8},     # ~8 pt font
    cbar_kws={'shrink': 0.8},
    linewidths=0.3,              # Line width = 0.3 pt = 0.1 mm
    linecolor='gray'
)

# Remove title inside figure
ax.set_title('')  # Ensure no internal title

# Set consistent font sizes
ax.tick_params(axis='both', labelsize=10)

# Save with tight layout and high-resolution TIFF
plt.tight_layout()
plt.savefig('Correlation_Matrix.tiff', dpi=1200, format='tiff')
plt.show()
