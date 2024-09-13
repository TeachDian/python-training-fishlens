import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Load the results.csv file
results_df = pd.read_csv('runs/detect/train/results.csv')

# Strip any leading/trailing spaces in the column names
results_df.columns = results_df.columns.str.strip()

# Convert 'train/box_loss' column to numeric, forcing errors to NaN
results_df['train/box_loss'] = pd.to_numeric(results_df['train/box_loss'], errors='coerce')

# Replace inf and NaN values with the median of the column
median_value = results_df['train/box_loss'].median()
results_df['train/box_loss'] = results_df['train/box_loss'].replace([np.inf, -np.inf], np.nan)
results_df['train/box_loss'].fillna(median_value, inplace=True)

# Check the max and min again to ensure that inf and invalid values are handled
print("Max value in train/box_loss after cleaning:", results_df['train/box_loss'].max())
print("Min value in train/box_loss after cleaning:", results_df['train/box_loss'].min())

# Now, you can safely clip the values if needed
results_df['train/box_loss'] = results_df['train/box_loss'].clip(upper=1.0)

# Define a smoothing function using a rolling window
def smooth(data, window_size=5):
    return data.rolling(window=window_size).mean()

# Extract columns and apply smoothing to relevant metrics
epochs = results_df['epoch']
train_box_loss = results_df['train/box_loss']
train_cls_loss = results_df['train/cls_loss']
train_dfl_loss = results_df['train/dfl_loss']
val_box_loss = results_df['val/box_loss']
val_cls_loss = results_df['val/cls_loss']
val_dfl_loss = results_df['val/dfl_loss']
precision = results_df['metrics/precision(B)']
recall = results_df['metrics/recall(B)']
map50 = results_df['metrics/mAP50(B)']
map50_95 = results_df['metrics/mAP50-95(B)']

# Apply smoothing to some metrics
smoothed_precision = smooth(precision)
smoothed_recall = smooth(recall)
smoothed_map50 = smooth(map50)
smoothed_map50_95 = smooth(map50_95)

# Create the plot
fig, axs = plt.subplots(2, 5, figsize=(20, 8))

# First row: Training losses and precision/recall
axs[0, 0].plot(epochs, train_box_loss, label='Box Loss')
axs[0, 0].set_title('train/box_loss')

axs[0, 1].plot(epochs, train_cls_loss, label='Class Loss')
axs[0, 1].set_title('train/cls_loss')

axs[0, 2].plot(epochs, train_dfl_loss, label='DFL Loss')
axs[0, 2].set_title('train/dfl_loss')

axs[0, 3].plot(epochs, precision, label='results', marker='o', markersize=2)
axs[0, 3].plot(epochs, smoothed_precision, label='smooth', linestyle='dotted')
axs[0, 3].set_title('metrics/precision(B)')

axs[0, 4].plot(epochs, recall, label='results', marker='o', markersize=2)
axs[0, 4].plot(epochs, smoothed_recall, label='smooth', linestyle='dotted')
axs[0, 4].set_title('metrics/recall(B)')

# Second row: Validation losses and mAP metrics
axs[1, 0].plot(epochs, val_box_loss, label='Box Loss')
axs[1, 0].set_title('val/box_loss')

axs[1, 1].plot(epochs, val_cls_loss, label='Class Loss')
axs[1, 1].set_title('val/cls_loss')

axs[1, 2].plot(epochs, val_dfl_loss, label='DFL Loss')
axs[1, 2].set_title('val/dfl_loss')

axs[1, 3].plot(epochs, map50, label='results', marker='o', markersize=2)
axs[1, 3].plot(epochs, smoothed_map50, label='smooth', linestyle='dotted')
axs[1, 3].set_title('metrics/mAP50(B)')

axs[1, 4].plot(epochs, map50_95, label='results', marker='o', markersize=2)
axs[1, 4].plot(epochs, smoothed_map50_95, label='smooth', linestyle='dotted')
axs[1, 4].set_title('metrics/mAP50-95(B)')

# Add legends and grid
for ax in axs.flat:
    ax.legend()
    ax.grid(True)

# Adjust layout to avoid overlap
plt.tight_layout()

# Save the plot as a PNG file
plt.savefig('results_smoothed.png')

# Show the plot
plt.show()
