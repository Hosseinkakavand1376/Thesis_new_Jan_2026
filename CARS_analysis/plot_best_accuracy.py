#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os

# Step 1: Define the dataset
data = {
    "PLS components": [2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4],
    "Algorithm": ["Original CARS", "Original CARS", "Original CARS", "Calibrated CARS", "Calibrated CARS", "Calibrated CARS",
                    "Original CARS", "Original CARS", "Original CARS", "Calibrated CARS", "Calibrated CARS", "Calibrated CARS",
                    "Original CARS", "Original CARS", "Original CARS", "Calibrated CARS", "Calibrated CARS"],
    "Wavelengths": [1133, 12, 6, 1133, 10, 6, 1133, 25, 19, 1133, 30, 20, 1133, 34, 19, 1133, 19],
    "Accuracy": [0.77, 0.84, 0.83, 0.76, 0.85, 0.85, 0.93, 0.93, 0.95, 0.95, 0.95, 0.91, 0.99, 0.93, 0.93, 0.96, 0.93]
}

# Step 2: Create a DataFrame
df = pd.DataFrame(data)

# Step 3: Filter out rows with Wavelengths = 1133
filtered_df = df[df["Wavelengths"] != 1133]

# Step 4: Ensure the directory for saving plots exists
os.makedirs("plots", exist_ok=True)

# Step 5: Increase font size and figure size for the plot
plt.figure(figsize=(12, 8))
sns.set_style('whitegrid')
sns.set_context("paper", font_scale=2)  # Adjust font scale for better readability

# Step 6: Create scatter plot with seaborn
sns.scatterplot(data=filtered_df, x="Wavelengths", y="Accuracy", hue="PLS components", style="Algorithm", s=350, palette='viridis')

plt.xlabel("Number of Wavelengths")
plt.ylabel("Accuracy")
# plt.legend(title="Legend")
plt.title("Best Accuracy Combination")
plt.grid(True)

plt.show()
# creae folder if it does not exist
import os
if not os.path.exists('plots'):
    os.makedirs('plots')

# Show and save the plot
plt.savefig("plots/best_accuracy_plot.pdf")
plt.close()



# In[ ]:




