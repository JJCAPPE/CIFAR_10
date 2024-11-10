import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.linear_model import LinearRegression

# Load CSV file
df = pd.read_csv("results.csv")

# Function to calculate average kernel size for each model
def calculate_average_kernel_size(layers_column):
    avg_kernel_sizes = []
    for layers in layers_column:
        layers = eval(layers)  # Convert string to list of dictionaries
        kernel_sizes = [layer["kernel_size"] for layer in layers if layer["type"] == "Conv2D"]
        avg_kernel_size = np.mean(kernel_sizes) if kernel_sizes else 0
        avg_kernel_sizes.append(avg_kernel_size)
    return avg_kernel_sizes

# Calculate the average kernel size for each model
df["average_kernel_size"] = calculate_average_kernel_size(df["layers"])

# Extract variables for the 3D plot
X = df[["num_layers", "average_kernel_size"]].values
y = df["accuracy"].values

# Fit a linear regression model to get the plane of best fit
reg = LinearRegression().fit(X, y)

# Create a meshgrid for the plane of best fit
x_surf = np.linspace(df["num_layers"].min(), df["num_layers"].max(), 10)
y_surf = np.linspace(df["average_kernel_size"].min(), df["average_kernel_size"].max(), 10)
x_surf, y_surf = np.meshgrid(x_surf, y_surf)
z_surf = reg.intercept_ + reg.coef_[0] * x_surf + reg.coef_[1] * y_surf

# Create a 3D plot
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')

# Plot layers, accuracy, and average kernel size
scatter = ax.scatter(df["num_layers"], df["accuracy"], df["average_kernel_size"], c=df["accuracy"], cmap="viridis", s=50)
ax.plot_surface(x_surf, z_surf, y_surf, color="orange", alpha=0.3, rstride=100, cstride=100)

# Add labels and title
ax.set_xlabel("Number of Layers")
ax.set_ylabel("Accuracy")
ax.set_zlabel("Average Kernel Size")
ax.set_title("3D Plot of Layers vs. Accuracy vs. Average Kernel Size with Best-Fit Plane")

plt.colorbar(scatter, ax=ax, label="Accuracy")
plt.show()