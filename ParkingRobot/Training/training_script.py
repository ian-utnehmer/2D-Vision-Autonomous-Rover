import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Load the data
try:
    df = pd.read_csv('paper_data_inches.csv')
    print("Data loaded successfully.")
except FileNotFoundError:
    print("Error: Create 'paper_data_inches.csv' first!")
    exit()

# Fit a 3rd-degree polynomial (Raw_Value -> Inches)
weights_inches = np.polyfit(df['raw_depth'], df['inches'], 3)

# Convert to Meters for ROS (1 meter = 39.37 inches)
# We divide the result by 39.37 so Robot is in Meters
weights_meters = weights_inches / 39.37

print("\n" + "="*45)
print("   ROS COEFFICIENTS (METERS)")
print("="*45)
print(f"A = {weights_meters[0]:.10f}")
print(f"B = {weights_meters[1]:.10f}")
print(f"C = {weights_meters[2]:.10f}")
print(f"D = {weights_meters[3]:.10f}")
print("="*45)
print("\nEquation: distance_m = A*x^3 + B*x^2 + C*x + D")

# Visual Validation
plt.figure(figsize=(10,6))
plt.scatter(df['raw_depth'], df['inches'], color='blue', label='Collected Points')
x_range = np.linspace(df['raw_depth'].min(), df['raw_depth'].max(), 100)
plt.plot(x_range, np.polyval(weights_inches, x_range), color='red', label='Polynomial Fit')
plt.xlabel('Jetson Raw Depth Value')
plt.ylabel('Distance (Inches)')
plt.title('Monodepth Calibration Curve')
plt.legend()
plt.grid(True)
plt.show()