"""
Fuzzy Logic Temperature Control System
Course: Soft Computing
Description: A fuzzy logic controller that determines fan speed based on 
             temperature and humidity inputs using scikit-fuzzy library.
"""

import numpy as np
import matplotlib.pyplot as plt
import skfuzzy as fuzz
from skfuzzy import control as ctrl
from mpl_toolkits.mplot3d import Axes3D

# ============================================================================
# STEP 1: Define Input and Output Variables with their universes of discourse
# ============================================================================

# Input variable: Temperature (0-100°C)
temperature = ctrl.Antecedent(np.arange(0, 101, 1), 'temperature')

# Input variable: Humidity (0-100%)
humidity = ctrl.Antecedent(np.arange(0, 101, 1), 'humidity')

# Output variable: Fan Speed (0-100%)
fan_speed = ctrl.Consequent(np.arange(0, 101, 1), 'fan_speed')

# ============================================================================
# STEP 2: Define Membership Functions using Triangular MFs (trimf)
# ============================================================================

# Temperature membership functions
# Cold: peaks at 0°C, extends to 50°C
temperature['Cold'] = fuzz.trimf(temperature.universe, [0, 0, 50])
# Warm: peaks at 50°C, spans 25-75°C
temperature['Warm'] = fuzz.trimf(temperature.universe, [25, 50, 75])
# Hot: peaks at 100°C, starts from 50°C
temperature['Hot'] = fuzz.trimf(temperature.universe, [50, 100, 100])

# Humidity membership functions
# Dry: peaks at 0%, extends to 50%
humidity['Dry'] = fuzz.trimf(humidity.universe, [0, 0, 50])
# Comfort: peaks at 50%, spans 25-75%
humidity['Comfort'] = fuzz.trimf(humidity.universe, [25, 50, 75])
# Wet: peaks at 100%, starts from 50%
humidity['Wet'] = fuzz.trimf(humidity.universe, [50, 100, 100])

# Fan Speed membership functions
# Low: peaks at 0%, extends to 50%
fan_speed['Low'] = fuzz.trimf(fan_speed.universe, [0, 0, 50])
# Medium: peaks at 50%, spans 25-75%
fan_speed['Medium'] = fuzz.trimf(fan_speed.universe, [25, 50, 75])
# High: peaks at 100%, starts from 50%
fan_speed['High'] = fuzz.trimf(fan_speed.universe, [50, 100, 100])

# ============================================================================
# STEP 3: Visualize Membership Functions
# ============================================================================

# Plot Temperature membership functions
temperature.view()
plt.title('Temperature Membership Functions')
plt.tight_layout()

# Plot Humidity membership functions
humidity.view()
plt.title('Humidity Membership Functions')
plt.tight_layout()

# Plot Fan Speed membership functions
fan_speed.view()
plt.title('Fan Speed Membership Functions')
plt.tight_layout()

# ============================================================================
# STEP 4: Define Fuzzy Rules (Rule Base)
# ============================================================================

# Rule 1: Cold temperature + Dry humidity → Low fan speed
rule1 = ctrl.Rule(temperature['Cold'] & humidity['Dry'], fan_speed['Low'])

# Rule 2: Cold temperature + Comfort humidity → Low fan speed
rule2 = ctrl.Rule(temperature['Cold'] & humidity['Comfort'], fan_speed['Low'])

# Rule 3: Cold temperature + Wet humidity → Medium fan speed
rule3 = ctrl.Rule(temperature['Cold'] & humidity['Wet'], fan_speed['Medium'])

# Rule 4: Warm temperature + Dry humidity → Low fan speed
rule4 = ctrl.Rule(temperature['Warm'] & humidity['Dry'], fan_speed['Low'])

# Rule 5: Warm temperature + Comfort humidity → Medium fan speed
rule5 = ctrl.Rule(temperature['Warm'] & humidity['Comfort'], fan_speed['Medium'])

# Rule 6: Warm temperature + Wet humidity → High fan speed
rule6 = ctrl.Rule(temperature['Warm'] & humidity['Wet'], fan_speed['High'])

# Rule 7: Hot temperature + Dry humidity → Medium fan speed
rule7 = ctrl.Rule(temperature['Hot'] & humidity['Dry'], fan_speed['Medium'])

# Rule 8: Hot temperature + Comfort humidity → High fan speed
rule8 = ctrl.Rule(temperature['Hot'] & humidity['Comfort'], fan_speed['High'])

# Rule 9: Hot temperature + Wet humidity → High fan speed
rule9 = ctrl.Rule(temperature['Hot'] & humidity['Wet'], fan_speed['High'])

# ============================================================================
# STEP 5: Create Control System and Simulation
# ============================================================================

# Create the control system with all rules
fan_control = ctrl.ControlSystem([rule1, rule2, rule3, rule4, rule5, 
                                   rule6, rule7, rule8, rule9])

# Create a simulation instance
fan_simulation = ctrl.ControlSystemSimulation(fan_control)

# ============================================================================
# STEP 6: Test the System with Sample Inputs
# ============================================================================

# Define test inputs
test_temperature = 30  # °C
test_humidity = 70     # %

# Pass inputs to the simulation
fan_simulation.input['temperature'] = test_temperature
fan_simulation.input['humidity'] = test_humidity

# Compute the result (defuzzification)
fan_simulation.compute()

# Get the crisp output value
output_fan_speed = fan_simulation.output['fan_speed']

# Display results
print("=" * 60)
print("FUZZY LOGIC TEMPERATURE CONTROL SYSTEM - SIMULATION RESULTS")
print("=" * 60)
print(f"Input Temperature: {test_temperature}°C")
print(f"Input Humidity: {test_humidity}%")
print(f"Output Fan Speed: {output_fan_speed:.2f}%")
print("=" * 60)

# Visualize the defuzzification process
fan_speed.view(sim=fan_simulation)
plt.title(f'Fan Speed Output for T={test_temperature}°C, H={test_humidity}%')
plt.tight_layout()

# ============================================================================
# STEP 7: Generate 3D Control Surface
# ============================================================================

# Create meshgrid for temperature and humidity
temp_range = np.arange(0, 101, 5)
humid_range = np.arange(0, 101, 5)
temp_mesh, humid_mesh = np.meshgrid(temp_range, humid_range)

# Initialize output array
fan_output = np.zeros_like(temp_mesh)

# Compute fan speed for each combination of temperature and humidity
print("\nGenerating 3D control surface...")
for i in range(temp_mesh.shape[0]):
    for j in range(temp_mesh.shape[1]):
        try:
            fan_simulation.input['temperature'] = temp_mesh[i, j]
            fan_simulation.input['humidity'] = humid_mesh[i, j]
            fan_simulation.compute()
            fan_output[i, j] = fan_simulation.output['fan_speed']
        except:
            # Handle any edge cases
            fan_output[i, j] = 0

# Create 3D surface plot
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')

# Plot the surface
surf = ax.plot_surface(temp_mesh, humid_mesh, fan_output, 
                       cmap='viridis', edgecolor='none', alpha=0.9)

# Add labels and title
ax.set_xlabel('Temperature (°C)', fontsize=12, labelpad=10)
ax.set_ylabel('Humidity (%)', fontsize=12, labelpad=10)
ax.set_zlabel('Fan Speed (%)', fontsize=12, labelpad=10)
ax.set_title('3D Control Surface: Fan Speed vs Temperature & Humidity', 
             fontsize=14, fontweight='bold', pad=20)

# Add colorbar
fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5, label='Fan Speed (%)')

# Set viewing angle for better visualization
ax.view_init(elev=25, azim=135)

plt.tight_layout()

# ============================================================================
# STEP 8: Additional Test Cases
# ============================================================================

print("\n" + "=" * 60)
print("ADDITIONAL TEST CASES")
print("=" * 60)

# Define multiple test scenarios
test_cases = [
    (10, 20),   # Cold & Dry
    (30, 50),   # Warm & Comfort
    (80, 80),   # Hot & Wet
    (45, 30),   # Warm & Dry
    (90, 40),   # Hot & Dry
]

for temp, humid in test_cases:
    fan_simulation.input['temperature'] = temp
    fan_simulation.input['humidity'] = humid
    fan_simulation.compute()
    result = fan_simulation.output['fan_speed']
    print(f"T={temp:3d}°C, H={humid:3d}% → Fan Speed: {result:6.2f}%")

print("=" * 60)

# Show all plots
plt.show()

print("\nSimulation completed successfully!")
print("All membership functions, control surface, and results are displayed.")

