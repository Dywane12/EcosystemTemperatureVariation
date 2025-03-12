import tkinter as tk
from tkinter import messagebox
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint

# Constants
lambda_ = 1.0      # Average heat capacity
b = 0.3            # Incoming solar radiation flux
S = 0.3            # Surface albedo
kappa = 5.67e-8    # Stefan–Boltzmann constant
e = 0.6            # Effective emissivity
K = 1.0            # Carrying capacity
sigma = 0.1        # Mortality rate
b1 = 0.02          # Growth rate coefficient
T0 = 20            # Reference temperature (°C)
alpha2 = 0.01      # Temperature sensitivity coefficient

# Temperature evolution (Budyko–Sellers equation)
def dT_dt(T, t):
    return (b * (1 - S) - e * kappa * T**4) / lambda_

# Logistic growth model for vegetation density
def B(T):
    return b1 * np.exp(-T0 / T) * np.exp(-alpha2 * T)

def du_dt(u, T):
    return (B(T) - u / K) * u - sigma * u

# GUI Setup
def run_simulation():
    try:
        days = int(days_entry.get())
        time_points = int(points_entry.get())
        initial_temp = float(temp_entry.get()) + 273.15  # Convert to Kelvin

        # Time points
        time = np.linspace(0, days, time_points)

        # Solving the ODE for temperature
        temp_solution = odeint(dT_dt, initial_temp, time)

        # Solving the ODE for vegetation density (using the final temperature from previous calculation)
        vegetation_solution = odeint(lambda u, t: du_dt(u, temp_solution[int(t) % len(temp_solution)][0]), [0.5], time)

        # Plotting the results
        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1)
        plt.plot(time, temp_solution, label='Temperature (K)', color='royalblue')
        plt.xlabel('Time (days)')
        plt.ylabel('Temperature (K)')
        plt.title('Temperature Evolution')
        plt.legend()
        plt.grid()

        plt.subplot(1, 2, 2)
        plt.plot(time, vegetation_solution, label='Vegetation Density', color='forestgreen')
        plt.xlabel('Time (days)')
        plt.ylabel('Density')
        plt.title('Vegetation Density Evolution')
        plt.legend()
        plt.grid()

        plt.tight_layout()
        plt.show()
    except ValueError:
        messagebox.showerror("Input Error", "Please enter valid numbers.")

# Create the main window
root = tk.Tk()
root.title("Climate Simulation GUI")
root.geometry("400x300")
root.configure(bg="#2E3440")

# Styling variables
label_style = {"fg": "#D8DEE9", "bg": "#2E3440", "font": ("Arial", 12)}
entry_style = {"font": ("Arial", 12), "bg": "#434C5E", "fg": "#ECEFF4", "bd": 2}
button_style = {"font": ("Arial", 14), "bg": "#81A1C1", "fg": "#2E3440", "activebackground": "#88C0D0", "bd": 3}

# Input fields and labels
tk.Label(root, text="Number of Days:", **label_style).pack(pady=5)
days_entry = tk.Entry(root, **entry_style)
days_entry.insert(0, "1000")
days_entry.pack(pady=5)

tk.Label(root, text="Number of Time Points:", **label_style).pack(pady=5)
points_entry = tk.Entry(root, **entry_style)
points_entry.insert(0, "1000")
points_entry.pack(pady=5)

tk.Label(root, text="Initial Temperature (°C):", **label_style).pack(pady=5)
temp_entry = tk.Entry(root, **entry_style)
temp_entry.insert(0, "15")
temp_entry.pack(pady=5)

# Run button
run_button = tk.Button(root, text="Run Simulation", command=run_simulation, **button_style)
run_button.pack(pady=20)

# Start the GUI event loop
root.mainloop()
