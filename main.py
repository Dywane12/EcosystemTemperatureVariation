import tkinter as tk
from tkinter import messagebox, Scrollbar, Canvas, Frame
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
root = tk.Tk()
root.title("Climate Simulation GUI")
root.geometry("400x600")
root.configure(bg="#2E3440")

# Create a scrollable canvas
canvas = Canvas(root)
scrollbar = Scrollbar(root, orient="vertical", command=canvas.yview)
scrollable_frame = Frame(canvas)
scrollable_frame.bind(
    "<Configure>",
    lambda e: canvas.configure(
        scrollregion=canvas.bbox("all")
    )
)
canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
canvas.configure(yscrollcommand=scrollbar.set)

# Styling variables
label_style = {"fg": "#D8DEE9", "bg": "#2E3440", "font": ("Arial", 12)}
entry_style = {"font": ("Arial", 12), "bg": "#434C5E", "fg": "#ECEFF4", "bd": 2}
button_style = {"font": ("Arial", 14), "bg": "#81A1C1", "fg": "#2E3440", "activebackground": "#88C0D0", "bd": 3}

# Parameter descriptions and effects
descriptions = {
    "lambda_": "Average heat capacity. Higher values slow down temperature changes.",
    "b": "Incoming solar radiation flux. Increasing this will raise the temperature.",
    "S": "Surface albedo. Higher values reflect more sunlight, decreasing temperature.",
    "e": "Effective emissivity. Higher values increase energy loss to space.",
    "K": "Carrying capacity of vegetation. Higher values allow more vegetation.",
    "sigma": "Mortality rate. Higher values decrease vegetation faster.",
    "b1": "Growth rate coefficient. Higher values increase vegetation growth.",
    "T0": "Reference temperature (°C). Affects vegetation response to temperature.",
    "alpha2": "Temperature sensitivity coefficient. Higher values increase sensitivity.",
}

# Input fields and labels
def add_input(param_name, default_value):
    tk.Label(scrollable_frame, text=f"{param_name}:", **label_style).pack(pady=5)
    entry = tk.Entry(scrollable_frame, **entry_style)
    entry.insert(0, str(default_value))
    entry.pack(pady=5)
    explanation = descriptions.get(param_name, "No description available.")
    tk.Label(scrollable_frame, text=explanation, **label_style).pack(pady=1)
    return entry

days_entry = add_input("days", 1000)
time_points_entry = add_input("time_points", 1000)
temp_entry = add_input("temperature", 15)
lambda_entry = add_input("lambda_", 1.0)
b_entry = add_input("b", 0.3)
S_entry = add_input("S", 0.3)
e_entry = add_input("e", 0.6)
K_entry = add_input("K", 1.0)
sigma_entry = add_input("sigma", 0.1)
b1_entry = add_input("b1", 0.02)
T0_entry = add_input("T0", 20)
alpha2_entry = add_input("alpha2", 0.01)

# Run Simulation Button
def run_simulation():
    try:
        lambda_ = float(lambda_entry.get())
        b = float(b_entry.get())
        S = float(S_entry.get())
        e = float(e_entry.get())
        K = float(K_entry.get())
        sigma = float(sigma_entry.get())
        b1 = float(b1_entry.get())
        T0 = float(T0_entry.get())
        alpha2 = float(alpha2_entry.get())

        days = int(days_entry.get())
        time_points = int(time_points_entry.get())
        initial_temp = float(temp_entry.get()) + 273.15

        time = np.linspace(0, days, time_points)
        temp_solution = odeint(dT_dt, initial_temp, time)
        vegetation_solution = odeint(lambda u, t: du_dt(u, temp_solution[int(t) % len(temp_solution)][0]), [0.5], time)

        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1)
        plt.plot(time, temp_solution, label='Temperature (K)', color='royalblue')
        plt.xlabel('Time (days)')
        plt.ylabel('Temperature (K)')
        plt.title('Temperature Evolution')
        plt.figtext(0.5, 0.01, 'This graph shows how temperature evolves over time, based on the given parameters.', ha='center')
        plt.legend()
        plt.grid()

        plt.subplot(1, 2, 2)
        plt.plot(time, vegetation_solution, label='Vegetation Density', color='forestgreen')
        plt.xlabel('Time (days)')
        plt.ylabel('Density')
        plt.title('Vegetation Density Evolution')
        plt.figtext(0.5, 0.01, 'This graph shows how vegetation density evolves over time, based on the given parameters.', ha='center')
        plt.legend()
        plt.grid()

        plt.tight_layout()
        plt.show()
    except ValueError:
        messagebox.showerror("Input Error", "Please enter valid numbers.")

run_button = tk.Button(scrollable_frame, text="Run Simulation", command=run_simulation, **button_style)
run_button.pack(pady=20)

canvas.pack(side="left", fill="both", expand=True)
scrollbar.pack(side="right", fill="y")
canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")

root.mainloop()
