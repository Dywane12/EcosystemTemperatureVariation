import sys
from PyQt5.QtWidgets import (QApplication, QWidget, QVBoxLayout, QLabel, QLineEdit, QPushButton, QScrollArea, QFormLayout, QMessageBox, QGroupBox, QHBoxLayout, QSpacerItem, QSizePolicy)
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint

# Constants
default_values = {
    'lambda_': 1.0, 'b': 0.3, 'S': 0.3, 'e': 0.6, 'K': 1.0, 'sigma': 0.1, 'b1': 0.02, 'T0': 20.0, 'alpha2': 0.01,
    'days': 1000, 'time_points': 1000, 'initial_temp': 288.15
}

descriptions = {
    'lambda_': "Average heat capacity. Higher values slow down temperature changes.",
    'b': "Incoming solar radiation flux. Increasing this will raise the temperature.",
    'S': "Surface albedo. Higher values reflect more sunlight, decreasing temperature.",
    'e': "Effective emissivity. Higher values increase energy loss to space.",
    'K': "Carrying capacity of vegetation. Higher values allow more vegetation.",
    'sigma': "Mortality rate. Higher values decrease vegetation faster.",
    'b1': "Growth rate coefficient. Higher values increase vegetation growth.",
    'T0': "Reference temperature (°C). Affects vegetation response to temperature.",
    'alpha2': "Temperature sensitivity coefficient. Higher values increase sensitivity.",
    'days': "Number of simulation days. More days give longer-term trends.",
    'time_points': "Number of time points for the simulation. Higher values give smoother curves.",
    'initial_temp': "Initial temperature in Kelvin. Starting point for the simulation."
}

# Temperature evolution (Budyko–Sellers equation)
def dT_dt(T, t, lambda_, b, S, e):
    return (b * (1 - S) - e * 5.67e-8 * T**4) / lambda_

# Logistic growth model for vegetation density
def B(T, b1, T0, alpha2):
    return b1 * np.exp(-T0 / T) * np.exp(-alpha2 * T)

def du_dt(u, T, K, sigma, b1, T0, alpha2):
    return (B(T, b1, T0, alpha2) - u / K) * u - sigma * u

class ClimateSimulation(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Climate Simulation GUI")
        self.setGeometry(100, 100, 800, 800)

        layout = QVBoxLayout(self)
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)

        content = QWidget()
        content_layout = QVBoxLayout(content)

        self.inputs = {}
        for param, default in default_values.items():
            group_box = QGroupBox(param)
            group_layout = QVBoxLayout()

            label = QLabel(f"{param} - {descriptions[param]}")
            entry = QLineEdit(str(default))
            self.inputs[param] = entry

            group_layout.addWidget(label)
            group_layout.addWidget(entry)
            group_box.setLayout(group_layout)

            content_layout.addWidget(group_box)

        run_button = QPushButton("Run Simulation")
        run_button.clicked.connect(self.run_simulation)

        content_layout.addWidget(run_button)
        scroll_area.setWidget(content)
        layout.addWidget(scroll_area)

    def run_simulation(self):
        try:
            params = {k: float(self.inputs[k].text()) for k in default_values.keys()}
            time = np.linspace(0, params['days'], int(params['time_points']))
            temp_solution = odeint(lambda T, t: dT_dt(T, t, params['lambda_'], params['b'], params['S'], params['e']), [params['initial_temp']], time)
            vegetation_solution = odeint(lambda u, t: du_dt(u, temp_solution[int(t) % len(temp_solution)][0], params['K'], params['sigma'], params['b1'], params['T0'], params['alpha2']), [0.5], time)

            plt.figure(figsize=(14, 7))
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
            QMessageBox.critical(self, "Input Error", "Please enter valid numbers.")

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = ClimateSimulation()
    window.show()
    sys.exit(app.exec_())
