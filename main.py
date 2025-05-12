import sys
from PyQt5.QtWidgets import (QApplication, QWidget, QVBoxLayout, QLabel, QLineEdit, QPushButton,
                             QScrollArea, QFormLayout, QMessageBox, QGroupBox, QHBoxLayout,
                             QStackedWidget, QRadioButton, QButtonGroup)
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# Constants
default_values = {
    # First model parameters
    'lambda_': 1.0, 'b': 240.0, 'S_base': 0.3, 'e': 0.6, 'K': 1.0, 'sigma': 0.1, 'b1': 0.02, 'T0': 20.0, 'alpha2': 0.01,
    'albedo_veg': 0.15, 'albedo_bare': 0.3, 'days': 1000, 'time_points': 1000, 'initial_temp': 288.15,
    'initial_veg': 0.5,

    # Second model parameters (from the image)
    'a': 5.67e-8, 's1': 0.95, 's0': 0.1, 'b1_alt': 0.1, 'T0_alt': 300.0, 'sigma_alt': 0.1, 'initial_temp_alt': 288.15,
    'initial_veg_alt': 0.5, 'q': 3, 'r': 2.0, 'a2': 0.004
}

descriptions = {
    # First model descriptions
    'lambda_': "Average heat capacity. Higher values slow down temperature changes.",
    'b': "Incoming solar radiation flux. Increasing this will raise the temperature.",
    'S_base': "Base surface albedo (without vegetation). Higher values reflect more sunlight.",
    'e': "Effective emissivity. Higher values increase energy loss to space.",
    'K': "Carrying capacity of vegetation. Higher values allow more vegetation.",
    'sigma': "Mortality rate. Higher values decrease vegetation faster.",
    'b1': "Growth rate coefficient. Higher values increase vegetation growth.",
    'T0': "Reference temperature (°C). Affects vegetation response to temperature.",
    'alpha2': "Temperature sensitivity coefficient. Higher values increase sensitivity.",
    'albedo_veg': "Albedo of vegetation-covered surface. Lower than bare soil albedo.",
    'albedo_bare': "Albedo of bare soil. Higher than vegetation albedo.",
    'days': "Number of simulation days. More days give longer-term trends.",
    'time_points': "Number of time points for the simulation. Higher values give smoother curves.",
    'initial_temp': "Initial temperature in Kelvin. Starting point for the simulation.",
    'initial_veg': "Initial vegetation density (0-1). Starting vegetation coverage.",

    # Second model descriptions (from the image)
    'a': "Temperature scaling coefficient in equation (9).",
    's1': "Albedo parameter s₁ in equation (9).",
    's0': "Albedo parameter s₀ in equation (9).",
    'b1_alt': "Growth rate coefficient b₁ in equation (10).",
    'T0_alt': "Reference temperature T₀ in equation (10).",
    'sigma_alt': "Mortality rate σ in equation (10).",
    'initial_temp_alt': "Initial temperature T for the second model.",
    'initial_veg_alt': "Initial vegetation density u for the second model.",
    'q': "Vegetation sensitivity parameter q in equation (9).",
    'r': "Temperature scaling parameter r in equation (9).",
    'a2': "Temperature inhibition parameter a₂ in equation (10)."
}


# Coupled system for the first model
def coupled_system_1(t, y, params):
    T, u = y  # Temperature and vegetation density

    # Calculate effective albedo based on vegetation coverage
    S_effective = params['albedo_bare'] * (1 - u) + params['albedo_veg'] * u

    # Temperature evolution (modified Budyko–Sellers equation)
    dT_dt = (params['b'] * (1 - S_effective) - params['e'] * 5.67e-8 * T ** 4) / params['lambda_']

    # Vegetation growth (temperature-dependent logistic growth)
    B = params['b1'] * np.exp(-params['T0'] / T) * np.exp(-params['alpha2'] * T)
    du_dt = (B - u / params['K']) * u - params['sigma'] * u

    return [dT_dt, du_dt]


# Update your coupled_system_2 function:
def coupled_system_2(t, y, params):
    T, u = y  # Temperature and vegetation density

    # Prevent division by zero or very small temperatures
    T = max(T, 1e-6)  # Ensure T doesn't get too close to zero

    # Implementation of the differential equation for temperature
    # dT/dt = (1/λ)(-aT⁴ + b[1-(s₁-s₀)e⁻ᶜ¹ᵘ-s₀])
    dT_dt = (1 / params['r']) * (-params['a'] * T ** 4 + params['b'] *
                                 (1 - (params['s1'] - params['s0']) * np.exp(-params['q'] * u) - params['s0']))

    # Implement safety checks to avoid overflow in the exponential terms
    # Limit the maximum/minimum values for the exponents
    exp1_arg = min(max(params['T0_alt'] / T, -700), 700)  # Limit between -700 and 700
    exp2_arg = min(max(-params['a2'] * T, -700), 700)  # Limit between -700 and 700

    # Implementation of the differential equation for vegetation
    # du/dt = (b₁e^(T₀/T)e^(-a₂T) - u)u - σu
    du_dt = (params['b1_alt'] * np.exp(exp1_arg) * np.exp(exp2_arg) - u) * u - params['sigma_alt'] * u

    return [dT_dt, du_dt]


class ClimateSimulation(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Coupled Climate-Vegetation Simulation")
        self.setGeometry(100, 100, 900, 800)
        self.setStyleSheet("background-color: #2c3e50; color: #ecf0f1;")

        # Main layout - using stacked widget for main screens
        self.main_layout = QVBoxLayout(self)
        self.main_stack = QStackedWidget()
        self.main_layout.addWidget(self.main_stack)

        # Create model selection screen
        self.create_model_selection_screen()

        # Create simulation screens for both models
        self.create_simulation_screen_model1()
        self.create_simulation_screen_model2()

        # Show the model selection screen first
        self.main_stack.setCurrentIndex(0)

    def create_model_selection_screen(self):
        # Model selection screen
        model_selection_screen = QWidget()
        model_selection_layout = QVBoxLayout(model_selection_screen)

        title = QLabel("Climate-Vegetation Model Selection")
        title.setStyleSheet("font-size: 30px; font-weight: bold; text-align: center; margin-bottom: 30px;")
        model_selection_layout.addWidget(title)

        # Radio button group for model selection
        self.model_group = QButtonGroup(self)

        # First model option
        self.model1_radio = QRadioButton("Model 1: Temperature-Vegetation Albedo Feedback")
        self.model1_radio.setStyleSheet("font-size: 18px; padding: 10px;")
        self.model1_radio.setChecked(True)  # Default selection
        self.model_group.addButton(self.model1_radio, 1)
        model_selection_layout.addWidget(self.model1_radio)

        model1_desc = QLabel(
            "This model couples temperature evolution with vegetation growth, where vegetation affects albedo.")
        model1_desc.setStyleSheet("font-size: 16px; margin-left: 25px; margin-bottom: 20px;")
        model_selection_layout.addWidget(model1_desc)

        # Second model option
        self.model2_radio = QRadioButton("Model 2: Alternative Temperature-Vegetation Coupling")
        self.model2_radio.setStyleSheet("font-size: 18px; padding: 10px;")
        self.model_group.addButton(self.model2_radio, 2)
        model_selection_layout.addWidget(self.model2_radio)

        model2_desc = QLabel(
            "This model implements the system of equations (9) and (10) with different temperature-vegetation interactions.")
        model2_desc.setStyleSheet("font-size: 16px; margin-left: 25px; margin-bottom: 20px;")
        model_selection_layout.addWidget(model2_desc)

        # Continue button
        continue_button = QPushButton("Continue to Simulation Setup")
        continue_button.setStyleSheet(
            "font-size: 22px; padding: 15px; margin-top: 40px; background-color: #2980b9; color: white; border-radius: 8px;")
        continue_button.clicked.connect(self.continue_to_simulation)
        model_selection_layout.addWidget(continue_button)

        # Add some spacing at the bottom
        model_selection_layout.addStretch()

        # Add to main stack
        self.main_stack.addWidget(model_selection_screen)

    def create_simulation_screen_model1(self):
        # Simulation screen for Model 1
        self.sim_screen1 = QWidget()
        sim_layout = QVBoxLayout(self.sim_screen1)

        # Title
        title = QLabel("Model 1: Temperature-Vegetation Albedo Feedback")
        title.setStyleSheet("font-size: 30px; font-weight: bold; text-align: center;")
        sim_layout.addWidget(title)

        # Description
        description = QLabel("Explore how temperature and vegetation interact through albedo feedback.")
        description.setStyleSheet("font-size: 18px; margin-bottom: 20px; text-align: center;")
        sim_layout.addWidget(description)

        # Parameter categories
        self.param_stack1 = QStackedWidget()

        # Create parameter groups for Model 1
        general_params1 = self.create_param_group(['days', 'time_points'], "model1")
        temp_params1 = self.create_param_group(['lambda_', 'b', 'e'], "model1")
        albedo_params1 = self.create_param_group(['S_base', 'albedo_veg', 'albedo_bare'], "model1")
        veg_params1 = self.create_param_group(['K', 'sigma', 'b1', 'T0', 'alpha2'], "model1")
        init_params1 = self.create_param_group(['initial_temp', 'initial_veg'], "model1")

        # Add parameter groups to stack
        self.param_stack1.addWidget(general_params1)
        self.param_stack1.addWidget(temp_params1)
        self.param_stack1.addWidget(albedo_params1)
        self.param_stack1.addWidget(veg_params1)
        self.param_stack1.addWidget(init_params1)

        # Button layout for switching parameter groups
        button_layout = QHBoxLayout()

        # Parameter group buttons
        button_configs = [
            ("General Settings", 0),
            ("Temperature Parameters", 1),
            ("Albedo Parameters", 2),
            ("Vegetation Parameters", 3),
            ("Initial Conditions", 4)
        ]

        for label, index in button_configs:
            button = QPushButton(label)
            button.setStyleSheet(
                "font-size: 16px; padding: 8px; margin: 5px; background-color: #3498db; color: white; border-radius: 8px;")
            button.clicked.connect(lambda checked, idx=index: self.param_stack1.setCurrentIndex(idx))
            button_layout.addWidget(button)

        sim_layout.addLayout(button_layout)
        sim_layout.addWidget(self.param_stack1)

        # Run button
        run_button = QPushButton("Run Simulation")
        run_button.setStyleSheet(
            "font-size: 22px; padding: 16px; margin-top: 20px; background-color: #e74c3c; color: white; border-radius: 8px;")
        run_button.clicked.connect(lambda: self.run_simulation("model1"))
        sim_layout.addWidget(run_button)

        # Back button
        back_button = QPushButton("Back to Model Selection")
        back_button.setStyleSheet(
            "font-size: 16px; padding: 10px; margin-top: 10px; background-color: #7f8c8d; color: white; border-radius: 8px;")
        back_button.clicked.connect(lambda: self.main_stack.setCurrentIndex(0))
        sim_layout.addWidget(back_button)

        # Add to main stack
        self.main_stack.addWidget(self.sim_screen1)

    def create_simulation_screen_model2(self):
        # Simulation screen for Model 2
        self.sim_screen2 = QWidget()
        sim_layout = QVBoxLayout(self.sim_screen2)

        # Title
        title = QLabel("Model 2: Alternative Temperature-Vegetation Coupling")
        title.setStyleSheet("font-size: 30px; font-weight: bold; text-align: center;")
        sim_layout.addWidget(title)

        # Description
        description = QLabel("Explore the system of equations (9) and (10) for temperature-vegetation dynamics.")
        description.setStyleSheet("font-size: 18px; margin-bottom: 20px; text-align: center;")
        sim_layout.addWidget(description)

        # Parameter categories
        self.param_stack2 = QStackedWidget()

        # Create parameter groups for Model 2
        general_params2 = self.create_param_group(['days', 'time_points'], "model2")
        temp_params2 = self.create_param_group(['a', 'r', 'b'], "model2")
        albedo_params2 = self.create_param_group(['s1', 's0', 'q'], "model2")
        veg_params2 = self.create_param_group(['b1_alt', 'T0_alt', 'a2', 'sigma_alt'], "model2")
        init_params2 = self.create_param_group(['initial_temp_alt', 'initial_veg_alt'], "model2")

        # Add parameter groups to stack
        self.param_stack2.addWidget(general_params2)
        self.param_stack2.addWidget(temp_params2)
        self.param_stack2.addWidget(albedo_params2)
        self.param_stack2.addWidget(veg_params2)
        self.param_stack2.addWidget(init_params2)

        # Button layout for switching parameter groups
        button_layout = QHBoxLayout()

        # Parameter group buttons
        button_configs = [
            ("General Settings", 0),
            ("Temperature Parameters", 1),
            ("Albedo Parameters", 2),
            ("Vegetation Parameters", 3),
            ("Initial Conditions", 4)
        ]

        for label, index in button_configs:
            button = QPushButton(label)
            button.setStyleSheet(
                "font-size: 16px; padding: 8px; margin: 5px; background-color: #3498db; color: white; border-radius: 8px;")
            button.clicked.connect(lambda checked, idx=index: self.param_stack2.setCurrentIndex(idx))
            button_layout.addWidget(button)

        sim_layout.addLayout(button_layout)
        sim_layout.addWidget(self.param_stack2)

        # Run button
        run_button = QPushButton("Run Simulation")
        run_button.setStyleSheet(
            "font-size: 22px; padding: 16px; margin-top: 20px; background-color: #e74c3c; color: white; border-radius: 8px;")
        run_button.clicked.connect(lambda: self.run_simulation("model2"))
        sim_layout.addWidget(run_button)

        # Back button
        back_button = QPushButton("Back to Model Selection")
        back_button.setStyleSheet(
            "font-size: 16px; padding: 10px; margin-top: 10px; background-color: #7f8c8d; color: white; border-radius: 8px;")
        back_button.clicked.connect(lambda: self.main_stack.setCurrentIndex(0))
        sim_layout.addWidget(back_button)

        # Add to main stack
        self.main_stack.addWidget(self.sim_screen2)

    def create_param_group(self, params, model_id):
        group = QWidget()
        layout = QFormLayout(group)

        # Create dictionaries to store input fields if they don't exist
        if not hasattr(self, 'inputs_model1'):
            self.inputs_model1 = {}
        if not hasattr(self, 'inputs_model2'):
            self.inputs_model2 = {}

        # Reference the correct dictionary
        inputs_dict = self.inputs_model1 if model_id == "model1" else self.inputs_model2

        # Create input fields
        for param in params:
            label = QLabel(f"{param} - {descriptions[param]}")
            label.setStyleSheet("font-size: 16px; margin-right: 10px;")

            entry = QLineEdit(str(default_values[param]))
            entry.setStyleSheet("font-size: 16px; padding: 6px;")

            # Store the input field
            inputs_dict[param] = entry

            layout.addRow(label, entry)

        return group

    def continue_to_simulation(self):
        if self.model1_radio.isChecked():
            self.main_stack.setCurrentIndex(1)  # Show Model 1 simulation screen
        else:
            self.main_stack.setCurrentIndex(2)  # Show Model 2 simulation screen

    def run_simulation(self, model_id):
        try:
            # Get parameters based on model
            params = {}

            # First add all default values
            for k in default_values:
                params[k] = default_values[k]

            # Then update with values from the input fields
            if model_id == "model1":
                for k, widget in self.inputs_model1.items():
                    try:
                        params[k] = float(widget.text())
                    except (ValueError, Exception):
                        # If there's an issue, use default value
                        params[k] = default_values[k]
            else:  # model2
                for k, widget in self.inputs_model2.items():
                    try:
                        params[k] = float(widget.text())
                    except (ValueError, Exception):
                        # If there's an issue, use default value
                        params[k] = default_values[k]

            # Time array for simulation
            t_span = (0, params['days'])
            t_eval = np.linspace(0, params['days'], int(params['time_points']))

            # Run the appropriate model
            if model_id == "model1":
                # Model 1 initial conditions
                y0 = [params['initial_temp'], params['initial_veg']]

                solution = solve_ivp(
                    lambda t, y: coupled_system_1(t, y, params),
                    t_span,
                    y0,
                    method='RK45',
                    t_eval=t_eval
                )

                # Extract results
                time = solution.t
                temp_solution = solution.y[0]
                veg_solution = solution.y[1]

                # Calculate the effective albedo over time for plotting
                effective_albedo = params['albedo_bare'] * (1 - veg_solution) + params['albedo_veg'] * veg_solution

                model_title = "Model 1: Temperature-Vegetation Albedo Feedback"

            else:  # model2
                # Model 2 initial conditions
                y0 = [params['initial_temp_alt'], params['initial_veg_alt']]

                # Solve the coupled system
                solution = solve_ivp(
                    lambda t, y: coupled_system_2(t, y, params),
                    t_span,
                    y0,
                    method='RK45',
                    t_eval=t_eval
                )

                # Extract results
                time = solution.t
                temp_solution = solution.y[0]
                veg_solution = solution.y[1]

                # Calculate the effective albedo over time for model 2
                effective_albedo = params['s0'] + (params['s1'] - params['s0']) * np.exp(-params['q'] * veg_solution)

                model_title = "Model 2: Alternative Temperature-Vegetation Coupling"

            # Create plots (common for both models)
            plt.figure(figsize=(15, 10))

            # Temperature plot
            plt.subplot(2, 2, 1)
            plt.plot(time, temp_solution, label='Temperature (K)', color='royalblue')
            plt.xlabel('Time (days)')
            plt.ylabel('Temperature (K)')
            plt.title('Temperature Evolution')
            plt.legend()
            plt.grid(True)

            # Vegetation plot
            plt.subplot(2, 2, 2)
            plt.plot(time, veg_solution, label='Vegetation Density', color='forestgreen')
            plt.xlabel('Time (days)')
            plt.ylabel('Density (0-1)')
            plt.title('Vegetation Density Evolution')
            plt.legend()
            plt.grid(True)

            # Albedo plot
            plt.subplot(2, 2, 3)
            plt.plot(time, effective_albedo, label='Effective Albedo', color='darkorange')
            plt.xlabel('Time (days)')
            plt.ylabel('Albedo')
            plt.title('Effective Surface Albedo')
            plt.legend()
            plt.grid(True)

            # Phase portrait
            plt.subplot(2, 2, 4)
            plt.plot(temp_solution, veg_solution, '-', color='purple')
            plt.plot(temp_solution[0], veg_solution[0], 'go', label='Initial state')
            plt.plot(temp_solution[-1], veg_solution[-1], 'ro', label='Final state')
            plt.xlabel('Temperature (K)')
            plt.ylabel('Vegetation Density')
            plt.title('Phase Portrait')
            plt.legend()
            plt.grid(True)

            # Common for both models
            plt.tight_layout()
            plt.suptitle(model_title + " Simulation Results", fontsize=16, y=1.02)
            plt.show()

        except ValueError as e:
            QMessageBox.critical(self, "Input Error", f"Please enter valid numbers. Error: {str(e)}")
        except Exception as e:
            QMessageBox.critical(self, "Simulation Error", f"An error occurred: {str(e)}")


if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = ClimateSimulation()
    window.show()
    sys.exit(app.exec_())