"""
This script 
"""
import os
import matplotlib.pyplot as plt
from contextlib import redirect_stdout

def create_report_folder(filename):
    """
    Creates a folder named after the filename (without extension) inside a 'Reports' folder
    in the current working directory. If the folder exists, appends _1, _2, etc. until a unique name is found.
    Returns the full path to the new folder.
    """
    reports_dir = os.path.join(os.getcwd(), "Reports")
    os.makedirs(reports_dir, exist_ok=True)
    base_folder_name = os.path.splitext(filename)[0]
    folder_name = base_folder_name
    report_folder = os.path.join(reports_dir, folder_name)
    counter = 1
    while os.path.exists(report_folder):
        folder_name = f"{base_folder_name}_{counter}"
        report_folder = os.path.join(reports_dir, folder_name)
        counter += 1
    os.makedirs(report_folder)
    return report_folder

def write_report(filename,
                N_collocation,
                epoch_start,
                epoch_end,
                problem_description="",
                layer_number=None,
                layer_size=None,
                activation=None,
                network_notes="",
                N_walls=None,
                N_inlet=None,
                N_outlet=None,
                L_total=None,
                H_ref=None,
                domain_notes="",
                inlet_type=None,
                U_max=None,
                U_profile=None,
                wall_type=None,
                outlet_type=None,
                mu=None,
                rho=None,
                Re=None,
                BC_Loss_notes="",
                Optimizer=None,
                initial_lr=None,
                lr_drop_epoch=None,
                gamma=None,
                lambda_x_mom=None,
                lambda_y_mom=None,
                lambda_cont=None,
                lambda_wall=None,
                lambda_outlet=None,
                lambda_inlet=None,
                Training_notes="",
                **kwargs
                 ):
    """
    Write a summary report text file of the PINN training run.

    Some Parameters:
    For network notes, use the `network_notes` variable to include any additional information such as 
                                input normalization, weight initialization, pressure branching, etc.
    For domain notes, use the `domain_notes` variable to include any additional information such as
                                the cylinder details.
    For BC and Loss notes, use the `BC_Loss_notes` variable to include any additional information about the 
                                boundary conditions or about the loss (which functions were used, etc.).
    For training notes, use the `Training_notes` variable to include any additional information about the training
                                process (e.g., optimization settings, learning rate schedule, etc.).
    The `kwargs` dictionary can be used to include any other notes you want to add to the report.
    """
    report_folder = create_report_folder(filename)
    report_path = os.path.join(report_folder, filename)

    with open(report_path, "w") as f:
        f.write("PINN Report\n")
        f.write(f"The problem: {problem_description}\n\n")


        f.write("===================================\n")
        f.write("Network Configuration\n")
        f.write(f"Number of hidden layers: {layer_number}\n")
        f.write(f"Number of neurons per hidden layer: {layer_size}\n")
        f.write(f"Activation function: {activation}\n")
        f.write("Other notes:")
        f.write(f"{network_notes}")


        f.write("\n===================================\n")
        f.write("The Domain\n")
        f.write(f"Number of collocation points: {N_collocation}\n")
        f.write(f"Wall Boundary points: {N_walls}\n")
        f.write(f"Inlet Boundary points: {N_inlet}\n")
        f.write(f"Outlet Boundary points: {N_outlet}\n")

        f.write(f"The total length: {L_total}\n")
        f.write(f"The inlet height: {H_ref}\n")
        f.write("Other notes:")
        f.write(f"{domain_notes}")


        f.write("\n===================================\n")
        f.write("The Boundary Conditions and Loss\n")
        f.write(f"Inlet Boundary Type: {inlet_type}\n")
        f.write(f"Maximum Velocity: {U_max}\n")
        f.write(f"Velocity Profile: {U_profile}\n")
        f.write(f"Wall Boundary Type: {wall_type}\n")
        f.write(f"Outlet Boundary Type: {outlet_type}\n")
        f.write(f"Dynamic Viscosity: {mu}\n")
        f.write(f"Density of Fluid: {rho}\n")
        f.write(f"Reynolds number: {Re}\n")        
        f.write("Other notes:")
        f.write(f"{BC_Loss_notes}")

        f.write("\n===================================\n")
        f.write("Training Summary\n")
        f.write(f"Epochs: {epoch_start} to {epoch_end-1} (total: {epoch_end-epoch_start})\n")
        f.write(f"Optimizer: {Optimizer}\n")
        f.write(f"Initial learning rate: {initial_lr}\n")
        f.write(f"Scheduled Learning rate drop: {lr_drop_epoch}\n")
        f.write(f"Scheduler gamma: {gamma}\n")
        f.write(f"X-Momentum Loss Lambda: {lambda_x_mom}\n")
        f.write(f"Y-Momentum Loss Lambda: {lambda_y_mom}\n")
        f.write(f"Continuity Loss Lambda: {lambda_cont}\n")
        f.write(f"Wall Boundary Loss Lambda: {lambda_wall}\n")
        f.write(f"Outlet Loss Lambda: {lambda_outlet}\n")
        f.write(f"Inlet Loss Lambda: {lambda_inlet}\n")
        f.write("Other notes:")
        f.write(f"{Training_notes}\n")


        f.write("\n===================================\n")
        f.write("Any other notes at all?\n")
        for key, value in kwargs.items():
            f.write(f"{key}: {value}\n")
    print(f"Report written to {filename}")

    print(f"Report written to {report_path}")
    return report_folder


def save_outputs(folder, calls):
    """
    Calls plotting or text-output functions and saves their outputs to the specified folder.

    Args:
        folder (str): Path to the folder where outputs will be saved.
        calls (list): List of dicts, each with:
            {
                'func': function,
                'args': tuple of positional arguments,
                'kwargs': dict of keyword arguments,
                'filename': filename for the saved file (without extension for plots, with .txt for text),
                'type': 'plot' or 'text'
            }
    """

    os.makedirs(folder, exist_ok=True)
    for call in calls:
        func = call['func']
        args = call.get('args', ())
        kwargs = call.get('kwargs', {})
        filename = call.get('filename', func.__name__)
        out_type = call.get('type', 'plot')
        if out_type == 'plot':
            result = func(*args, **kwargs)
            if isinstance(result, tuple):
                for i, fig in enumerate(result):
                    if hasattr(fig, 'savefig'):
                        fig.savefig(os.path.join(folder, f"{filename}_{i+1}.png"))
                        plt.close(fig)
            elif hasattr(result, 'savefig'):
                result.savefig(os.path.join(folder, f"{filename}.png"))
                plt.close(result)
        elif out_type == 'text':
            file_path = os.path.join(folder, filename if filename.endswith('.txt') else filename + ".txt")
            with open(file_path, "w") as f, redirect_stdout(f):
                func(*args, **kwargs)