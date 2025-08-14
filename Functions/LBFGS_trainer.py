##################################   Continue Training with L-BFGS  Optimizer  ########################################

#Import stuff
import torch
import torch.nn as nn
import os

from UnNormalized_Plots import plot_losses, predict_and_plot, predict_and_quiver, plot_residuals, plot_velocity_profiles, validation_by_flux, print_boundary_summary, predict_and_plot_on_collocation
from Training_Montage import get_unique_run_folder
from The_Reporter import write_report, save_outputs
from PINN_Functions import domain, normalize_domain_outputs, PINN, PINN_branched, physics_loss, boundary_loss

#Move to GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Change these to match your run
input_path = r"C:\Users\nv1n24\OneDrive - University of Southampton\PhD\Research\PINNs Code\Functions\compare7.pth"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Network parameters
shared_layers = [2, 60, 60, 60, 60, 60, 60, 60, 60, 60, 3]
velocity_layers = [2]
pressure_layers = [1]
activation = nn.Tanh()
use_xavier = False
N_walls=1000
N_inlet=200
N_outlet=200
N_cylinder=400
N_collocation=49152
L_total=1.0
pipe_radius=0.2
cylinder_radius=0.05
cylinder_center=(0.2, 0.0)
margin=0.01
U_max = 1.0
Re = 20
learningrate =0.5
epoch_number = 3
sections = [
    ("Pipe", 0, 1.0)
]

domain_outputs = domain(N_walls=N_walls,
                                N_inlet=N_inlet,
                                N_outlet=N_outlet,
                                N_cylinder=N_cylinder,
                                N_collocation=N_collocation,
                                L_total=L_total,
                                pipe_radius=pipe_radius,
                                cylinder_radius=cylinder_radius,
                                cylinder_center=cylinder_center,
                                margin=margin)
    
x_y_top, x_y_bottom, x_y_inlet, x_y_outlet, x_y_cylinder, x_y_collocation, h_total, x_vals_total = domain_outputs

x_y_top_norm, x_y_bottom_norm, x_y_inlet_norm, x_y_outlet_norm, x_y_cylinder_norm, x_y_collocation_norm, \
    _, _, scale_factor = normalize_domain_outputs(domain_outputs, L_total, pipe_radius, cylinder_center, cylinder_radius)

h_total_norm = h_total * scale_factor
x_vals_total_norm = (x_vals_total - L_total / 2.0) * scale_factor

# Move to device
x_y_top = x_y_top.to(device)
x_y_bottom = x_y_bottom.to(device)
x_y_inlet = x_y_inlet.to(device)
x_y_outlet = x_y_outlet.to(device)
x_y_cylinder = x_y_cylinder.to(device)
x_y_collocation = x_y_collocation.to(device)
x_y_top_norm = x_y_top_norm.to(device)
x_y_bottom_norm = x_y_bottom_norm.to(device)
x_y_inlet_norm = x_y_inlet_norm.to(device)
x_y_outlet_norm = x_y_outlet_norm.to(device)
x_y_cylinder_norm = x_y_cylinder_norm.to(device)
x_y_collocation_norm = x_y_collocation_norm.to(device)

#### Load the pre-trained model 
model = PINN(shared_layers, activation, use_xavier).to(device)
checkpoint = torch.load(input_path, map_location=device)
model.load_state_dict(checkpoint['model_state_dict'])
epoch_start = checkpoint['epoch'] + 1  # Continue from last epoch
loss = checkpoint['loss_history_weighted']

epoch_end = epoch_start + epoch_number  # Train for X more epochs

new_loss_history = []
new_physics_loss_history = []
new_boundary_loss_history = []
new_loss_phys_x_history = []
new_loss_phys_y_history = []
new_loss_continuity_history = []
new_loss_bc_inlet_history = []
new_loss_bc_wall_history = []
new_loss_bc_outlet_history = []

model.train()

closure_calls = 0

def closure():
    global closure_calls
    closure_calls += 1
    optimizer.zero_grad(set_to_none=True)

    loss_phys_x, loss_phys_y, loss_continuity = physics_loss(model, x_y_collocation_norm, Re)
    loss_bc_inlet, loss_bc_wall, loss_bc_outlet = boundary_loss(model, x_y_inlet_norm, x_y_outlet_norm, x_y_top_norm, x_y_bottom_norm, x_y_cylinder_norm, U_max, pipe_radius, scale_factor)

    loss_phys = loss_phys_x + loss_phys_y + loss_continuity
    loss_bc = loss_bc_inlet + loss_bc_wall + loss_bc_outlet
    
    total_loss = loss_phys + loss_bc 

    total_loss.backward()

    return total_loss

script_name = "PINN_1_1_LBFGS"
run_folder = get_unique_run_folder(base_folder="Training Montages", name=script_name)

for epoch in range(epoch_start, epoch_end):

    optimizer = torch.optim.LBFGS(
                model.parameters(),
                        lr=learningrate,                    
                max_iter=250,             
                max_eval=None,             
                history_size=100,          
                #line_search_fn='strong_wolfe',  
                tolerance_grad=1e-5,
                tolerance_change=1e-9 
            )
    prev_params = torch.cat([p.detach().clone().flatten() for p in model.parameters()])

    closure_calls = 0   

    optimizer.step(closure)

    loss_phys_x, loss_phys_y, loss_continuity = physics_loss(model, x_y_collocation_norm, Re)
    loss_bc_inlet, loss_bc_wall, loss_bc_outlet = boundary_loss(model, x_y_inlet_norm, x_y_outlet_norm, x_y_top_norm, x_y_bottom_norm, x_y_cylinder_norm, U_max, pipe_radius, scale_factor)
    loss_phys = loss_phys_x + loss_phys_y + loss_continuity
    loss_bc = loss_bc_inlet + loss_bc_wall + loss_bc_outlet
    total_loss = loss_phys + loss_bc

    # Store logs
    new_loss_history.append(total_loss.item())
    new_physics_loss_history.append(loss_phys.item())
    new_boundary_loss_history.append(loss_bc.item())
    new_loss_phys_x_history.append(loss_phys_x.item())
    new_loss_phys_y_history.append(loss_phys_y.item())
    new_loss_continuity_history.append(loss_continuity.item())
    new_loss_bc_inlet_history.append(loss_bc_inlet.item())
    new_loss_bc_wall_history.append(loss_bc_wall.item())
    new_loss_bc_outlet_history.append(loss_bc_outlet.item())

    new_params = torch.cat([p.detach().clone().flatten() for p in model.parameters()])
    #print(torch.norm(prev_params - new_params))  # Should not be 0

    param_delta = torch.norm(prev_params - new_params)
    
    if epoch % 10 == 0:  # print every x epochs
        total_norm = 0.0
        for p in model.parameters():
            if p.grad is not None:
                total_norm += p.grad.data.norm(2).item() ** 2
        grad_norm = total_norm ** 0.5

        loss_total = new_loss_history[-1]
        loss_phys_x = new_loss_phys_x_history[-1]
        loss_phys_y = new_loss_phys_y_history[-1]
        loss_continuity = new_loss_continuity_history[-1]
        loss_bc_inlet = new_loss_bc_inlet_history[-1]
        loss_bc_wall = new_loss_bc_wall_history[-1]
        loss_bc_outlet = new_loss_bc_outlet_history[-1]

        print(f"Epoch {epoch} Gradient norm: {grad_norm:.2e}, Param delta: {param_delta:.3e}, Loss: {loss_total:.4e}")

        if epoch % 100 == 0:
            checkpoint_path = os.path.join(run_folder, f'checkpoint_epoch_{epoch}.pth')
            torch.save({
                        'epoch': epoch,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'loss_history': new_loss_history[-1]
                    }, checkpoint_path)   # save the model for later re-loading
            
#create a report folder and write the report
write_report(report_folder=run_folder,
filename=script_name,

####
N_collocation=N_collocation,
epoch_start=epoch_start,
epoch_end=epoch_end,
problem_description="2D flow around a cylinder in a channel.",

# Network 
model=model,
activation=activation,
use_xavier=use_xavier,

# Domain
N_walls=N_walls,
N_inlet=N_inlet,
N_outlet=N_outlet,
L_total=L_total,
H_ref= pipe_radius,
batch_size = None,
margin = margin,
domain_notes=f"Cylinder info: center at {cylinder_center}, number of boundary wall points {N_cylinder}.",

# Boundary conditions and loss
inlet_type="Velocity inlet",
U_max=U_max,
U_profile="Parabolic velocity profile: U_max * (1 - (y/h0)^2)",
wall_type="No-slip",
outlet_type="Pressure outlet",
mu=None,
rho=None,
Re=Re,
BC_Loss_notes="Loss  MSE.",

# Training
Optimizer="L-BFGS",
lr_shared=learningrate,
lr_velocity= None,
lr_pressure = None,
lr_drop_epoch=None,
gamma=None,
beta_1 = None,
beta_2 = None,
Training_notes="Scheduler: StepLR, gradient clipping.",
)
                
# save outputs
save_outputs(
run_folder,
[
    # Loss plots
    {
        'func': plot_losses,
        'args': ("Total Loss",),
        'kwargs': {"Total Loss": new_loss_history},
        'filename': "total_loss",
        'type': 'plot'
    },
    {
        'func': plot_losses,
        'args': ("Physics and Boundary Losses",),
        'kwargs': {
            "Physics Loss": new_physics_loss_history,
            "Boundary Loss": new_boundary_loss_history
        },
        'filename': "physics_boundary_losses",
        'type': 'plot'
    },
    {
        'func': plot_losses,
        'args': ("Physics-Based Losses",),
        'kwargs': {
            "X-Momentum (Physics)": new_loss_phys_x_history,
            "Y-Momentum (Physics)": new_loss_phys_y_history,
            "Continuity (Mass)": new_loss_continuity_history
        },
        'filename': "physics_losses",
        'type': 'plot'
    },
    {
        'func': plot_losses,
        'args': ("Boundary Condition Losses",),
        'kwargs': {
            "Inlet BC": new_loss_bc_inlet_history,
            "Wall BC": new_loss_bc_wall_history,
            "Outlet BC": new_loss_bc_outlet_history
        },
        'filename': "boundary_losses",
        'type': 'plot'
    },
    # Prediction plots
    {
        'func': predict_and_plot,
        'args': (model, device, sections, x_vals_total, h_total, L_total, scale_factor, cylinder_center, cylinder_radius, (5, 2), -0.6, 0.6, 100, False),
        'filename': "predict_and_plot",
        'type': 'plot'
    },
    {
        'func': predict_and_quiver,
        'args': (model, device, sections, scale_factor, L_total, x_vals_total, h_total, cylinder_center, cylinder_radius, (15, 7), 100, 1, None, None, -0.6, 0.6, 200, None),
        'filename': "predict_and_quiver",
        'type': 'plot'
    },
    {
        'func': plot_residuals,
        'args': (model, device, Re, scale_factor, x_vals_total, h_total, cylinder_center, cylinder_radius, (0,1), (-0.2, 0.2), (7, 3), 200, 100),
        'filename': "residuals",
        'type': 'plot'
    },
    {
        'func': plot_velocity_profiles,
        'args': (model, device, cylinder_center, cylinder_radius, x_vals_total, h_total, x_vals_total_norm, h_total_norm, L_total, x_y_inlet_norm, x_y_outlet_norm, scale_factor, 1, [0.25, 0.5, 0.75], (8, 5)),
        'filename': "velocity_profiles",
        'type': 'plot'
    },
    {
        'func': predict_and_plot_on_collocation,
        'args': (model, device, x_y_collocation_norm, x_y_inlet_norm, x_y_outlet_norm, x_y_top_norm, x_y_bottom_norm, x_y_cylinder_norm, 0, 3, -50, 50, (6, 2)),
        'filename': "predict_and_plot_collocation",
        'type': 'plot'
    },
    # Text outputs
    {
        'func': validation_by_flux,
        'args': (model, x_vals_total, x_vals_total_norm, h_total, h_total_norm, U_max, scale_factor),
        'filename': "flux_info.txt",
        'type': 'text'
    },
    {
        'func': print_boundary_summary,
        'args': (model, x_y_inlet_norm, x_y_outlet_norm, x_y_top_norm, x_y_bottom_norm, scale_factor),
        'filename': "boundary_summary.txt",
        'type': 'text'
    }
])

