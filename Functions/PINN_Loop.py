##################################   The Training Loop   ########################################

#Import stuff
import torch
import torch.optim as optim
import matplotlib.pyplot as plt
import os

from UnNormalized_Plots import plot_losses, predict_and_plot, predict_and_quiver, plot_residuals, plot_velocity_profiles, validation_by_flux, print_boundary_summary, predict_and_plot_on_collocation
from Training_Montage import get_unique_run_folder, save_figure, make_video
from The_Reporter import write_report, save_outputs
from PINN_Functions import domain, physics_loss, boundary_loss, normalize_domain_outputs, PINN_branched, PINN

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train_pinn_with_Adam(model_name, shared_layers, velocity_layers, pressure_layers, activation, use_xavier,
                        N_walls, N_inlet, N_outlet, N_cylinder, N_collocation, L_total, pipe_radius, cylinder_radius, cylinder_center, margin, 
                        lr_shared, lr_velocity, lr_pressure, beta_1, beta_2, lr_drop_epoch, gamma,
                        epoch_start, epoch_end, batch_size, Re, U_max, sections,
                        lambda_x_mom, lambda_y_mom, lambda_cont,
                        lambda_inlet, lambda_outlet, lambda_wall
                        ):
    
    ########################### Name this script and set up the run folder ###########################
    script_name = "PINN_1_1_Adam"
    run_folder = get_unique_run_folder(base_folder="Training Montages", name=script_name)
    figure_folder = get_unique_run_folder(base_folder=run_folder, name="Figures")
    print(f"Saving in Run folder: {run_folder}")

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

    x_y_top_norm, x_y_bottom_norm, x_y_inlet_norm, x_y_outlet_norm, x_y_cylinder_norm, x_y_collocation_norm, cylinder_center_norm, cylinder_radius_norm, \
                                            scale_factor = normalize_domain_outputs(domain_outputs, L_total, pipe_radius, cylinder_center, cylinder_radius)

    h_total_norm = h_total * scale_factor
    x_vals_total_norm = (x_vals_total - L_total / 2.0) * scale_factor

    # Move to GPU
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
        
    if model_name == "PINN":
        print("Using single model for training")
        model = PINN(shared_layers, activation, use_xavier).to(device)
        optimizer = optim.Adam(model.parameters(), lr=lr_shared, betas=(beta_1, beta_2))

    elif model_name == "PINN_branched":
        model = PINN_branched(shared_layers, velocity_layers, pressure_layers, activation, use_xavier).to(device)
        print("Using branched model for training")
        optimizer = optim.Adam([
                                    {'params': model.shared.parameters(), 'lr': lr_shared},           # Shared layers
                                    {'params': model.velocity_branch.parameters(), 'lr': lr_velocity},  # Velocity branch
                                    {'params': model.pressure_branch.parameters(), 'lr': lr_pressure}  # Pressure branch 
                                ], betas=(beta_1, beta_2))

    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=lr_drop_epoch, gamma=gamma)

    num_collocation_points = x_y_collocation_norm.shape[0]

    loss_history = {
    'total': [],
    'physics': [],
    'boundary': [],
    'loss_phys_x': [],
    'loss_phys_y': [],
    'loss_continuity': [],
    'loss_bc_inlet': [],
    'loss_bc_wall': [],
    'loss_bc_outlet': []    }

    print(model)
    
    for epoch in range(epoch_start,epoch_end):
        # Shuffle collocation points at the start of each epoch (only the collocation points get batches, the boundary points are fixed)
        perm = torch.randperm(num_collocation_points)
        x_y_collocation_shuffled = x_y_collocation_norm[perm]

        # Initialize accumulators for batch losses
        total_loss_epoch = 0.0
        loss_phys_epoch = 0.0
        loss_bc_epoch = 0.0
        loss_phys_x_epoch = 0.0
        loss_phys_y_epoch = 0.0
        loss_continuity_epoch = 0.0
        loss_bc_inlet_epoch = 0.0
        loss_bc_wall_epoch = 0.0
        loss_bc_outlet_epoch = 0.0

        num_batches = 0

        for batch_start in range(0, num_collocation_points, batch_size):
            batch_end = min(batch_start + batch_size, num_collocation_points)
            x_y_collocation_batch = x_y_collocation_shuffled[batch_start:batch_end].clone().detach().requires_grad_(True)

            optimizer.zero_grad()

            loss_phys_x, loss_phys_y, loss_continuity = physics_loss(model, x_y_collocation_batch, Re)
            loss_bc_inlet, loss_bc_wall, loss_bc_outlet = boundary_loss(model, x_y_inlet_norm, x_y_outlet_norm, x_y_top_norm, x_y_bottom_norm, x_y_cylinder_norm, U_max, pipe_radius, scale_factor)

            loss_phys_x = loss_phys_x * lambda_x_mom
            loss_phys_y = loss_phys_y * lambda_y_mom
            loss_continuity = loss_continuity * lambda_cont
            loss_bc_inlet = loss_bc_inlet * lambda_inlet
            loss_bc_wall = loss_bc_wall * lambda_wall
            loss_bc_outlet = loss_bc_outlet * lambda_outlet

            # Combine all losses 
            loss_phys = loss_phys_x + loss_phys_y + loss_continuity
            loss_bc = loss_bc_inlet + loss_bc_wall + loss_bc_outlet
            total_loss = loss_phys + loss_bc 

            if model_name == "PINN":
                total_loss.backward() 

            elif model_name == "PINN_branched":
                loss_common = loss_phys_x + loss_phys_y
                loss_common.backward(retain_graph=True)
                loss_velocity = loss_bc_inlet + loss_bc_wall + loss_continuity
                velocity_params = list(model.shared.parameters()) + list(model.velocity_branch.parameters())
                loss_velocity.backward(retain_graph=True, inputs=velocity_params)
                pressure_params = list(model.shared.parameters()) + list(model.pressure_branch.parameters())
                loss_bc_outlet.backward(inputs=pressure_params)

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            # Accumulate batch losses
            total_loss_epoch += total_loss.item()
            loss_phys_epoch += loss_phys.item()
            loss_bc_epoch += loss_bc.item()
            loss_phys_x_epoch += loss_phys_x.item()
            loss_phys_y_epoch += loss_phys_y.item()
            loss_continuity_epoch += loss_continuity.item()
            loss_bc_inlet_epoch += loss_bc_inlet.item()
            loss_bc_wall_epoch += loss_bc_wall.item()
            loss_bc_outlet_epoch += loss_bc_outlet.item()
            num_batches += 1
            
        scheduler.step()

        # Average losses over all batches in this epoch and append to history
        loss_history['total'].append(total_loss_epoch / num_batches)
        loss_history['physics'].append(loss_phys_epoch / num_batches)
        loss_history['boundary'].append(loss_bc_epoch / num_batches)
        loss_history['loss_phys_x'].append(loss_phys_x_epoch / num_batches)
        loss_history['loss_phys_y'].append(loss_phys_y_epoch / num_batches)
        loss_history['loss_continuity'].append(loss_continuity_epoch / num_batches)
        loss_history['loss_bc_inlet'].append(loss_bc_inlet_epoch / num_batches)
        loss_history['loss_bc_wall'].append(loss_bc_wall_epoch / num_batches)
        loss_history['loss_bc_outlet'].append(loss_bc_outlet_epoch / num_batches)

        if epoch % 500 == 0:
            fig1, fig2 = predict_and_plot(model, device, sections, x_vals_total, h_total, L_total, scale_factor, cylinder_center, cylinder_radius, 
                     fig_sizes=(5, 2), ymin=-0.6, ymax=0.6, y_density=100, grid_onoff=False)
            save_figure(fig1, epoch, prefix="velocity", folder=figure_folder)
            save_figure(fig2, epoch, prefix="pressure", folder=figure_folder)

            if epoch % 5000 == 0:
                print(f"Epoch {epoch} | Total Loss: {total_loss_epoch/num_batches:.4e} | Phys: {loss_phys_epoch/num_batches:.4e} | BC: {loss_bc_epoch/num_batches:.4e}")
                checkpoint_path = os.path.join(run_folder, f'checkpoint_epoch_{epoch}.pth')
                torch.save({
                            'epoch': epoch,
                            'model_state_dict': model.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict(),
                            'scheduler_state_dict': scheduler.state_dict(),
                            'loss_history': loss_history,
                        }, checkpoint_path)   # save the model for later re-loading
                
            plt.close(fig1)
            plt.close(fig2)

    make_video(input_folder=figure_folder, output_folder=run_folder, output="velocity_montage.mp4", fps=10, prefix="velocity")
    make_video(input_folder=figure_folder, output_folder=run_folder, output="pressure_montage.mp4", fps=10, prefix="pressure")
    
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
    batch_size = batch_size,
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
    Optimizer="Adam",
    lr_shared=lr_shared,
    lr_velocity= lr_velocity,
    lr_pressure = lr_pressure,
    lr_drop_epoch=lr_drop_epoch,
    gamma=gamma,
    beta_1 = beta_1,
    beta_2 = beta_2,
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
            'kwargs': {"Total Loss": loss_history['total']},
            'filename': "total_loss",
            'type': 'plot'
        },
        {
            'func': plot_losses,
            'args': ("Physics and Boundary Losses",),
            'kwargs': {
                "Physics Loss": loss_history['physics'],
                "Boundary Loss": loss_history['boundary']
            },
            'filename': "physics_boundary_losses",
            'type': 'plot'
        },
        {
            'func': plot_losses,
            'args': ("Physics-Based Losses",),
            'kwargs': {
                "X-Momentum (Physics)": loss_history['loss_phys_x'],
                "Y-Momentum (Physics)": loss_history['loss_phys_y'],
                "Continuity (Mass)": loss_history['loss_continuity']
            },
            'filename': "physics_losses",
            'type': 'plot'
        },
        {
            'func': plot_losses,
            'args': ("Boundary Condition Losses",),
            'kwargs': {
                "Inlet BC": loss_history['loss_bc_inlet'],
                "Wall BC": loss_history['loss_bc_outlet'],
                "Outlet BC": loss_history['loss_bc_wall']
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

