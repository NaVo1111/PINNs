import matplotlib.pyplot as plt
import numpy as np
import torch

def plot_losses(title, **loss_histories):
    """
    Plots any number of loss histories on a single log-scale plot.

    Args:
        title (str): The title for the plot.
        **loss_histories: Keyword arguments where the key is the desired
                          label for the plot (e.g., "Physics Loss") and the
                          value is the list or array of loss values.

    Example usage:
        plot_losses("Boundary Condition Losses",
                    **{
                        "Inlet BC": loss_bc_inlet_history,
                        "Wall BC": loss_bc_wall_history,
                        "Outlet BC": loss_bc_outlet_history
                    })
    """
    fig, ax = plt.subplots(figsize=(8, 4))

    for label, history in loss_histories.items():
        ax.plot(history, label=label)

    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_yscale('log')
    ax.set_title(title)
    ax.legend()
    ax.grid(True, which="both", ls="--")
    fig.tight_layout()

    return fig

def predict_and_plot(model, device, sections, x_vals_total, h_total,  min_vel, max_vel, min_pres, max_pres, ymin=-0.6, ymax=0.6, y_density=100):
    """
    Predicts velocity and pressure using the trained model and plots the results.

    Args:
        model (torch.nn.Module): The trained model for prediction.
        device (torch.device): The device to run the model on (CPU or GPU).
        sections (list): List of tuples containing section names and x_start, x_end values.
            Example: sections = [("Pipe", 0.0, 3.0), ("Nozzle", 3.0, 5.0)]
        x_vals_total (np.ndarray): Array of x values for the nozzle profile.
        h_total (np.ndarray): Array of height values for the nozzle profile.
        min_vel (float): Minimum value for velocity color scale.
        max_vel (float): Maximum value for velocity color scale.
        min_pres (float): Minimum value for pressure color scale.
        max_pres (float): Maximum value for pressure color scale.
        ymin (float): Minimum y value for the plot. Default is -0.6.
        ymax (float): Maximum y value for the plot. Default is 0.6. 
        y_density (int): Number of points in the y direction for the meshgrid. Default is 100.

    Returns:
        tuple: Figures for velocity and pressure plots.
    """
    
    
    y_vals = np.linspace(ymin, ymax, y_density)
    fig1, ax1 = plt.subplots(figsize=(5, 2))
    fig2, ax2 = plt.subplots(figsize=(5, 2))

    all_contours_v = []
    all_contours_p = []

    for name, x_start, x_end in sections:
        x_section = np.linspace(x_start, x_end, 200)
        x_grid, y_grid = np.meshgrid(x_section, y_vals)
        x_flat = x_grid.flatten()
        y_flat = y_grid.flatten()

        h_section = np.interp(x_flat, x_vals_total, h_total)
        mask = np.abs(y_flat) <= h_section

        x_valid = x_flat[mask]
        y_valid = y_flat[mask]

        input_tensor = torch.tensor(np.stack([x_valid, y_valid], axis=1), dtype=torch.float32).to(device)
        with torch.no_grad():
            pred = model(input_tensor)
            u = pred[:, 0].cpu().numpy()
            v = pred[:, 1].cpu().numpy()
            vel_mag = np.sqrt(u**2 + v**2)

            pressure = pred[:, 2].cpu().numpy()

        # Add tricontourf with fixed color range for velocity
        contour_v = ax1.tricontourf(x_valid, y_valid, vel_mag, levels=100, cmap='rainbow', vmin=min_vel, vmax=max_vel)
        all_contours_v.append(contour_v)

        # Add tricontourf with fixed color range for pressure
        contour_p = ax2.tricontourf(x_valid, y_valid, pressure, levels=100, cmap='bwr', vmin=min_pres, vmax=max_pres)
        all_contours_p.append(contour_p)

    # Overlay nozzle walls
    ax1.plot(x_vals_total, h_total, 'k-', linewidth=2)
    ax1.plot(x_vals_total, -h_total, 'k-', linewidth=2)

    ax1.set_title("Predicted Velocity Magnitude")
    ax1.set_xlabel("x [m]")
    ax1.set_ylabel("y [m]")
    ax1.axis("equal")
    ax1.grid(True)

    # Use the first contour for the colorbar
    plt.colorbar(all_contours_v[0], ax=ax1, label="Velocity Magnitude [m/s]")

    # Overlay nozzle walls
    ax2.plot(x_vals_total, h_total, 'k-', linewidth=2)
    ax2.plot(x_vals_total, -h_total, 'k-', linewidth=2)

    ax2.set_title("Predicted Pressure")
    ax2.set_xlabel("x [m]")
    ax2.set_ylabel("y [m]")
    ax2.axis("equal")
    ax2.grid(True)

    # Use the first contour for the colorbar
    plt.colorbar(all_contours_p[0], ax=ax2, label="Pressure [Pa]")

    return fig1, fig2

def predict_and_quiver(model, device, sections, x_vals_total, h_total, density=100, scale=1, xlim=None, ylim=None, ymin=-0.6, ymax=0.6, y_density=200):
    """
    Predicts velocity vectors using the trained model and plots them as a quiver plot.  

    Args:   
        model (torch.nn.Module): The trained model for prediction.
        device (torch.device): The device to run the model on (e.g., 'cpu' or 'cuda').
        sections (list): List of tuples containing section names and x_start, x_end values.
            Example: sections = [("Pipe", 0.0, 3.0), ("Nozzle", 3.0, 5.0)]
        x_vals_total (np.ndarray): Array of x values for the profile.
        h_total (np.ndarray): Array of height values for the profile.
        density (int): Number of arrows per section for quiver density.
        scale (float): Scale factor for quiver arrows.  
        xlim (tuple): Optional x-axis limits for the plot.
        ylim (tuple): Optional y-axis limits for the plot.
        ymin (float): Minimum y value for the plot. Default is -0.6.
        ymax (float): Maximum y value for the plot. Default is 0.6.
        y_density (int): Number of points in the y direction for the meshgrid. Default is 200.

    Returns:
        Figure object containing the quiver plot.

    """

    y_vals = np.linspace(ymin, ymax, y_density)
    fig, ax = plt.subplots(figsize=(15, 7))

    for name, x_start, x_end in sections:
        x_section = np.linspace(x_start, x_end, 200)
        x_grid, y_grid = np.meshgrid(x_section, y_vals)
        x_flat = x_grid.flatten()
        y_flat = y_grid.flatten()

        h_section = np.interp(x_flat, x_vals_total, h_total)
        mask = np.abs(y_flat) <= h_section

        x_valid = x_flat[mask]
        y_valid = y_flat[mask]

        input_tensor = torch.tensor(np.stack([x_valid, y_valid], axis=1), dtype=torch.float32).to(device)
        with torch.no_grad():
            pred = model(input_tensor)
            u = pred[:, 0].cpu().numpy()
            v = pred[:, 1].cpu().numpy()

        # Downsample for quiver density
        stride = max(1, len(x_valid) // density)

        ax.quiver(
            x_valid[::stride],
            y_valid[::stride],
            u[::stride],
            v[::stride],
            angles='xy',
            scale_units='xy',
            scale=scale,
            width=0.002,
            color='black',
            alpha=0.8
        )

    # Overlay nozzle walls
    ax.plot(x_vals_total, h_total, 'r-', linewidth=2)
    ax.plot(x_vals_total, -h_total, 'r-', linewidth=2)

    ax.set_title("Predicted Velocity Vectors (Quiver Plot)")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.axis("equal")
    if xlim is not None:
        ax.set_xlim(*xlim)
    if ylim is not None:
        ax.set_ylim(*ylim)
    ax.grid(True)

    plt.tight_layout()

    return fig

def plot_residuals(model, device, mu, x_range, y_range, x_points, y_points,
                   x_y_inlet=None, x_y_outlet=None, x_y_top=None, x_y_bottom=None):
    """
    Plots residuals (momentum x, momentum y, continuity) and boundary results for a given model.

    Args:
        model: Trained PINN model.
        device: Device to run the model on (e.g., 'cpu' or 'cuda').
        x_range: Tuple (min_x, max_x) for plotting domain in x.
        y_range: Tuple (min_y, max_y) for plotting domain in y.
        x_points: Number of points in x direction.
        y_points: Number of points in y direction.
        x_y_inlet, x_y_outlet, x_y_top, x_y_bottom: Boundary coordinate tensors.
        L_total: Total length for outlet plot (required for velocity profile plot).

    Returns:
        Displays plots of residuals and boundary results.
    """
    # Create grid
    x = torch.linspace(x_range[0], x_range[1], x_points)
    y = torch.linspace(y_range[0], y_range[1], y_points)
    X, Y = torch.meshgrid(x, y, indexing='ij')
    x_y = torch.stack([X.flatten(), Y.flatten()], dim=1).to(device)
    x_y.requires_grad_(True)

    out = model(x_y)
    u = out[:, 0:1]
    v = out[:, 1:2]
    p = out[:, 2:3]

    grads = lambda f, wrt: torch.autograd.grad(f, wrt, grad_outputs=torch.ones_like(f), create_graph=True)[0]

    du_dx = grads(u, x_y)[:, 0:1]
    du_dy = grads(u, x_y)[:, 1:2]
    dv_dx = grads(v, x_y)[:, 0:1]
    dv_dy = grads(v, x_y)[:, 1:2]
    dp_dx = grads(p, x_y)[:, 0:1]
    dp_dy = grads(p, x_y)[:, 1:2]

    d2u_dx2 = grads(du_dx, x_y)[:, 0:1]
    d2u_dy2 = grads(du_dy, x_y)[:, 1:2]
    d2v_dx2 = grads(dv_dx, x_y)[:, 0:1]
    d2v_dy2 = grads(dv_dy, x_y)[:, 1:2]

    momentum_x = mu * (d2u_dx2 + d2u_dy2) - u * du_dx - v * du_dy + dp_dx
    momentum_y = mu * (d2v_dx2 + d2v_dy2) - u * dv_dx - v * dv_dy + dp_dy
    continuity = du_dx + dv_dy

    def plot_residual(field, title):
        fig, ax = plt.subplots(figsize=(6, 2))
        tcf = ax.tricontourf(x_y[:, 0].detach().cpu(), x_y[:, 1].detach().cpu(), field.detach().cpu().flatten(), levels=100, cmap='RdBu')
        plt.colorbar(tcf, ax=ax, label='Residual')
        ax.set_title(title)
        ax.set_xlabel('x [m]')
        ax.set_ylabel('y [m]')
        fig.tight_layout()
        return fig

    fig_mx = plot_residual(momentum_x, "Momentum x-Residual")
    fig_my = plot_residual(momentum_y, "Momentum y-Residual")
    fig_cont = plot_residual(continuity, "Continuity Residual")

    return fig_mx, fig_my, fig_cont



def print_boundary_summary(model, x_y_inlet, x_y_outlet, x_y_top, x_y_bottom):
    """
    Prints summary statistics for boundary velocities and outlet pressure for a given model.

    Args:
        model (torch.nn.Module): Trained PINN model.
        x_y_inlet (torch.Tensor): Tensor of inlet coordinates (x, y).
        x_y_outlet (torch.Tensor): Tensor of outlet coordinates (x, y).
        x_y_top (torch.Tensor): Tensor of top wall coordinates (x, y).
        x_y_bottom (torch.Tensor): Tensor of bottom wall coordinates (x, y).

    Returns:
        None. Prints the maximum velocity at the inlet, top wall, and bottom wall,
        and the maximum pressure at the outlet.
    """
    pred_inlet = model(x_y_inlet)
    pred_outlet = model(x_y_outlet)
    pred_top = model(x_y_top)
    pred_bottom = model(x_y_bottom)

    # Velocity magnitude for inlet, top, bottom
    vel_inlet = torch.sqrt(pred_inlet[:, 0]**2 + pred_inlet[:, 1]**2)
    vel_top = torch.sqrt(pred_top[:, 0]**2 + pred_top[:, 1]**2)
    vel_bottom = torch.sqrt(pred_bottom[:, 0]**2 + pred_bottom[:, 1]**2)

    # Pressure at outlet
    pressure_outlet = pred_outlet[:, 2]

    print('Inlet max velocity:', vel_inlet.max().item())
    print('Outlet max pressure:', pressure_outlet.max().item())
    print('Wall top max velocity:', vel_top.max().item())
    print('Wall bottom max velocity:', vel_bottom.max().item())

    

def plot_velocity_profiles(model, device, x_vals_total, h_total, L_total, x_y_inlet, x_y_outlet, x_locs, scale=1):
    """
    Plots velocity profiles at specified cross-sections.

    Args:
        model (torch.nn.Module): The trained model for prediction.
        device (torch.device): The device to run the model on (e.g., 'cpu' or 'cuda').
        x_vals_total (np.ndarray): Array of x values for the profile.
        h_total (np.ndarray): Array of height values for the profile.
        L_total (float): Total length of the profile for outlet plotting.
        x_y_inlet (torch.Tensor): Tensor of inlet coordinates (x, y).
        x_y_outlet (torch.Tensor): Tensor of outlet coordinates (x, y).
        x_locs (float): Location along the x-axis to plot velocity profiles.
        scale (float): Scale factor for the velocity profiles. Default is 1.

    Returns:
        plt: Matplotlib figure object containing the velocity profiles.
    """

    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot shape
    ax.plot(x_vals_total, h_total, 'k-', label='_nolegend_', linewidth=2)
    ax.plot(x_vals_total, -h_total, 'k-', label='_nolegend_', linewidth=2)

    # Extract inlet predictions and coordinates
    u_inlet_pred = model(x_y_inlet).detach().cpu().numpy()[:, 0] * scale
    y_inlet = x_y_inlet[:, 1].detach().cpu().numpy()

    # Shift the inlet profile to start at x = 0
    x_inlet_plot = u_inlet_pred + u_inlet_pred.min()

    # Plot at x = 0
    ax.plot(x_inlet_plot, y_inlet, 'r--', label='Predicted Inlet', linewidth=2)

    # Extract outlet predictions and coordinates
    u_outlet_pred = model(x_y_outlet).detach().cpu().numpy()[:, 0] * scale
    y_outlet = x_y_outlet[:, 1].detach().cpu().numpy()

    # Plot outlet velocity profile at x = L_total
    ax.plot(u_outlet_pred + L_total, y_outlet, 'g-', label='Predicted Outlet', linewidth=2)

    # --- Add plots at x_locs---
    for x_loc in x_locs:
        h_loc = np.interp(x_loc, x_vals_total, h_total)
        y_vals_loc = torch.linspace(-h_loc, h_loc, 100).view(-1, 1).to(device)
        x_vals_loc = torch.full_like(y_vals_loc, x_loc)
        xy_loc = torch.cat([x_vals_loc, y_vals_loc], dim=1)

        with torch.no_grad():
            u_pred_loc = model(xy_loc)[:, 0].cpu().numpy() * scale

        ax.plot(x_loc + u_pred_loc, y_vals_loc.cpu().numpy(), 'b-', label=f'Predicted x={x_loc}', linewidth=2)

    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_title("Velocity Profiles at Different Cross-sections")
    ax.grid(True)
    ax.axis("equal")
    ax.legend()
    return fig

def validation_by_flux(model, x_vals_total, h_total, L_total):
    """
    Computes and prints the volumetric flow rate at the inlet and outlet.

    Args:
        model: Trained PINN model.
        x_vals_total: Array of x values for the nozzle profile.
        h_total: Array of height values for the nozzle profile.
        L_total: Total length of the nozzle/domain.

    Returns:
        None: Prints the volumetric flow rates at the inlet and outlet, and their relative difference.
    """
    x_inlet = 0.0
    x_outlet = L_total

    h_inlet = np.interp(x_inlet, x_vals_total, h_total)
    h_outlet = np.interp(x_outlet, x_vals_total, h_total)

    def compute_flux(model, x_fixed, y_min, y_max, N=100):
        y_vals = torch.linspace(y_min, y_max, N).view(-1, 1)
        x_vals = torch.full_like(y_vals, x_fixed)
        xy = torch.cat([x_vals, y_vals], dim=1).to(next(model.parameters()).device)

        with torch.no_grad():
            u_pred = model(xy)[:, 0:1]  # horizontal velocity
            dy = (y_max - y_min) / (N - 1)
            flux = torch.sum(u_pred) * dy
        return flux.item()

    Q_in = compute_flux(model, x_fixed=x_inlet, y_min=-h_inlet, y_max=h_inlet)
    Q_out = compute_flux(model, x_fixed=x_outlet, y_min=-h_outlet, y_max=h_outlet)

    print(f"Volumetric flow rate at inlet:  {Q_in:.6f}")
    print(f"Volumetric flow rate at outlet: {Q_out:.6f}")
    print(f"Relative difference: {abs(Q_in - Q_out) / abs(Q_in):.2%}")

def predict_and_plot_on_collocation(model, device, x_y_collocation, x_y_inlet, x_y_outlet, x_y_top, x_y_bottom, x_y_cylinder,
                                    x_vals_total, h_total, min_vel, max_vel, min_pres, max_pres):
    """
    Plots velocity magnitude and pressure at the collocation points only, with boundary points overlaid.

    Args:
        model (torch.nn.Module): Trained model.
        device (torch.device): Device to run the model on.
        x_y_collocation (torch.Tensor): Collocation points [N, 2].
        x_y_inlet, x_y_outlet, x_y_top, x_y_bottom, x_y_cylinder (torch.Tensor): Boundary points.
        x_vals_total (np.ndarray): x values for nozzle/pipe walls.
        h_total (np.ndarray): height values for nozzle/pipe walls.
        min_vel, max_vel, min_pres, max_pres: Color scale limits.
    """
    model.eval()
    with torch.no_grad():
        x_y = x_y_collocation.to(device)
        out = model(x_y)
        u = out[:, 0].cpu().numpy()
        v = out[:, 1].cpu().numpy()
        p = out[:, 2].cpu().numpy()
        velocity_magnitude = np.sqrt(u**2 + v**2)
        x = x_y_collocation[:, 0].cpu().numpy()
        y = x_y_collocation[:, 1].cpu().numpy()

    # Velocity magnitude scatter
    fig1, ax1 = plt.subplots(figsize=(6, 2))
    sc1 = ax1.scatter(x, y, c=velocity_magnitude, cmap='rainbow', s=5, vmin=min_vel, vmax=max_vel, label='Collocation')
    ax1.scatter(x_y_inlet[:, 0].cpu(), x_y_inlet[:, 1].cpu(), s=8, c='k', label='Inlet')
    ax1.scatter(x_y_outlet[:, 0].cpu(), x_y_outlet[:, 1].cpu(), s=8, c='k', label='Outlet')
    ax1.scatter(x_y_top[:, 0].cpu(), x_y_top[:, 1].cpu(), s=8, c='k', label='Top Wall')
    ax1.scatter(x_y_bottom[:, 0].cpu(), x_y_bottom[:, 1].cpu(), s=8, c='k', label='Bottom Wall')
    ax1.scatter(x_y_cylinder[:, 0].cpu(), x_y_cylinder[:, 1].cpu(), s=8, c='k', label='Cylinder')
    ax1.set_title("Velocity Magnitude at Collocation Points")
    ax1.set_xlabel("x")
    ax1.set_ylabel("y")
    ax1.axis("equal")
    ax1.grid(True)
    #ax1.legend()
    plt.colorbar(sc1, ax=ax1, label="Velocity Magnitude")

    # Pressure scatter
    fig2, ax2 = plt.subplots(figsize=(6, 2))
    sc2 = ax2.scatter(x, y, c=p, cmap='bwr', s=5, vmin=min_pres, vmax=max_pres, label='Collocation')
    ax2.scatter(x_y_inlet[:, 0].cpu(), x_y_inlet[:, 1].cpu(), s=8, c='k', label='Inlet')
    ax2.scatter(x_y_outlet[:, 0].cpu(), x_y_outlet[:, 1].cpu(), s=8, c='k', label='Outlet')
    ax2.scatter(x_y_top[:, 0].cpu(), x_y_top[:, 1].cpu(), s=8, c='k', label='Top Wall')
    ax2.scatter(x_y_bottom[:, 0].cpu(), x_y_bottom[:, 1].cpu(), s=8, c='k', label='Bottom Wall')
    ax2.scatter(x_y_cylinder[:, 0].cpu(), x_y_cylinder[:, 1].cpu(), s=8, c='k', label='Cylinder')
    ax2.set_title("Pressure at Collocation Points")
    ax2.set_xlabel("x")
    ax2.set_ylabel("y")
    ax2.axis("equal")
    ax2.grid(True)
    #ax2.legend()
    plt.colorbar(sc2, ax=ax2, label="Pressure")

    return fig1, fig2