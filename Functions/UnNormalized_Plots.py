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

def predict_and_plot(model, device, sections, x_vals_total, h_total, L_total, scale_factor, cylinder_center, cylinder_radius, 
                     fig_sizes=(5, 2), ymin=-0.6, ymax=0.6, y_density=100, grid_onoff=False):
    """
    Predicts velocity and pressure using a model trained on normalized inputs and plots the results.

    Args:
        model (torch.nn.Module): The trained model for prediction.
        device (torch.device): The device to run the model on (CPU or GPU).
        sections (list): List of tuples containing section names and x_start, x_end values.
        x_vals_total (np.ndarray): Array of x values for the nozzle profile.
        h_total (np.ndarray): Array of height values for the nozzle profile.
        L_total (float): The original total length of the domain.
        scale_factor (float): Factor to scale the normalized inputs back to original values.
        cylinder_center (tuple): (x, y) coordinates of the cylinder center.
        cylinder_radius (float): Radius of the cylinder.
        fig_sizes (tuple): Size of the figures for the plots. Default is (5, 2).
        ymin (float): Minimum y value for the plot. Default is -0.6.
        ymax (float): Maximum y value for the plot. Default is 0.6.
        y_density (int): Number of points in the y direction for the meshgrid. Default is 100.
        grid_onoff (bool): Whether to display the grid on the plots. Default is False.

    Returns:
        tuple: Figures for velocity and pressure plots.
    """
    
    x_center = L_total / 2.0
    y_vals = np.linspace(ymin, ymax, y_density)
    fig1, ax1 = plt.subplots(figsize=fig_sizes)
    fig2, ax2 = plt.subplots(figsize=fig_sizes)

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

        # Normalize Inputs 
        x_valid_norm = (x_valid - x_center) * scale_factor
        y_valid_norm = y_valid * scale_factor
        
        input_tensor = torch.tensor(np.stack([x_valid_norm, y_valid_norm], axis=1), dtype=torch.float32).to(device)

        with torch.no_grad():
            pred = model(input_tensor)
            u = pred[:, 0].cpu().numpy()
            v = pred[:, 1].cpu().numpy()
            vel_mag = np.sqrt(u**2 + v**2) / scale_factor
            pressure = pred[:, 2].cpu().numpy() 

        contour_v = ax1.tricontourf(x_valid, y_valid, vel_mag, levels=100, cmap='rainbow')
        all_contours_v.append(contour_v)

        contour_p = ax2.tricontourf(x_valid, y_valid, pressure, levels=100, cmap='bwr')
        all_contours_p.append(contour_p)

    cylinder_patch_v = plt.Circle(cylinder_center, cylinder_radius, color='black', zorder=2)
    ax1.add_patch(cylinder_patch_v)
    ax1.plot(x_vals_total, h_total, 'k-', linewidth=2)
    ax1.plot(x_vals_total, -h_total, 'k-', linewidth=2)

    ax1.set_title("Velocity Magnitude")
    ax1.set_xlabel("x [m]")
    ax1.set_ylabel("y [m]")
    ax1.axis("equal")
    ax1.grid(grid_onoff)

    plt.colorbar(all_contours_v[0], ax=ax1, label="Velocity Magnitude [m/s]")

    cylinder_patch_v = plt.Circle(cylinder_center, cylinder_radius, color='black', zorder=2)
    ax2.add_patch(cylinder_patch_v)
    ax2.plot(x_vals_total, h_total, 'k-', linewidth=2)
    ax2.plot(x_vals_total, -h_total, 'k-', linewidth=2)

    ax2.set_title("Pressure")
    ax2.set_xlabel("x [m]")
    ax2.set_ylabel("y [m]")
    ax2.axis("equal")
    ax2.grid(grid_onoff)

    plt.colorbar(all_contours_p[0], ax=ax2, label="Pressure [Pa]")

    return fig1, fig2

def predict_and_quiver(model, device, sections, scale_factor, L_total, x_vals_total, h_total, cylinder_center, cylinder_radius, fig_sizes=(15, 7), density=100, scale=1, xlim=None, ylim=None, ymin=-0.6, ymax=0.6, y_density=200, isolate_region=None):
    """
    Predicts velocity vectors using a trained PINN model and visualizes them as a quiver plot over a custom domain.

    This function generates a grid of points covering the specified domain, excluding regions inside a cylinder and optionally isolating a subregion.
    It predicts velocity components at each valid point, un-normalizes them, and downsamples the results for even arrow spacing.
    The function overlays the nozzle walls and cylinder boundary on the plot.

    Args:
        model (torch.nn.Module): Trained PINN model for velocity prediction.
        device (torch.device): Device to run the model on ('cpu' or 'cuda').
        sections (list): List of tuples (name, x_start, x_end) defining domain sections.
        scale_factor (float): Factor to un-normalize coordinates for model input.
        L_total (float): Total length of the domain (used for centering).
        x_vals_total (np.ndarray): Array of x values for the nozzle/profile wall.
        h_total (np.ndarray): Array of height values for the nozzle/profile wall.
        cylinder_center (tuple): (x, y) coordinates of the cylinder center.
        cylinder_radius (float): Radius of the cylinder.
        density (int): Number of arrows for quiver plot (controls downsampling).
        scale (float): Scale factor for quiver arrow length.
        xlim (tuple, optional): x-axis limits for the plot.
        ylim (tuple, optional): y-axis limits for the plot.
        ymin (float): Minimum y value for the grid.
        ymax (float): Maximum y value for the grid.
        y_density (int): Number of points in the y direction for the meshgrid.
        isolate_region (tuple, optional): (x_min, x_max, y_min, y_max) to restrict the plotted arrows to a subregion.

    Returns:
        fig: Figure object containing the quiver plot.
    """
    x_center = L_total / 2.0
    
    # Figure out total x range from sections
    x_min_domain = min(s[1] for s in sections)
    x_max_domain = max(s[2] for s in sections)

    # Create full grid once
    x_domain = np.linspace(x_min_domain, x_max_domain, 500)
    y_domain = np.linspace(ymin, ymax, y_density)
    x_grid, y_grid = np.meshgrid(x_domain, y_domain)
    x_flat = x_grid.flatten()
    y_flat = y_grid.flatten()

    # Nozzle mask
    h_interp = np.interp(x_flat, x_vals_total, h_total)
    nozzle_mask = np.abs(y_flat) <= h_interp

    # Cylinder mask (exclude points inside cylinder)
    dist_to_cyl_center = np.sqrt((x_flat - cylinder_center[0])**2 + (y_flat - cylinder_center[1])**2)
    cylinder_mask = dist_to_cyl_center >= cylinder_radius

    # Region isolation mask (keep only inside region)
    if isolate_region is not None:
        x_min, x_max, y_min_r, y_max_r = isolate_region
        region_mask = (x_flat >= x_min) & (x_flat <= x_max) & \
                        (y_flat >= y_min_r) & (y_flat <= y_max_r)
    else:
        region_mask = np.ones_like(x_flat, dtype=bool)

    # Combine masks: inside nozzle & outside cylinder & inside region
    mask = nozzle_mask & cylinder_mask & region_mask
    x_valid = x_flat[mask]
    y_valid = y_flat[mask]

    x_valid_norm = (x_valid - x_center) * scale_factor
    y_valid_norm = y_valid * scale_factor

    input_tensor = torch.tensor(np.stack([x_valid_norm, y_valid_norm], axis=1), dtype=torch.float32).to(device)
    with torch.no_grad():
        pred = model(input_tensor)
        u = pred[:, 0].cpu().numpy()
        v = pred[:, 1].cpu().numpy()

    u = u / scale_factor  # Un-normalize velocity components
    v = v / scale_factor

    # 2D binning downsampling for even arrow spacing
    nx_bins = int(np.sqrt(density))
    ny_bins = int(np.sqrt(density))
    x_bins = np.linspace(x_valid.min(), x_valid.max(), nx_bins + 1)
    y_bins = np.linspace(y_valid.min(), y_valid.max(), ny_bins + 1)

    sample_idx = []
    for i in range(nx_bins):
        for j in range(ny_bins):
            in_bin = (
                (x_valid >= x_bins[i]) & (x_valid < x_bins[i+1]) &
                (y_valid >= y_bins[j]) & (y_valid < y_bins[j+1])
            )
            bin_indices = np.where(in_bin)[0]
            if bin_indices.size > 0:
                sample_idx.append(bin_indices[0])

    x_ds = x_valid[sample_idx]
    y_ds = y_valid[sample_idx]
    u_ds = u[sample_idx]
    v_ds = v[sample_idx]

    fig, ax = plt.subplots(figsize=fig_sizes)

    ax.quiver(
            x_ds, 
            y_ds, 
            u_ds, 
            v_ds,
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

    # Overlay cylinder wall
    cyl = plt.Circle(cylinder_center, cylinder_radius, color='blue', fill=False, linewidth=2)
    ax.add_patch(cyl)

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

def plot_residuals(model, device, Re, scale_factor, x_vals_total, h_total, cylinder_center, cylinder_radius, x_range, y_range, fig_sizes=(7, 3),
                   x_points=200, y_points=100):
    """
    Plots residuals (momentum x, momentum y, continuity) for a given model. The output is normalized.

    Args:
        model: Trained PINN model.
        device: Device to run the model on (e.g., 'cpu' or 'cuda').
        Re: Reynolds number for the flow.
        scale_factor: Factor to normalize coordinates for model input.
        x_vals_total: Array of x values for the nozzle/profile wall. The input should correspond to the original, not normalized domain. 
        h_total: Array of height values for the nozzle/profile wall. The input should correspond to the original, not normalized domain.
        cylinder_center: Tuple (x, y) for the center of the cylinder. The input should correspond to the original, not normalized domain.
        cylinder_radius: Radius of the cylinder. The input should correspond to the original, not normalized domain.
        fig_sizes: Size of the figures for the plots.
        x_range: Tuple (min_x, max_x) for plotting domain in x. The input should correspond to the original, not normalized domain. 
        y_range: Tuple (min_y, max_y) for plotting domain in y. The input should correspond to the original, not normalized domain. 
        x_points: Number of points in x direction. 
        y_points: Number of points in y direction. 

    Returns:
        fig_mx, fig_my, fig_cont: Plots of momentum in x, in y, and continuity residuals.
    """
    x_span = x_range[1] - x_range[0]
    x_center = x_range[0] + x_span / 2.0
    
    # Normalize nozzle walls
    x_vals_total = (x_vals_total - x_center) * scale_factor
    h_total = h_total * scale_factor
    
    # Normalize cylinder
    cylinder_center = ((cylinder_center[0] - x_center) * scale_factor,
                            cylinder_center[1] * scale_factor)
    cylinder_radius = cylinder_radius * scale_factor

    # Normalize x_range and y_range
    x_range = (x_range[0] - x_center) * scale_factor, (x_range[1] - x_center) * scale_factor
    y_range = (y_range[0] * scale_factor, y_range[1] * scale_factor)

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

    momentum_x = - 1 / Re * (d2u_dx2 + d2u_dy2) + u * du_dx + v * du_dy + dp_dx
    momentum_y = - 1 / Re * (d2v_dx2 + d2v_dy2) + u * dv_dx + v * dv_dy + dp_dy
    continuity = du_dx + dv_dy

    def plot_residual(field, title):
        fig, ax = plt.subplots(figsize=fig_sizes)
        tcf = ax.tricontourf(x_y[:, 0].detach().cpu(), x_y[:, 1].detach().cpu(), field.detach().cpu().flatten(), levels=100, cmap='RdBu')
        plt.colorbar(tcf, ax=ax, label='Residual')

        # Overlay nozzle walls
        ax.plot(x_vals_total, h_total, 'black', linewidth=2)
        ax.plot(x_vals_total, -h_total, 'black', linewidth=2)

        # Overlay cylinder wall
        cyl = plt.Circle(cylinder_center, cylinder_radius, color='black', zorder=2)
        ax.add_patch(cyl)
        ax.set_title(title)
        ax.set_xlabel('x [m]')
        ax.set_ylabel('y [m]')
        ax.axis('equal')
        fig.tight_layout()
        return fig

    fig_mx = plot_residual(momentum_x, "Momentum x-Residual")
    fig_my = plot_residual(momentum_y, "Momentum y-Residual")
    fig_cont = plot_residual(continuity, "Continuity Residual")

    return fig_mx, fig_my, fig_cont

def print_boundary_summary(model, x_y_inlet_norm, x_y_outlet_norm, x_y_top_norm, x_y_bottom_norm, scale_factor):
    """
    Prints summary statistics for boundary velocities and outlet pressure for a given model.

    Args:
        model (torch.nn.Module): Trained PINN model.
        x_y_inlet (torch.Tensor): Tensor of inlet coordinates (x, y). The input should correspond to the original, not normalized domain.
        x_y_outlet (torch.Tensor): Tensor of outlet coordinates (x, y). The input should correspond to the original, not normalized domain.
        x_y_top (torch.Tensor): Tensor of top wall coordinates (x, y). The input should correspond to the original, not normalized domain.
        x_y_bottom (torch.Tensor): Tensor of bottom wall coordinates (x, y). The input should correspond to the original, not normalized domain.
        scale_factor (float): Factor to normalize coordinates for model input. 

    Returns:
        None. Prints the maximum velocity at the inlet, top wall, and bottom wall,
        and the maximum pressure at the outlet.
    """
    pred_inlet = model(x_y_inlet_norm)
    pred_outlet = model(x_y_outlet_norm)
    pred_top = model(x_y_top_norm)
    pred_bottom = model(x_y_bottom_norm)

    # Velocity magnitude for inlet, top, bottom
    vel_inlet = torch.sqrt(pred_inlet[:, 0]**2 + pred_inlet[:, 1]**2) / scale_factor
    vel_top = torch.sqrt(pred_top[:, 0]**2 + pred_top[:, 1]**2) / scale_factor
    vel_bottom = torch.sqrt(pred_bottom[:, 0]**2 + pred_bottom[:, 1]**2) / scale_factor

    # Pressure at outlet
    pressure_outlet = pred_outlet[:, 2]

    print('Inlet max velocity:', vel_inlet.max().item())
    print('Outlet max pressure:', pressure_outlet.max().item())
    print('Wall top max velocity:', vel_top.max().item())
    print('Wall bottom max velocity:', vel_bottom.max().item())

def plot_velocity_profiles(
    model, device, cylinder_center, cylinder_radius, 
    x_vals_total, h_total,
    x_vals_total_norm, h_total_norm,
    L_total,
    x_y_inlet_norm, x_y_outlet_norm,
    scale_factor, visual_scale, 
    x_locs, fig_sizes=(8, 5)
):
    """
    Plots velocity profiles at specified cross-sections.

    Args:
        model (torch.nn.Module): The trained model for prediction.
        device (torch.device): The device to run the model on.
        cylinder_center (tuple): (x, y) coordinates of the cylinder center.
        cylinder_radius (float): Radius of the cylinder.
        x_vals_total (np.ndarray): Original-scale x values for the profile.
        h_total (np.ndarray): Original-scale half-height values for the profile.
        x_vals_total_norm (np.ndarray): Normalised x values (range [-1, 1]).
        h_total_norm (np.ndarray): Normalised half-height values.
        L_total (float): Total length of the domain (original scale).
        x_y_inlet_norm (torch.Tensor): Normalised inlet coordinates (x, y).
        x_y_outlet_norm (torch.Tensor): Normalised outlet coordinates (x, y).
        scale_factor (float): Factor to scale back to physical units.
        visual_scale (float): Additional scaling for plotting velocities.
        x_locs (list[float]): Locations in original scale where velocity profiles are plotted.
        fig_sizes (tuple): Size of the figure for the plot.

    Returns:
        matplotlib.figure.Figure: Figure with velocity profiles.
    """
    import numpy as np
    import torch
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=fig_sizes)

    # Plot channel shape
    ax.plot(x_vals_total, h_total, 'k-', linewidth=2)
    ax.plot(x_vals_total, -h_total, 'k-', linewidth=2)

    # --- Inlet ---
    u_inlet_pred = model(x_y_inlet_norm).detach().cpu().numpy()[:, 0] / scale_factor * visual_scale
    y_inlet = x_y_inlet_norm[:, 1].detach().cpu().numpy() / scale_factor
    ax.plot(u_inlet_pred + u_inlet_pred.min(), y_inlet, 'r--', label='Predicted Inlet', linewidth=2)

    # --- Outlet ---
    u_outlet_pred = model(x_y_outlet_norm).detach().cpu().numpy()[:, 0] / scale_factor * visual_scale
    y_outlet = x_y_outlet_norm[:, 1].detach().cpu().numpy() / scale_factor
    ax.plot(u_outlet_pred + L_total, y_outlet, 'g--', label='Predicted Outlet', linewidth=2)

    # --- Profiles at x_locs ---
    x_min, x_max = x_vals_total[0], x_vals_total[-1]
    for x_loc_orig in x_locs:
        # Convert to normalised scale
        x_loc_norm = 2 * (x_loc_orig - x_min) / (x_max - x_min) - 1

        # Interpolate half-height in normalised space
        h_loc_norm = np.interp(x_loc_norm, x_vals_total_norm, h_total_norm)

        # Create y-values in normalised space
        y_vals_norm = torch.linspace(-h_loc_norm, h_loc_norm, 100).view(-1, 1).to(device)
        x_vals_norm = torch.full_like(y_vals_norm, x_loc_norm)
        xy_loc_norm = torch.cat([x_vals_norm, y_vals_norm], dim=1)

        # Predict in normalised space
        with torch.no_grad():
            u_pred_norm = model(xy_loc_norm)[:, 0].cpu().numpy()

        # Convert y back to original scale for plotting
        y_vals_orig = y_vals_norm.cpu().numpy() / scale_factor
        u_pred_orig = u_pred_norm / scale_factor * visual_scale

        # Plot in original coordinates
        ax.plot(x_loc_orig + u_pred_orig, y_vals_orig, 'b-', label=f'Predicted x={x_loc_orig:.2f}', linewidth=2)

    # Overlay cylinder wall (requires cylinder_center and cylinder_radius to be defined globally or passed in)
    try:
        cyl = plt.Circle(cylinder_center, cylinder_radius, color='black', zorder=2)
        ax.add_patch(cyl)
    except NameError:
        pass

    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_title("Velocity Profiles at Different Cross-sections")
    ax.grid(True)
    ax.axis("equal")
    ax.legend()
    return fig

def validation_by_flux(model, x_vals_total, x_vals_total_norm, h_total, h_total_norm, U_max, scale_factor):
    """
    Computes and prints the volumetric flow rate at the inlet and outlet.

    Args:
        model: Trained PINN model.
        x_vals_total: Array of x values for the nozzle profile.
        x_vals_total_norm: Normalised array of x values for the nozzle profile.
        h_total: Array of height values for the nozzle profile in original scale.
        h_total_norm: Normalised array of height values for the nozzle profile.
        U_max: Maximum velocity in the flow.
        scale_factor: Factor to scale back to physical units.

    Returns:
        None: Prints the volumetric flow rates at the inlet and outlet, and their relative difference.
    """
    x_inlet_org = x_vals_total.min().item()
    x_inlet_norm = x_vals_total_norm.min().item()
    x_outlet_norm = x_vals_total_norm.max().item()

    h_inlet_original = np.interp(x_inlet_org, x_vals_total, h_total)
    h_inlet_norm = np.interp(x_inlet_norm, x_vals_total_norm, h_total_norm)
    h_outlet_norm = np.interp(x_outlet_norm, x_vals_total_norm, h_total_norm)

    def compute_flux(model, x_fixed, y_min, y_max, N=100):
        y_vals = torch.linspace(y_min, y_max, N).view(-1, 1)
        x_vals = torch.full_like(y_vals, x_fixed)
        xy = torch.cat([x_vals, y_vals], dim=1).to(next(model.parameters()).device)

        with torch.no_grad():
            u_pred = model(xy)[:, 0:1]  
            dy = (y_max - y_min) / (N - 1)
            flux_norm = torch.sum(u_pred) * dy

        # Convert to physical scale
        flux = flux_norm.item() / (scale_factor**2)
        return flux

    Q_in = compute_flux(model, x_fixed=x_inlet_norm, y_min=-h_inlet_norm, y_max=h_inlet_norm)
    Q_out = compute_flux(model, x_fixed=x_outlet_norm, y_min=-h_outlet_norm, y_max=h_outlet_norm)

    Q_in_analytical = (4 / 3) * U_max * (h_inlet_original)

    print(f"Volumetric flow rate at inlet:  {Q_in:.6f} m^2/s")
    print(f"Volumetric flow rate at outlet: {Q_out:.6f} m^2/s")
    print(f"Relative difference: {abs(Q_in - Q_out) / abs(Q_in):.2%}")
    print(f"Volumetric flow rate analytical for pipe flow: {Q_in_analytical} m^2/s")


def predict_and_plot_on_collocation(model, device, x_y_collocation, x_y_inlet, x_y_outlet, x_y_top, x_y_bottom, x_y_cylinder,
                                    min_vel, max_vel, min_pres, max_pres, fig_sizes=(6, 2)):
    """
    Plots velocity magnitude and pressure at the collocation points only, with boundary points overlaid.

    Args:
        model (torch.nn.Module): Trained model.
        device (torch.device): Device to run the model on.
        x_y_collocation (torch.Tensor): Collocation points [N, 2].
        x_y_inlet, x_y_outlet, x_y_top, x_y_bottom, x_y_cylinder (torch.Tensor): Boundary points.
        min_vel, max_vel, min_pres, max_pres: Color scale limits.
        fig_sizes (tuple): Size of the figures for the plots.
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
    fig1, ax1 = plt.subplots(figsize=fig_sizes)
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
    #ax1.grid(True)
    #ax1.legend()
    plt.colorbar(sc1, ax=ax1, label="Velocity Magnitude")

    # Pressure scatter
    fig2, ax2 = plt.subplots(figsize=fig_sizes)
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
    #ax2.grid(True)
    #ax2.legend()
    plt.colorbar(sc2, ax=ax2, label="Pressure")

    return fig1, fig2