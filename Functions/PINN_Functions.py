######################################### Functions needed for the PINN ######################################################

#Import stuff
import torch
import torch.nn as nn
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def domain(N_walls, N_inlet, N_outlet, N_cylinder, N_collocation, L_total, pipe_radius, cylinder_radius, cylinder_center, margin):
    """
    Generates boundary and collocation points for a 2D pipe domain with a circular cylinder.

    Args:
        N_walls (int): Number of points along each wall.
        N_inlet (int): Number of points at the inlet.
        N_outlet (int): Number of points at the outlet.
        N_cylinder (int): Number of points along the cylinder circumference.
        N_collocation (int): Approximate number of collocation points inside the domain.
        L_total (float): Total pipe length (original scale).
        pipe_radius (float): Pipe radius (half-height).
        cylinder_radius (float): Cylinder radius.
        cylinder_center (tuple[float, float]): (x, y) coordinates of the cylinder center.
        margin (float): Distance from boundaries to keep collocation points away.

    Returns:
        tuple:
            x_y_top (torch.Tensor): Top wall coordinates.
            x_y_bottom (torch.Tensor): Bottom wall coordinates.
            x_y_inlet (torch.Tensor): Inlet coordinates.
            x_y_outlet (torch.Tensor): Outlet coordinates.
            x_y_cylinder (torch.Tensor): Cylinder surface coordinates.
            x_y_collocation (torch.Tensor): Interior collocation points (excluding cylinder).
            h_total (np.ndarray): Pipe half-height values along x.
            x_vals_total (np.ndarray): Pipe x-coordinate values.
    """
    ##############################  Boundary Points  ################################################
    x_vals_total = np.linspace(0, L_total, N_walls)  # x coordinates across the domain
    h_total = np.zeros_like(x_vals_total)  # height array

    # pipe section:
    h_total[x_vals_total <= L_total] = pipe_radius  # Constant height

    # Top and bottom walls:
    x_y_top = np.stack([x_vals_total, h_total], axis=1)
    x_y_bottom = np.stack([x_vals_total, -h_total], axis=1)

    # Inlet:
    y_inlet = np.linspace(-h_total[0], h_total[0], N_inlet)
    x_y_inlet = np.stack([np.zeros_like(y_inlet), y_inlet], axis=1)

    # Outlet:
    y_outlet = np.linspace(-h_total[-1], h_total[-1], N_outlet)
    x_y_outlet = np.stack([L_total * np.ones_like(y_outlet), y_outlet], axis=1)

    # Cylinder boundary points
    theta = np.linspace(0, 2 * np.pi, N_cylinder, endpoint=False)
    x_cylinder = cylinder_center[0] + cylinder_radius * np.cos(theta)
    y_cylinder = cylinder_center[1] + cylinder_radius * np.sin(theta)
    x_y_cylinder = np.stack([x_cylinder, y_cylinder], axis=1)

    # Convert to torch tensors 
    x_y_top = torch.tensor(x_y_top, dtype=torch.float32)
    x_y_bottom = torch.tensor(x_y_bottom, dtype=torch.float32)
    x_y_inlet = torch.tensor(x_y_inlet, dtype=torch.float32)
    x_y_outlet = torch.tensor(x_y_outlet, dtype=torch.float32)
    x_y_cylinder = torch.tensor(x_y_cylinder, dtype=torch.float32)

    ################################## Collocation points #########################################
    # Generates evenly spaced collocation points inside the domain
    num_x_points = max(2, (int(np.sqrt(N_collocation * L_total / (2 * pipe_radius)))))
    num_y_points = max(2, (int(np.sqrt(N_collocation * (2 * pipe_radius) / L_total))))

    # Apply margin to avoid points too close to the walls
    x_coords_grid = np.linspace(0 + margin, L_total - margin, num_x_points)
    y_coords_grid = np.linspace(-pipe_radius + margin, pipe_radius - margin, num_y_points)

    # Create a meshgrid from the x and y arrays
    X, Y = np.meshgrid(x_coords_grid, y_coords_grid)
    x_y_collocation = np.hstack([X.flatten()[:, None], Y.flatten()[:, None]])

    # Convert to torch tensor
    x_y_collocation = torch.tensor(x_y_collocation, dtype=torch.float32, requires_grad=True)

    # Filter out points that are inside the cylinder
    coll_np = x_y_collocation.detach().cpu().numpy()
    dist_to_center = np.sqrt((coll_np[:, 0] - cylinder_center[0])**2 + (coll_np[:, 1] - cylinder_center[1])**2)
    mask = dist_to_center > cylinder_radius  
    x_y_collocation = torch.tensor(coll_np[mask], dtype=torch.float32, requires_grad=True)

    return x_y_top, x_y_bottom, x_y_inlet, x_y_outlet, x_y_cylinder, x_y_collocation, h_total, x_vals_total

def physics_loss(model, x_y, Re):
    """
    Computes the physics-based loss terms for the 2D incompressible Navier–Stokes equations.

    Args:
        model (torch.nn.Module): Neural network model.
        x_y (torch.Tensor): Input collocation point coordinates of shape (N, 2). The input should be normalized to [-1, 1].
        Re (float): Reynolds number.

    Returns:
        loss_momentum_x (torch.Tensor): Mean squared residual of the x-momentum equation.
        loss_momentum_y (torch.Tensor): Mean squared residual of the y-momentum equation.
        loss_continuity (torch.Tensor): Mean squared residual of the continuity equation.
    """
    x_y.requires_grad_(True)
    out = model(x_y)    #predict the velocities
    u = out[:, 0:1]     #get the velocity in x
    v = out[:, 1:2]     #get the velocity in y
    p = out[:, 2:3]     #get the pressure

    # Compute gradients
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

    # Residuals of Navier-Stokes
    momentum_x = - 1 / Re * (d2u_dx2 + d2u_dy2) + u * du_dx + v * du_dy + dp_dx
    momentum_y = - 1 / Re * (d2v_dx2 + d2v_dy2) + u * dv_dx + v * dv_dy + dp_dy
    continuity = du_dx + dv_dy

    # Return the mean of the residuals
    loss_momentum_x = torch.mean(momentum_x**2)
    loss_momentum_y = torch.mean(momentum_y**2)
    loss_continuity = torch.mean(continuity**2)
    
    return loss_momentum_x, loss_momentum_y, loss_continuity

def boundary_loss(model, x_y_inlet, x_y_outlet, x_y_top, x_y_bottom, x_y_cylinder, U_max, pipe_radius, scale_factor):
    """
    Computes boundary condition losses for a PINN simulation of flow in a pipe/nozzle with a cylinder.

    Args:
        model (torch.nn.Module): Neural network model.
        x_y_inlet (torch.Tensor): Coordinates of inlet boundary points, shape (N_in, 2). The input should be normalized to [-1, 1].
        x_y_outlet (torch.Tensor): Coordinates of outlet boundary points, shape (N_out, 2). The input should be normalized to [-1, 1].
        x_y_top (torch.Tensor): Coordinates of top wall boundary points, shape (N_top, 2). The input should be normalized to [-1, 1].
        x_y_bottom (torch.Tensor): Coordinates of bottom wall boundary points, shape (N_bottom, 2). The input should be normalized to [-1, 1].
        x_y_cylinder (torch.Tensor): Coordinates of cylinder wall boundary points, shape (N_cyl, 2). The input should be normalized to [-1, 1].
        U_max (float): Maximum inlet velocity (in original physical scale).
        pipe_radius (float): Radius of the pipe in original scale.
        scale_factor (float): Factor to convert original coordinates to normalized [-1, 1] domain.

    Returns:
        inlet_loss (torch.Tensor): Mean squared error for inlet velocity profile.
        wall_loss (torch.Tensor): Mean squared error for no-slip condition on walls and cylinder.
        outlet_loss (torch.Tensor): Mean squared error for zero outlet pressure.
    """
    pred_inlet = model(x_y_inlet)
    pred_outlet = model(x_y_outlet)
    pred_top = model(x_y_top)
    pred_bottom = model(x_y_bottom)
    pred_cylinder = model(x_y_cylinder) 

    ##################################### Inlet ####
    # Inlet: match u, v (no p). Inlet is a parabola
    u_inlet = (U_max * scale_factor) * (1 - (x_y_inlet[:, 1] / (pipe_radius * scale_factor)) ** 2).unsqueeze(1)
    v_inlet = torch.zeros_like(u_inlet)   # velocity in y is zero for the input
    target_inlet = torch.cat([u_inlet, v_inlet], dim=1).to(device)
    inlet_loss = torch.mean((pred_inlet[:, 0:1] - target_inlet[:, 0:1])**2) + \
                 torch.mean((pred_inlet[:, 1:2] - target_inlet[:, 1:2])**2)

    ##################################### Outlet ####
    # Pressure outlet: pressure goes to zero
    outlet_loss = torch.mean(pred_outlet[:, 2:3]**2) 

    ##################################### Walls ####
    # Walls: no-slip
    def compute_normals(coords):
        dx = torch.gradient(coords[:, 0])[0]  # change in x
        dy = torch.gradient(coords[:, 1])[0] # change in y
        tangents = torch.stack([dx, dy], dim=1) # get tangent
        normals = torch.stack([-tangents[:, 1], tangents[:, 0]], dim=1)  # Rotate tangents 90° to get the normals
        normals = torch.nn.functional.normalize(normals, dim=1)  #normalize to unit length
        return normals
    
    normals_top = compute_normals(x_y_top).to(device)
    normals_bottom = compute_normals(x_y_bottom).to(device)
    normals_cylinder = compute_normals(x_y_cylinder).to(device)

    def decompose_wall_velocity(pred, normals):
        tangents = torch.stack([normals[:, 1], -normals[:, 0]], dim=1)  # Rotate normal -90°
        u = pred[:, 0:1]
        v = pred[:, 1:2]
        u_n = u * normals[:, 0:1] + v * normals[:, 1:2]        # normal velocity
        u_t = u * tangents[:, 0:1] + v * tangents[:, 1:2]      # tangential velocity
        return u_n, u_t
    
    u_n_top, u_t_top = decompose_wall_velocity(pred_top, normals_top)
    u_n_bot, u_t_bot = decompose_wall_velocity(pred_bottom, normals_bottom)
    u_n_cyl, u_t_cyl = decompose_wall_velocity(pred_cylinder, normals_cylinder)

    wall_loss = torch.mean(u_n_top**2) + torch.mean(u_t_top**2) \
                + torch.mean(u_n_bot**2) + torch.mean(u_t_bot**2) \
                + torch.mean(u_n_cyl**2) + torch.mean(u_t_cyl**2)   
    
    return inlet_loss, wall_loss, outlet_loss

def xavier_initialization(layer):
    """
    Applies Xavier (Glorot) normal initialization to the weights of a given layer.

    If the layer is a fully-connected (nn.Linear) layer:
      - Weight matrix is initialized using Xavier normal distribution.
      - Bias vector (if present) is initialized to zeros.

    Args:
        layer (torch.nn.Module): Layer to initialize. Typically passed inside
                                 `model.apply(xavier_initialization)`.
    """
    if isinstance(layer, nn.Linear):
        nn.init.xavier_normal_(layer.weight)
        if layer.bias is not None:
            nn.init.zeros_(layer.bias)

def build_mlp(layer_sizes, activations, add_final_activation=True):
    """
    Builds a feed-forward multilayer perceptron (MLP) from layer sizes and activation functions.

    Args:
        layer_sizes (list[int]): Sizes of each layer, including input and output.
                                 For example, [2, 64, 64, 1] builds:
                                 Input(2) → Hidden(64) → Hidden(64) → Output(1).
        activations (list[nn.Module]): Activation functions for the hidden and/or final layers.
                                       For example, [nn.Tanh(), nn.Tanh(), nn.Identity()].
        add_final_activation (bool, optional): If True, applies the last activation in `activations`
                                               to the final output layer. If False, leaves final
                                               output unactivated. Default is True.

    Returns:
        nn.Sequential: PyTorch model implementing the specified MLP architecture.
    """
    layers = []
    for i in range(len(layer_sizes) - 1):
        layers.append(nn.Linear(layer_sizes[i], layer_sizes[i + 1]))
        if i < len(activations):
            layers.append(activations[i])
        elif add_final_activation:
            layers.append(activations[-1])
    return nn.Sequential(*layers)

def normalize_domain_outputs(domain_outputs, L_total, pipe_radius, cylinder_center, cylinder_radius):
    """
    Normalizes the outputs of the domain function such that the larger dimension
    of the original domain is scaled to [-1, 1], preserving the aspect ratio.

    Args:
        domain_outputs (tuple): The tuple of outputs from the original domain function.
                                (x_y_top, x_y_bottom, x_y_inlet, x_y_outlet,
                                 x_y_cylinder, x_y_collocation, h_total, x_vals_total)
        L_total (float): The original total length of the domain.
        pipe_radius (float): The original half-height of the pipe.
        cylinder_center (tuple): The original (x, y) coordinates of the cylinder center.
        cylinder_radius (float): The original radius of the cylinder.

    Returns:
        tuple: A tuple containing the normalized torch tensors for all boundary and
               collocation points, and the normalized cylinder center and radius.
    """
    x_y_top, x_y_bottom, x_y_inlet, x_y_outlet, x_y_cylinder, x_y_collocation, _, _ = domain_outputs

    # Calculate the total span of the domain in x and y and get the scaling factor
    span_x = L_total
    span_y = 2 * pipe_radius 
    max_span = max(span_x, span_y)
    scale_factor = 2.0 / max_span

    # normalize and center coordinate tensors
    def _normalize_and_center_coords(coords_tensor):
        coords_np = coords_tensor.detach().cpu().numpy().copy()
        # Normalize x-coordinates: (original_x - center_of_original_x_range) * scale_factor
        coords_np[:, 0] = (coords_np[:, 0] - (L_total / 2.0)) * scale_factor
        # Normalize y-coordinates: original_y * scale_factor (y is already centered around 0)
        coords_np[:, 1] = coords_np[:, 1] * scale_factor
        return torch.tensor(coords_np, dtype=torch.float32, requires_grad=coords_tensor.requires_grad)

    # Apply normalization to boundary and collocation points
    x_y_top_norm = _normalize_and_center_coords(x_y_top)
    x_y_bottom_norm = _normalize_and_center_coords(x_y_bottom)
    x_y_inlet_norm = _normalize_and_center_coords(x_y_inlet)
    x_y_outlet_norm = _normalize_and_center_coords(x_y_outlet)
    x_y_collocation_norm = _normalize_and_center_coords(x_y_collocation)

    # Normalize cylinder parameters
    cylinder_center_norm = (
        (cylinder_center[0] - (L_total / 2.0)) * scale_factor, # Normalize x-coord of center
        cylinder_center[1] * scale_factor                      # Normalize y-coord of center
    )
    cylinder_radius_norm = cylinder_radius * scale_factor      # Normalize radius

    # For cylinder boundary points, re-calculate based on the normalized center and radius, or apply the transformation directly. 
    x_y_cylinder_np = x_y_cylinder.detach().cpu().numpy().copy()
    # Shift relative to original center, scale, then shift to new normalized center
    x_y_cylinder_np[:, 0] = ((x_y_cylinder_np[:, 0] - cylinder_center[0]) * scale_factor) + cylinder_center_norm[0]
    x_y_cylinder_np[:, 1] = ((x_y_cylinder_np[:, 1] - cylinder_center[1]) * scale_factor) + cylinder_center_norm[1]
    x_y_cylinder_norm = torch.tensor(x_y_cylinder_np, dtype=torch.float32)

    return (x_y_top_norm, x_y_bottom_norm, x_y_inlet_norm, x_y_outlet_norm,
            x_y_cylinder_norm, x_y_collocation_norm, cylinder_center_norm, cylinder_radius_norm, scale_factor)

class PINN(nn.Module):
    """
    Physics-Informed Neural Network (single-branch architecture). This class implements a fully connected MLP that predicts all output variables 
    from a single shared network (e.g., velocities and pressure together).

    Args:
        shared_layers (list[int]): Sizes of each layer in the shared MLP, 
                                   including input and output layer sizes.
                                   Example: [2, 64, 64, 3] → input(2) → hidden(64) → hidden(64) → output(3).
        activation (nn.Module): Activation function to use for all hidden layers
                                (e.g., nn.Tanh(), nn.ReLU()).
        use_xavier (bool): If True, applies Xavier normal initialization to all linear layers.

    Attributes:
        shared (nn.Sequential): The fully connected MLP model.
    
    Methods:
        forward(x: torch.Tensor) -> torch.Tensor:
            Forward pass through the shared MLP.
            Args:
                x (torch.Tensor): Input tensor of shape (batch_size, input_dim).
            Returns:
                torch.Tensor: Network predictions of shape (batch_size, output_dim).
    """
    def __init__(self, shared_layers, activation, use_xavier):
        super(PINN, self).__init__()
        activations = [activation] * (len(shared_layers) - 2)
        self.shared = build_mlp(shared_layers, activations, add_final_activation=False)
        if use_xavier:
            self.apply(xavier_initialization)
    def forward(self, x):
        return self.shared(x)
    
class PINN_branched(nn.Module):
    """
    Physics-Informed Neural Network (branched architecture).
    This class implements a branched MLP:
      1. A shared feature extractor network.
      2. A velocity branch for predicting velocity components (u, v).
      3. A pressure branch for predicting scalar pressure p.

    Args:
        shared_layers (list[int]): Sizes of each layer in the shared MLP feature extractor.
                                   Example: [2, 64, 64] → input(2) → hidden(64) → hidden(64).
        velocity_layers (list[int]): Sizes of each layer in the velocity branch (excluding shared input).
                                     Example: [64, 64, 2] → hidden(64) → hidden(64) → output(2).
        pressure_layers (list[int]): Sizes of each layer in the pressure branch (excluding shared input).
                                     Example: [64, 64, 1] → hidden(64) → hidden(64) → output(1).
        activation (nn.Module): Activation function to use for most layers 
                                (e.g., nn.Tanh(), nn.ReLU()).
        use_xavier (bool): If True, applies Xavier normal initialization to all linear layers.

    Attributes:
        shared (nn.Sequential): Shared feature extractor network.
        velocity_branch (nn.Sequential): Predicts velocity components from shared features.
        pressure_branch (nn.Sequential): Predicts pressure from shared features.

    Methods:
        forward(x: torch.Tensor) -> torch.Tensor:
            Forward pass through the shared network, then both branches.
            Args:
                x (torch.Tensor): Input tensor of shape (batch_size, input_dim).
            Returns:
                torch.Tensor: Concatenated predictions of shape (batch_size, velocity_dim + 1).
                              Velocity components first, followed by pressure.
    """
    def __init__(self, shared_layers, velocity_layers, pressure_layers, activation, use_xavier):
        super(PINN_branched, self).__init__()
        shared_activations = [activation] * (len(shared_layers) - 1)
        self.shared = build_mlp(shared_layers, shared_activations, add_final_activation=True)
        num_relu_layers = 0
        relus = [nn.ReLU() for _ in range(num_relu_layers)]
        remaining = len(velocity_layers) - 1 - num_relu_layers
        velocity_activations = relus + [activation] * remaining 
        self.velocity_branch = build_mlp([shared_layers[-1]] + velocity_layers, velocity_activations, add_final_activation=False)
        pressure_activations = [activation] * (len(pressure_layers) - 1)
        self.pressure_branch = build_mlp([shared_layers[-1]] + pressure_layers, pressure_activations, add_final_activation=False)
        if use_xavier:
            self.apply(xavier_initialization)
    def forward(self, x):
        features = self.shared(x)
        uv = self.velocity_branch(features)
        p = self.pressure_branch(features)
        return torch.cat([uv, p], dim=1)
