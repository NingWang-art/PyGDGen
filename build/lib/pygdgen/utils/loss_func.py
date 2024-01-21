import torch


def loss_function(x_new, x_old, r, ignore_xold_loss=False):
    x = torch.cat((x_old, x_new), dim=0)
    loss = 0
    for i in range(x.shape[0]):
        for j in range(x.shape[0]):
            if i != j:
                d_ij = torch.norm(x[i] - x[j])
                loss += torch.nn.functional.relu(r[i] + r[j] - d_ij)
    if ignore_xold_loss:
        for i in range(x_old.shape[0]):
            for j in range(x_old.shape[0]):
                if i != j:
                    d_ij = torch.norm(x_old[i] - x_old[j])
                    loss -= torch.nn.functional.relu(r[i] + r[j] - d_ij)
    return loss

def loss_function_PBC(x_new, x_old, r, lx, ly, lz, ignore_xold_loss=False):
    x = torch.cat((x_old, x_new), dim=0)
    loss = 0
    for i in range(x.shape[0]):
        for j in range(x.shape[0]):
            if i != j:
                d = torch.abs(x[i] - x[j])
                dp = torch.tensor([lx, ly, lz]) - d
                d_v, _ = torch.min(torch.stack([d, dp]), dim=0)
                d_ij = torch.norm(d_v)
                loss += torch.nn.functional.relu(r[i] + r[j] - d_ij)
            elif i == j:
                d_ij = min(lx, ly, lz)
                loss += torch.nn.functional.relu(r[i] + r[j] - d_ij)
    if ignore_xold_loss:
        for i in range(x_old.shape[0]):
            for j in range(x_old.shape[0]):
                if i != j:
                    d = x_old[i] - x_old[j]
                    dp = [lx, ly, lz] - d
                    d_v, _ = torch.min(torch.stack([d, dp]), dim=0)
                    d_ij = torch.norm(d_v)
                    loss -= torch.nn.functional.relu(r[i] + r[j] - d_ij)
                elif i == j:
                    d_ij = min(lx, ly, lz)
                    loss -= torch.nn.functional.relu(r[i] + r[j] - d_ij)
    return loss

def loss_function_PBC_gpu(x_new, x_old, r, lx, ly, lz, ignore_xold_loss=False):
    x = torch.cat((x_old, x_new), dim=0)
    lx, ly, lz = torch.tensor([lx, ly, lz], device=x_new.device)  # Ensure they are on the same device
    loss = 0
    for i in range(x.shape[0]):
        for j in range(x.shape[0]):
            if i != j:
                d = torch.abs(x[i] - x[j])
                dp = torch.tensor([lx, ly, lz], device=x_new.device) - d  # Use x_new.device directly
                d_v, _ = torch.min(torch.stack([d, dp]), dim=0)
                d_ij = torch.norm(d_v)
                loss += torch.nn.functional.relu(r[i] + r[j] - d_ij)
                
            elif i == j:
                d_ij = min(lx, ly, lz)
                loss += torch.nn.functional.relu(r[i] + r[j] - d_ij)
    
    if ignore_xold_loss:
        for i in range(x_old.shape[0]):
            for j in range(x_old.shape[0]):
                if i != j:
                    d = x_old[i] - x_old[j]
                    dp = torch.tensor([lx, ly, lz], device=x_new.device) - d  # Use x_new.device directly
                    d_v, _ = torch.min(torch.stack([d, dp]), dim=0)
                    d_ij = torch.norm(d_v)
                    loss -= torch.nn.functional.relu(r[i] + r[j] - d_ij)
                elif i == j:
                    d_ij = min(lx, ly, lz)
                    loss -= torch.nn.functional.relu(r[i] + r[j] - d_ij)

    return loss


def loss_function_atomwise(coord_i, coord_j, rcut, use_pbc=False, xlo=0, xhi=0, ylo=0, yhi=0, zlo=0, zhi=0):
    """
    Calculates the loss based on pairwise distances between atoms in two clusters,
    considering optional periodic boundary conditions.

    Args:
    coord_i (torch.Tensor): Coordinates of atoms in cluster i, shape (i, 3).
    coord_j (torch.Tensor): Coordinates of atoms in cluster j, shape (j, 3).
    rcut (float): Cut-off distance for filtering.
    use_pbc (bool): Whether to use periodic boundary conditions.
    xlo, xhi, ylo, yhi, zlo, zhi (float): Boundaries of the periodic box, if use_pbc is True.

    Returns:
    torch.Tensor: The loss computed as the sum of the filtered distances.
    """
    if use_pbc:
        # Box dimensions
        box = torch.tensor([xhi - xlo, yhi - ylo, zhi - zlo], device=coord_i.device)

        # Calculate pairwise differences with periodic boundary conditions
        deltas = coord_i[:, None, :] - coord_j[None, :, :]
        deltas_adjusted = deltas - box * torch.round(deltas / box)

        # Calculate the distances with periodic boundary conditions
        dist_matrix = torch.norm(deltas_adjusted, dim=2)
    else:
        # Calculate pairwise distance matrix in a standard manner
        dist_matrix = torch.norm(coord_i[:, None] - coord_j, dim=2)

    # Apply ReLU to rcut - distance_matrix
    filtered_distances = torch.relu(rcut - dist_matrix)

    # Sum up all the distances to get the loss
    loss_tmp = torch.sum(filtered_distances)

    loss = torch.relu(loss_tmp - 0.5 * rcut)

    return loss



def loss_outbox(coords, xlo, xhi, ylo, yhi, zlo, zhi, device):
    """
    Simplified calculation of the total loss for atoms outside a specified box.

    Args:
    new_coords_list (list of torch.Tensor): List of tensors, each of shape (num_atom_i, 3).
    xlo, xhi, ylo, yhi, zlo, zhi (float): Boundaries of the box.

    Returns:
    torch.Tensor: The total loss for atoms outside the box.
    """
    # Define the bounds as tensors
    lower_bounds = torch.tensor([xlo, ylo, zlo], device=device)
    upper_bounds = torch.tensor([xhi, yhi, zhi], device=device)

    # Calculate loss for each dimension
    loss = torch.sum(torch.relu(lower_bounds - coords) + torch.relu(coords - upper_bounds))
    return loss
