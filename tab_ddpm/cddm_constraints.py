import torch
import torch.nn.functional as F

class ConstraintHandler:
    def __init__(self, num_info, cat_info, device):
        """
        num_info: List of (column_index, min_val, max_val)
        cat_info: List of slices or indices for one-hot encoded categorical groups
        """
        self.num_info = num_info
        self.cat_info = cat_info
        self.device = device

    # --- Hard Constraint Enforcement (Sampling-Time) ---
    def project(self, x):
        """Enforce constraints on a generic batch x."""
        with torch.no_grad():
            x_proj = x.clone()

            # Generic Numerical Projection (Eq. 18 & 20)
            for idx, l, u in self.num_info:
                x_proj[:, idx] = torch.clamp(x_proj[:, idx], l, u)

            # Generic Categorical Projection (Eq. 21 & 91)
            for slc in self.cat_info:
                # Applies Euclidean simplex projection to the slice
                x_proj[:, slc] = self._project_simplex(x_proj[:, slc])

        return x_proj

    def _project_simplex(self, v, z=1.0):
        """Euclidean projection onto the probability simplex."""
        n_features = v.shape[1]
        u, _ = torch.sort(v, descending=True, dim=1)
        cssv = torch.cumsum(u, dim=1) - z
        ind = torch.arange(n_features, device=v.device).float()
        cond = u - cssv / (ind + 1) > 0
        rho = torch.sum(cond, dim=1) - 1
        theta = cssv[torch.arange(v.shape[0]), rho] / (rho.float() + 1)
        return torch.clamp(v - theta.unsqueeze(1), min=0.0)

    # --- Soft Constraint Regularization (Training-Time) ---
    def compute_soft_loss(self, x0_hat):
        """Implements Equations (24-26): Differentiable penalties."""
        loss = 0.0
        # Bound violations: max(0, x - u) + max(0, l - x)
        for idx, l, u in self.num_info:
            val = x0_hat[:, idx]
            loss += (F.relu(val - u) + F.relu(l - val)).mean()

        # Simplex violations: |sum(xj) - 1| + non-negativity
        for slc in self.cat_info:
            group = x0_hat[:, slc]
            loss += torch.abs(group.sum(dim=1) - 1.0).mean()
            loss += F.relu(-group).mean()
        return loss
    def project_categorical(self, x_cat):
        """
        Implements Equation (26): sum(xj) = 1 and xi >= 0.
        Uses a standard simplex projection algorithm.
        """
        mu, _ = torch.sort(x_cat, descending=True, dim=-1)
        cum_sum = torch.cumsum(mu, dim=-1)
        # Find rho: max {j | mu_j - (1/j)(sum(mu_i) - 1) > 0}
        indices = torch.arange(1, x_cat.size(-1) + 1, device=x_cat.device)
        rho = torch.max(indices * (mu - (cum_sum - 1) / indices > 0))
        # Calculate theta
        theta = (cum_sum[..., rho-1] - 1) / rho
        return torch.clamp(x_cat - theta.unsqueeze(-1), min=0)
