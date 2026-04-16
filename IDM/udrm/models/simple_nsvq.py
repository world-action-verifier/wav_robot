import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleNSVQ(nn.Module):
    def __init__(
        self,
        dim: int,
        codebook_size: int,
        discarding_threshold: float = 0.01,
        initialization: str = "normal",  # less meaningful now, kept for compatibility
        eps: float = 1e-12,
    ):
        super().__init__()
        self.dim = dim
        self.codebook_size = codebook_size
        self.discarding_threshold = discarding_threshold
        self.eps = eps

        # Random at init, but overwritten by data on first forward pass
        self.codebook = nn.Parameter(torch.randn(codebook_size, dim))
        
        # [Key 1] Initialization flag (stored as a buffer for save/load support)
        self.register_buffer("is_initialized", torch.tensor(0, dtype=torch.uint8))

        # Usage tracking
        self.register_buffer(
            "codebook_usage",
            torch.zeros(codebook_size, dtype=torch.long),
        )

    def _init_codebook(self, x_flat):
        """
        Force-initialize the codebook from the first batch (similar to K-means++).
        """
        with torch.no_grad():
            # Randomly sample codebook_size vectors from input data
            n_data = x_flat.size(0)
            if n_data < self.codebook_size:
                # Allow sampling with replacement when data is insufficient
                indices = torch.randint(0, n_data, (self.codebook_size,))
            else:
                # Sample without replacement when data is sufficient
                indices = torch.randperm(n_data)[:self.codebook_size]
            
            # Copy selected vectors into the codebook
            selected_data = x_flat[indices].clone()
            
            # [Key 2] L2 normalization (optional but often helps convergence)
            # Normalizing data and codebook onto a sphere stabilizes distance computation
            # Here we only align scale using selected data
            self.codebook.data.copy_(selected_data)
            
            self.is_initialized.fill_(1)
            print(f"[SimpleNSVQ] Codebook initialized from data! (Shape: {self.codebook.shape})")

    def forward(self, x: torch.Tensor, codebook_training_only: bool = False):
        x_flat = x.reshape(-1, self.dim)
        
        # [Key 1] Initialize from current data during training if not initialized
        if self.training and self.is_initialized.item() == 0:
            self._init_codebook(x_flat)

        # ------------------------------------------------------------------
        # [Key 3] L2 normalization (optional but strongly recommended)
        # Reduces mapping errors caused by vector magnitude differences
        # Normalize both data and codebook to behave similarly to cosine-based matching
        # (Comment out the two lines below if not needed)
        # x_norm = F.normalize(x_flat, dim=1)
        # codebook_norm = F.normalize(self.codebook, dim=1)
        # ------------------------------------------------------------------
        
        # Distance computation (using x_flat and self.codebook)
        x_sq = (x_flat ** 2).sum(dim=1, keepdim=True)
        e_sq = (self.codebook ** 2).sum(dim=1)
        distances = x_sq - 2 * (x_flat @ self.codebook.t()) + e_sq.unsqueeze(0)

        indices = torch.argmin(distances, dim=1)
        codes = self.codebook[indices]

        # --- NSVQ Logic ---
        resid = x_flat - codes
        resid_norm = resid.norm(dim=1, keepdim=True)
        
        noise = torch.randn_like(x_flat)
        noise_norm = noise.norm(dim=1, keepdim=True)
        
        scaled_noise = (resid_norm / (noise_norm + self.eps)) * noise

        if codebook_training_only:
            quantized_flat = codes
        else:
            quantized_flat = x_flat + scaled_noise

        # --- Loss (including latest revision) ---
        commitment_loss = F.mse_loss(x_flat, codes.detach())
        codebook_loss = F.mse_loss(x_flat.detach(), codes)
        
        # Increase codebook loss weight (1.0) to pull codebook toward data more strongly
        vq_loss = codebook_loss + 0.25 * commitment_loss

        # Usage Update
        if self.training:
            with torch.no_grad():
                self.codebook_usage.index_add_(
                    0, indices, torch.ones_like(indices, dtype=torch.long)
                )

        quantized = quantized_flat.view(*x.shape)
        indices = indices.view(*x.shape[:-1])

        return quantized, indices, vq_loss

    @torch.no_grad()
    def replace_unused_codebooks(self, num_batches: int):
        # Keep original logic
        if num_batches <= 0: return
        usage_rate = self.codebook_usage.float() / float(num_batches)
        unused = torch.where(usage_rate < self.discarding_threshold)[0]
        used = torch.where(usage_rate >= self.discarding_threshold)[0]

        if used.numel() == 0:
            self.codebook.add_(self.eps * torch.randn_like(self.codebook))
        elif unused.numel() > 0:
            # Randomly sample from active codes to replace dead codes
            # Add small noise to avoid exact overlap
            used_codes = self.codebook[used]
            idx = torch.randint(0, used_codes.size(0), (unused.size(0),))
            self.codebook[unused] = used_codes[idx] + torch.randn_like(self.codebook[unused]) * 0.02
        
        self.codebook_usage.zero_()
