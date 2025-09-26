import torch 

class NGD(torch.optim.SGD):
    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            for param in group["params"]:
                assert param.isnan().sum().item() == 0
                g = param.grad
                
                # Normalize the gradient
                g /= g.norm(dim=-1, keepdim=True)
                
                # Replace NaN, positive inf, and negative inf with zeros
                g = torch.where(torch.isnan(g), torch.zeros_like(g), g)  # NaNs -> 0
                g = torch.where(torch.isposinf(g), torch.zeros_like(g), g)  # +Inf -> 0
                g = torch.where(torch.isneginf(g), torch.zeros_like(g), g)  # -Inf -> 0

                # Update parameter
                param -= group["lr"] * g
