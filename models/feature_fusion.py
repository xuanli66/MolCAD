
import torch
import torch.nn as nn


class FeatureFusion(nn.Module):
    def __init__(self, graph_dim, desc_dim, fusion_dim=128,
                 dropout=0.1, use_shared_gate=True, gate_temp=1.0,
                 use_hadamard=True, hadamard_scale=True, use_bilinear=False):
        super().__init__()
        self.fusion_dim = fusion_dim
        self.use_shared_gate = use_shared_gate
        self.gate_temp = gate_temp
        self.use_hadamard = use_hadamard
        self.hadamard_scale = hadamard_scale
        self.use_bilinear = use_bilinear

        self.graph_proj = nn.Linear(graph_dim, fusion_dim)
        self.desc_proj  = nn.Linear(desc_dim,  fusion_dim)
        self.g_ln = nn.LayerNorm(fusion_dim)
        self.d_ln = nn.LayerNorm(fusion_dim)
        self.dp   = nn.Dropout(dropout)

        gate_in = fusion_dim * (3 if use_hadamard else 2)

        def make_gate():
            gate = nn.Sequential(
                nn.Linear(gate_in, fusion_dim),
                nn.ReLU(inplace=True),
                nn.Linear(fusion_dim, fusion_dim)
            )

            nn.init.zeros_(gate[-1].weight)
            nn.init.zeros_(gate[-1].bias)
            return gate

        if use_shared_gate:
            self.shared_gate = make_gate()
            self.gate_from_graph = None
            self.gate_from_desc  = None
        else:
            self.shared_gate = None
            self.gate_from_graph = make_gate()  # g→d
            self.gate_from_desc  = make_gate()  # d→g


        if use_bilinear:
            self.bilinear = nn.Bilinear(fusion_dim, fusion_dim, fusion_dim)
        else:
            self.bilinear = None

        if self.use_hadamard and self.hadamard_scale:
            self.hadamard_alpha = nn.Parameter(torch.tensor(0.0))

        self._last_gate_g = None
        self._last_gate_d = None

        nn.init.xavier_uniform_(self.graph_proj.weight); nn.init.zeros_(self.graph_proj.bias)
        nn.init.xavier_uniform_(self.desc_proj.weight);  nn.init.zeros_(self.desc_proj.bias)

    @staticmethod
    def _l2n(x, eps=1e-6):
        return x / (x.norm(dim=1, keepdim=True) + eps)

    def forward(self, graph_feat, desc_feat, modality_dropout_p: float = 0.0):

        g = self.dp(self.g_ln(self.graph_proj(graph_feat)))
        d = self.dp(self.d_ln(self.desc_proj(desc_feat)))

        if self.training and modality_dropout_p > 0.0:
            mask = torch.rand(g.size(0), device=g.device)
            drop_g = (mask < (modality_dropout_p / 2)).float().unsqueeze(1)
            drop_d = ((mask >= (modality_dropout_p / 2)) & (mask < modality_dropout_p)).float().unsqueeze(1)
            g = g * (1.0 - drop_g)
            d = d * (1.0 - drop_d)

        if self.use_hadamard:
            inter = g * d
            gd_for_gate_g = torch.cat([g, d, inter], dim=1)
            gd_for_gate_d = torch.cat([d, g, inter], dim=1)
        else:
            gd_for_gate_g = torch.cat([g, d], dim=1)
            gd_for_gate_d = torch.cat([d, g], dim=1)

        if self.use_shared_gate:
            raw = self.shared_gate(gd_for_gate_g)
            alpha_d = torch.sigmoid(raw / self.gate_temp)
            alpha_g = torch.sigmoid(self.shared_gate(gd_for_gate_d) / self.gate_temp)
        else:
            alpha_d = torch.sigmoid(self.gate_from_graph(gd_for_gate_g) / self.gate_temp)
            alpha_g = torch.sigmoid(self.gate_from_desc(gd_for_gate_d) / self.gate_temp)

        d_attn = alpha_d * d
        g_attn = alpha_g * g

        fused = g_attn + d_attn
        if self.use_hadamard:
            if self.hadamard_scale:
                fused = fused + torch.tanh(self.hadamard_alpha) * (g * d)
            else:
                fused = fused + (g * d)
        if self.bilinear is not None:
            fused = fused + self.bilinear(g, d)

        self._last_gate_g = alpha_g.detach()
        self._last_gate_d = alpha_d.detach()

        return fused, alpha_g, alpha_d

    def gate_entropy(self):
        def ent(a, eps=1e-8):
            p = torch.clamp(a, eps, 1 - eps)
            return -(p * p.log() + (1 - p) * (1 - p).log()).mean()
        if self._last_gate_g is None or self._last_gate_d is None:
            return torch.tensor(0.0)
        return (ent(self._last_gate_g) + ent(self._last_gate_d)) / 2
