
import torch
import torch.nn as nn

from models.gin_encoder import GINet
from models.feature_fusion import FeatureFusion


class UnifiedMoleculeEncoder(nn.Module):
    def __init__(
        self,
        graph_dim=512,
        fusion_dim=128,
        reduced_dim=128,
        gin_num_layer=5,
        gin_emb_dim=300,
        gin_drop_ratio=0.1,
        gin_pool="mean",
        gin_edge_drop=0.05,
        gin_feat_mask=0.05,
        gin_residual=True,
        gin_jk="last"
    ):
        super().__init__()
        self.moddrop_p = 0.2

        self.graph_encoder = GINet(
            num_layer=gin_num_layer,
            emb_dim=gin_emb_dim,
            feat_dim=graph_dim,
            drop_ratio=gin_drop_ratio,
            pool=gin_pool,
            edge_drop=gin_edge_drop,
            feat_mask=gin_feat_mask,
            residual=gin_residual,
            jk=gin_jk,
        )

        self.fusion_layer = FeatureFusion(
            graph_dim=graph_dim,
            fusion_dim=fusion_dim,
            dropout=0.1,
            use_shared_gate=True,
            gate_temp=1.0,
            use_hadamard=False,
            hadamard_scale=True,
            use_bilinear=False
        )

        self.post_proj = nn.Sequential(
            nn.Linear(fusion_dim, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, reduced_dim)
        )

        self.output_dim = reduced_dim

    def forward(self, data):

        graph_feat = self.graph_encoder(data)
        fused_feat, alpha_g, alpha_d = self.fusion_layer(
            graph_feat, modality_dropout_p=self.moddrop_p
        )
        fused_feat = self.post_proj(fused_feat)
        return fused_feat
