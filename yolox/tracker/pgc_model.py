import torch
import torch.nn as nn
import torch.nn.functional as F


class PGCTrackNet(nn.Module):
    def __init__(
        self,
        desc_dim=10,
        target_dim=8,
        hidden_dim=128,
        num_heads=4,
        num_layers=2,
        dropout=0.1,
        max_len=16,
    ):
        super().__init__()
        self.desc_dim = desc_dim
        self.target_dim = target_dim
        self.hidden_dim = hidden_dim
        self.max_len = max_len

        self.desc_proj = nn.Linear(desc_dim, hidden_dim)
        self.pos_embed = nn.Parameter(torch.zeros(1, max_len, hidden_dim))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.target_proj = nn.Sequential(
            nn.Linear(target_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.query = nn.Linear(hidden_dim, hidden_dim)
        self.key = nn.Linear(hidden_dim, hidden_dim)
        self.value = nn.Linear(hidden_dim, hidden_dim)
        self.gate = nn.Sequential(
            nn.Linear(hidden_dim + 1, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )
        self.pair_reliability = nn.Linear(hidden_dim, 1)
        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
        )
        self.pred_head = nn.Linear(hidden_dim, 6)

    def encode_pairs(self, pair_seq, pair_token_mask):
        batch_size, num_pairs, seq_len, _ = pair_seq.shape
        flat_seq = pair_seq.reshape(batch_size * num_pairs, seq_len, self.desc_dim)
        flat_mask = pair_token_mask.reshape(batch_size * num_pairs, seq_len).bool()
        safe_mask = flat_mask.clone()
        empty_pairs = ~safe_mask.any(dim=1)
        if empty_pairs.any():
            safe_mask[empty_pairs, 0] = True

        x = self.desc_proj(flat_seq)
        x = x + self.pos_embed[:, :seq_len]
        encoded = self.encoder(x, src_key_padding_mask=~safe_mask)
        weights = flat_mask.float()
        pooled = (encoded * weights.unsqueeze(-1)).sum(dim=1) / weights.sum(dim=1, keepdim=True).clamp_min(1.0)
        pooled = pooled.reshape(batch_size, num_pairs, self.hidden_dim)
        return pooled

    def forward(self, target_feat, pair_seq, pair_token_mask, pair_affinity, pair_mask):
        pair_memory = self.encode_pairs(pair_seq, pair_token_mask)
        target_memory = self.target_proj(target_feat)

        q = self.query(target_memory).unsqueeze(1)
        k = self.key(pair_memory)
        logits = (q * k).sum(dim=-1) / (self.hidden_dim ** 0.5)
        logits = logits.masked_fill(~pair_mask.bool(), -1e4)
        attn = torch.softmax(logits, dim=1)
        attn = torch.where(pair_mask.bool(), attn, torch.zeros_like(attn))

        gate_input = torch.cat([pair_memory, pair_affinity.unsqueeze(-1)], dim=-1)
        gates = torch.sigmoid(self.gate(gate_input)).squeeze(-1) * pair_mask.float()
        values = self.value(pair_memory)
        group = (attn * gates).unsqueeze(-1) * values
        group = group.sum(dim=1)
        group_reliability = (attn * gates).sum(dim=1).clamp(0.0, 1.0)

        fused = self.fusion(torch.cat([target_memory, group], dim=-1))
        pred = self.pred_head(fused)
        pair_logits = self.pair_reliability(pair_memory).squeeze(-1)

        return {
            "delta": pred[:, :4],
            "existence_logit": pred[:, 4],
            "occlusion_logit": pred[:, 5],
            "pair_logits": pair_logits,
            "pair_memory": pair_memory,
            "group_reliability": group_reliability,
            "attention": attn,
            "gates": gates,
        }


def pgc_loss(outputs, labels, weights=None):
    if weights is None:
        weights = {}
    motion_weight = weights.get("motion", 1.0)
    occ_weight = weights.get("occ", 1.0)
    pair_weight = weights.get("pair", 0.5)
    existence_weight = weights.get("existence", 0.2)

    motion = F.smooth_l1_loss(outputs["delta"], labels["delta"])
    occlusion = F.binary_cross_entropy_with_logits(outputs["occlusion_logit"], labels["occlusion"])
    existence = F.binary_cross_entropy_with_logits(outputs["existence_logit"], labels["existence"])

    pair_mask = labels["pair_mask"].bool()
    if pair_mask.any():
        pair = F.binary_cross_entropy_with_logits(
            outputs["pair_logits"][pair_mask],
            labels["pair_label"][pair_mask],
        )
    else:
        pair = outputs["pair_logits"].sum() * 0.0

    total = (
        motion_weight * motion
        + occ_weight * occlusion
        + pair_weight * pair
        + existence_weight * existence
    )
    return {
        "total": total,
        "motion": motion.detach(),
        "occlusion": occlusion.detach(),
        "existence": existence.detach(),
        "pair": pair.detach(),
    }
