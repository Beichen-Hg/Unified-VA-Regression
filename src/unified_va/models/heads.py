"""Linear/dual-MLP regression and training-only condition classifiers."""

import torch
from torch import nn
from torch.nn import functional as F


class LinearHead(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.fc = nn.Linear(hidden_dim, 2)

    def forward(self, value):
        return self.fc(value.to(self.fc.weight.dtype))


class DualHeadMLP(nn.Module):
    def __init__(self, hidden_dim, shared_hidden=512, branch_hidden=256, dropout=0.1):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(hidden_dim, shared_hidden), nn.GELU(), nn.Dropout(dropout)
        )

        def branch():
            return nn.Sequential(
                nn.Linear(shared_hidden, branch_hidden),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(branch_hidden, 1),
            )

        self.v_branch, self.a_branch = branch(), branch()

    def forward(self, value):
        shared = self.shared(value.to(self.shared[0].weight.dtype))
        return torch.cat([self.v_branch(shared), self.a_branch(shared)], dim=-1)


class ConditionClassifier(nn.Module):
    def __init__(self, langs, domains, embed_dim=16, hidden_dim=64, dropout=0.1):
        super().__init__()
        self.lang_to_idx = {"__unk__": 0, **{v: i + 1 for i, v in enumerate(langs)}}
        self.domain_to_idx = {"__unk__": 0, **{v: i + 1 for i, v in enumerate(domains)}}

        def head(size):
            return nn.Sequential(
                nn.LayerNorm(embed_dim),
                nn.Linear(embed_dim, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, size),
            )

        self.lang_head, self.domain_head = head(len(self.lang_to_idx)), head(
            len(self.domain_to_idx)
        )

    def forward(self, language, domain):
        return self.lang_head(
            language.to(self.lang_head[1].weight.dtype)
        ), self.domain_head(domain.to(self.domain_head[1].weight.dtype))

    def encode_group_ids(self, languages, domains, device):
        return tuple(
            torch.tensor([mapping.get(v, 0) for v in labels], device=device)
            for labels, mapping in [
                (languages, self.lang_to_idx),
                (domains, self.domain_to_idx),
            ]
        )

    def loss(self, embeddings, languages, domains, settings):
        logits = self(*embeddings)
        targets = self.encode_group_ids(languages, domains, logits[0].device)
        loss = logits[0].sum() * 0
        for name, values, labels in zip(("lang", "domain"), logits, targets):
            known = labels > 0
            if known.any():
                loss = loss + float(
                    settings.get(name + "_weight", 1.0)
                ) * F.cross_entropy(values[known], labels[known])
        return float(settings["loss_weight"]) * loss
