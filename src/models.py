"""Model definitions used in the manuscript."""

import torch
import torch.nn as nn


class MLPClassifier(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 16,
        num_layers: int = 2,
        dropout: float = 0.5,
        num_classes: int = 2,
    ):
        super().__init__()
        layers = []
        current_dim = input_dim
        for _ in range(num_layers):
            layers.extend(
                [
                    nn.Linear(current_dim, hidden_dim),
                    nn.BatchNorm1d(hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                ]
            )
            current_dim = hidden_dim
        layers.append(nn.Linear(current_dim, num_classes))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


class FTTransformer(nn.Module):
    def __init__(
        self,
        num_cont_features: int,
        cat_cardinalities: list[int],
        d_model: int = 32,
        n_heads: int = 4,
        n_layers: int = 3,
        dropout: float = 0.5,
        num_classes: int = 2,
    ):
        super().__init__()
        self.num_cont_features = num_cont_features
        self.num_cat_features = len(cat_cardinalities)
        self.d_model = d_model

        self.cont_weight = nn.Parameter(
            torch.randn(num_cont_features, d_model) * 0.02
        )
        self.cont_bias = nn.Parameter(torch.zeros(num_cont_features, d_model))
        self.cat_embeddings = nn.ModuleList(
            [nn.Embedding(cardinality, d_model) for cardinality in cat_cardinalities]
        )
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=n_layers,
        )
        self.classifier = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, num_classes),
        )

    def tokenize(self, x_cont, x_cat):
        cont_tokens = x_cont.unsqueeze(-1) * self.cont_weight.unsqueeze(0)
        cont_tokens = cont_tokens + self.cont_bias.unsqueeze(0)

        cat_tokens = [
            embedding(x_cat[:, idx]).unsqueeze(1)
            for idx, embedding in enumerate(self.cat_embeddings)
        ]
        if cat_tokens:
            return torch.cat([cont_tokens, torch.cat(cat_tokens, dim=1)], dim=1)
        return cont_tokens

    def forward(self, x_cont, x_cat):
        tokens = self.tokenize(x_cont, x_cat)
        cls = self.cls_token.expand(tokens.size(0), -1, -1)
        encoded = self.transformer(torch.cat([cls, tokens], dim=1))
        return self.classifier(encoded[:, 0, :])


class RobustFTTransformer(FTTransformer):
    def __init__(self, *args, feature_mask_prob: float = 0.2, **kwargs):
        super().__init__(*args, **kwargs)
        self.feature_mask_prob = feature_mask_prob
        self.feature_mask_token = nn.Parameter(torch.zeros(1, 1, self.d_model))

    def _replace_with_mask_token(self, tokens, mask):
        expanded = self.feature_mask_token.expand_as(tokens)
        return torch.where(mask.unsqueeze(-1).bool(), expanded, tokens)

    def forward(
        self,
        x_cont,
        x_cat,
        apply_feature_mask: bool = True,
        external_feature_mask=None,
    ):
        tokens = self.tokenize(x_cont, x_cat)

        if self.training and apply_feature_mask and self.feature_mask_prob > 0:
            random_mask = (
                torch.rand(tokens.size(0), tokens.size(1), device=tokens.device)
                < self.feature_mask_prob
            )
            tokens = self._replace_with_mask_token(tokens, random_mask)

        if external_feature_mask is not None:
            tokens = self._replace_with_mask_token(tokens, external_feature_mask)

        cls = self.cls_token.expand(tokens.size(0), -1, -1)
        encoded = self.transformer(torch.cat([cls, tokens], dim=1))
        return self.classifier(encoded[:, 0, :])
