import torch
import torch.nn as nn


class MLPClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim=16, num_layers=2, dropout=0.5, num_classes=2):
        super().__init__()
        layers = []
        current_dim = input_dim
        for _ in range(num_layers):
            layers.extend([
                nn.Linear(current_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            ])
            current_dim = hidden_dim
        layers.append(nn.Linear(current_dim, num_classes))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


class FTTransformer(nn.Module):
    def __init__(
        self, num_cont_features, cat_cardinalities,
        d_model=32, n_heads=4, n_layers=3, dropout=0.5, num_classes=2
    ):
        super().__init__()
        self.num_cont_features = num_cont_features
        self.num_cat_features = len(cat_cardinalities)
        self.d_model = d_model

        self.cont_weight = nn.Parameter(torch.randn(num_cont_features, d_model) * 0.02)
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
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.classifier = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, num_classes),
        )

    def make_feature_tokens(self, x_cont, x_cat):
        cont_tokens = x_cont.unsqueeze(-1) * self.cont_weight.unsqueeze(0)
        cont_tokens = cont_tokens + self.cont_bias.unsqueeze(0)

        cat_tokens = []
        for i, embedding in enumerate(self.cat_embeddings):
            cat_tokens.append(embedding(x_cat[:, i]).unsqueeze(1))

        if cat_tokens:
            return torch.cat([cont_tokens, torch.cat(cat_tokens, dim=1)], dim=1)
        return cont_tokens

    def forward(self, x_cont, x_cat):
        tokens = self.make_feature_tokens(x_cont, x_cat)
        cls_tokens = self.cls_token.repeat(tokens.size(0), 1, 1)
        encoded = self.transformer(torch.cat([cls_tokens, tokens], dim=1))
        return self.classifier(encoded[:, 0, :])


class RobustFTTransformer(FTTransformer):
    def __init__(self, *args, feature_mask_prob=0.2, **kwargs):
        super().__init__(*args, **kwargs)
        self.feature_mask_prob = feature_mask_prob
        self.feature_mask_token = nn.Parameter(torch.zeros(1, 1, self.d_model))

    def apply_random_feature_masking(self, tokens):
        if self.feature_mask_prob <= 0:
            return tokens
        batch_size, num_features, d_model = tokens.shape
        mask = (
            torch.rand(batch_size, num_features, 1, device=tokens.device)
            < self.feature_mask_prob
        )
        mask_token = self.feature_mask_token.expand(batch_size, num_features, d_model)
        return torch.where(mask, mask_token, tokens)

    def apply_external_feature_mask(self, tokens, external_feature_mask):
        if external_feature_mask is None:
            return tokens
        mask = external_feature_mask.unsqueeze(-1).to(tokens.device)
        mask_token = self.feature_mask_token.expand_as(tokens)
        return torch.where(mask, mask_token, tokens)

    def forward(
        self, x_cont, x_cat,
        apply_feature_mask=False,
        external_feature_mask=None,
    ):
        tokens = self.make_feature_tokens(x_cont, x_cat)

        if self.training and apply_feature_mask:
            tokens = self.apply_random_feature_masking(tokens)

        if external_feature_mask is not None:
            tokens = self.apply_external_feature_mask(tokens, external_feature_mask)

        cls_tokens = self.cls_token.repeat(tokens.size(0), 1, 1)
        encoded = self.transformer(torch.cat([cls_tokens, tokens], dim=1))
        return self.classifier(encoded[:, 0, :])
