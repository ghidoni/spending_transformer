import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# --- NOTE on Attention --- #
# The paper uses FlashAttention. PyTorch 2.0+ includes a native, efficient
# attention implementation `torch.nn.functional.scaled_dot_product_attention`.
# It automatically uses the most efficient backend available (like FlashAttention)
# if the hardware and inputs are compatible. We will use this for our implementation.


class CausalSelfAttention(nn.Module):
    """A single head of causal self-attention."""

    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.dropout = config.dropout

    def forward(self, x):
        B, T, C = (
            x.size()
        )  # batch size, sequence length, embedding dimensionality (n_embd)
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)

        k = k.view(B, T, self.n_head, C // self.n_head).transpose(
            1, 2
        )  # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(
            1, 2
        )  # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(
            1, 2
        )  # (B, nh, T, hs)

        # Use PyTorch's native scaled_dot_product_attention
        # is_causal=True ensures causal masking
        y = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=None,
            dropout_p=self.dropout if self.training else 0,
            is_causal=True,
        )

        y = (
            y.transpose(1, 2).contiguous().view(B, T, C)
        )  # re-assemble all head outputs side by side
        y = self.c_proj(y)
        return y


class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd)
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd)
        self.dropout = nn.Dropout(config.dropout)
        self.activation = nn.GELU()

    def forward(self, x):
        x = self.c_fc(x)
        x = self.activation(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x


class Block(nn.Module):
    """A single Transformer block."""

    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class CausalTransformer(nn.Module):
    """
    A GPT-like causal transformer model.
    The paper mentions using NoPE (No Positional Embeddings), relying on the causal
    mask to implicitly handle position, which we follow here.
    """

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.transformer = nn.ModuleDict(
            dict(
                wte=nn.Embedding(config.vocab_size, config.n_embd),
                drop=nn.Dropout(config.dropout),
                h=nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
                ln_f=nn.LayerNorm(config.n_embd),
            )
        )
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.transformer.wte.weight = self.lm_head.weight  # Weight tying

    def forward(self, idx, targets=None):
        device = idx.device
        b, t = idx.size()

        tok_emb = self.transformer.wte(idx)
        x = self.transformer.drop(tok_emb)

        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)

        if targets is not None:
            logits = self.lm_head(x)
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1
            )
        else:
            logits = self.lm_head(x[:, [-1], :])  # causal inference-time optimization
            loss = None

        return logits, loss, x  # Return final embeddings as well


# --- Tabular Models (DCNv2 and Periodic Embeddings) --- #


class PeriodicLinearEmbedding(nn.Module):
    """
    Implementation of periodic linear embeddings for numerical features, as described in
    the paper "On embeddings for numerical features in tabular deep learning" (arXiv:2203.05551).
    This was a key component for matching GBT performance.
    """

    def __init__(self, n_features, n_dim_per_feature, sigma=1.0):
        super().__init__()
        self.n_features = n_features
        self.n_dim_per_feature = n_dim_per_feature
        self.sigma = sigma
        # Learnable frequencies and biases
        self.freqs = nn.Parameter(torch.randn(n_features, n_dim_per_feature) * sigma)
        self.biases = nn.Parameter(torch.randn(n_features, n_dim_per_feature))

    def forward(self, x):
        # x shape: (batch_size, n_features)
        x = x.unsqueeze(-1)  # -> (batch_size, n_features, 1)
        # Linear transformation
        linear_trans = (
            self.freqs * x + self.biases
        )  # -> (batch_size, n_features, n_dim_per_feature)
        # Periodic activation (sin)
        periodic_trans = torch.sin(linear_trans)
        return periodic_trans.view(
            x.size(0), -1
        )  # -> (batch_size, n_features * n_dim_per_feature)


class CrossLayer(nn.Module):
    """A single cross layer for DCNv2."""

    def __init__(self, in_features):
        super().__init__()
        self.linear = nn.Linear(in_features, in_features)

    def forward(self, x0, x):
        # x0 is the original input, x is the input from the previous layer
        prod = self.linear(x)
        return x0 * prod + x


class DCNv2(nn.Module):
    """
    Implementation of Deep & Cross Network V2, as per paper arXiv:2008.13535.
    """

    def __init__(self, in_features, deep_layer_sizes, num_cross_layers):
        super().__init__()
        if isinstance(deep_layer_sizes, tuple):
            deep_layer_sizes = list(deep_layer_sizes)
        # Deep part
        deep_layers = []
        for in_size, out_size in zip(
            [in_features] + deep_layer_sizes[:-1], deep_layer_sizes
        ):
            deep_layers.append(nn.Linear(in_size, out_size))
            deep_layers.append(nn.ReLU())
        self.deep_net = nn.Sequential(*deep_layers)

        # Cross part
        self.cross_net = nn.ModuleList(
            [CrossLayer(in_features) for _ in range(num_cross_layers)]
        )

        # Expose the combined feature embedding (deep_out || cross_out) dimension
        # so downstream models can project it as needed. If a logit is desired,
        # add a separate head outside this module.
        self.output_dim = deep_layer_sizes[-1] + in_features

    def forward(self, x_deep, x_cross):
        # x_deep is input to deep net, x_cross is input to cross net (can be the same)
        deep_out = self.deep_net(x_deep)

        x0 = x_cross
        x = x_cross
        for layer in self.cross_net:
            x = layer(x0, x)

        combined = torch.cat([deep_out, x], dim=1)
        # Return feature embedding; caller can apply its own projection/head
        return combined


# --- Final Joint Fusion Model --- #


class NuFormer(nn.Module):
    """
    The final nuFormer model implementing joint fusion.
    It combines the CausalTransformer for sequential data and a modified DCNv2 for tabular data.
    """

    def __init__(self, transformer_config, tabular_feature_info, dcn_config):
        super().__init__()
        # 1. Sequential part
        self.transformer = CausalTransformer(transformer_config)

        # 2. Tabular part
        self.numerical_cols = tabular_feature_info["numerical_cols"]
        self.categorical_cols = tabular_feature_info["categorical_cols"]

        # Categorical embeddings
        self.cat_embeddings = nn.ModuleDict()
        cat_embed_dim = 16  # A reasonable default
        for col, cardinality in tabular_feature_info["cat_cardinalities"].items():
            self.cat_embeddings[col] = nn.Embedding(cardinality, cat_embed_dim)

        total_cat_embed_dim = len(self.categorical_cols) * cat_embed_dim

        # Numerical embeddings (Periodic)
        self.numerical_embedder = PeriodicLinearEmbedding(
            n_features=len(self.numerical_cols),
            n_dim_per_feature=dcn_config["numerical_embed_dim"],
        )
        total_num_embed_dim = (
            len(self.numerical_cols) * dcn_config["numerical_embed_dim"]
        )

        # DCNv2 for tabular features
        # The paper mentions a modification: DCNv2 processes only tabular features
        # and projects to a low-dimensional embedding.
        dcn_in_features = total_cat_embed_dim + total_num_embed_dim
        self.dcn = DCNv2(
            in_features=dcn_in_features,
            deep_layer_sizes=dcn_config["deep_layers"],
            num_cross_layers=dcn_config["cross_layers"],
        )
        dcn_out_features = dcn_config["deep_layers"][-1] + dcn_in_features
        self.tabular_projection = nn.Linear(
            dcn_out_features, dcn_config["projection_dim"]
        )

        # 3. Final Classifier MLP
        # This MLP takes the concatenated embeddings from the transformer and tabular model
        final_mlp_input_dim = transformer_config.n_embd + dcn_config["projection_dim"]
        self.classifier = nn.Sequential(
            nn.LayerNorm(final_mlp_input_dim),
            nn.Linear(final_mlp_input_dim, final_mlp_input_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(final_mlp_input_dim // 2, 1),
        )

    def forward(self, seq_data, tabular_data):
        # 1. Process sequential data
        _, _, transformer_embeddings = self.transformer(seq_data)
        # Use the embedding of the last token as the user representation
        seq_embedding = transformer_embeddings[:, -1, :]

        # 2. Process tabular data
        # Categorical
        cat_embeds = []
        for i, col in enumerate(self.categorical_cols):
            # Ensure integer indices for embedding lookup
            cat_embeds.append(
                self.cat_embeddings[col](tabular_data[:, i].to(dtype=torch.long))
            )
        x_cat = torch.cat(cat_embeds, dim=1)

        # Numerical
        x_num = self.numerical_embedder(tabular_data[:, len(self.categorical_cols) :])

        # Combine and pass through DCN
        x_tabular = torch.cat([x_cat, x_num], dim=1)
        tabular_dcn_out = self.dcn(x_tabular, x_tabular)
        tabular_embedding = self.tabular_projection(tabular_dcn_out)

        # 3. Concatenate and classify
        combined_embedding = torch.cat([seq_embedding, tabular_embedding], dim=1)
        logits = self.classifier(combined_embedding)

        return logits
