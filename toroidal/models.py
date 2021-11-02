import torch
from typing import Optional


class VisionPatchEmbedding(torch.nn.Module):
    def __init__(self, input_size, patch_size, n_channels):
        super().__init__()
        self.n_channels = n_channels
        num_patches = (input_size ** 2) // (
            patch_size ** 2
        )  # only when this is divisible...
        self.patch_to_vec = torch.nn.Conv2d(
            3,
            n_channels,
            kernel_size=(patch_size, patch_size),
            stride=(patch_size, patch_size),
        )  # equivalent to reshape + linear
        self.class_token = torch.nn.Parameter(
            torch.randn(1, 1, n_channels)
        )  # "global information"
        self.pos_embedding = torch.nn.Parameter(
            torch.randn(1, num_patches + 1, n_channels)
        )

        torch.nn.init.zeros_(self.patch_to_vec.bias)
        torch.nn.init.normal_(self.patch_to_vec.bias, std=0.02)  # scale with size?
        torch.nn.init.normal_(self.class_token, std=0.02)
        torch.nn.init.normal_(self.pos_embedding, std=0.02)

    def forward(self, x):
        bs = x.size(0)
        x = self.patch_to_vec(x)
        # batch x channels x num_patches_y x num_patches_x ---> batch x patches x n_channels
        x = x.view(bs, self.n_channels, -1).permute(0, 2, 1)
        x = torch.cat([self.class_token.expand(bs, -1, -1), x], dim=1)
        x = x + self.pos_embedding
        return x


class TextEmbedding(torch.nn.Module):
    def __init__(self, block_size=128, n_features=512, vocab_size=26, drop_p=0.0):
        super().__init__()
        self.n_features = n_features
        self.block_size = block_size
        self.text_embedding = torch.nn.Embedding(vocab_size, n_features)
        self.pos_embedding = torch.nn.Parameter(torch.zeros(1, block_size, n_features))
        if drop_p > 0.0:
            self.drop = torch.nn.Dropout(drop_p)
        else:  # pragma: no cover
            self.drop = None
        torch.nn.init.normal_(self.text_embedding.weight, std=0.02)
        torch.nn.init.normal_(self.pos_embedding, std=0.02)

    def forward(self, x):
        bs, seq_len = x.shape
        if seq_len > self.block_size:  # pragma: no cover
            raise ValueError(
                f"passed sequence of length {seq_len} is too long, only {self.block_size} is allowed"
            )

        x = self.text_embedding(x)
        x = x + self.pos_embedding[:, :seq_len]
        if self.drop is not None:
            x = self.drop(x)
        return x


class BERTEmbedding(torch.nn.Module):
    def __init__(self, block_size=128, n_features=512, vocab_size=26, drop_p=0.0):
        super().__init__()
        self.n_features = n_features
        self.block_size = block_size
        self.text_embedding = torch.nn.Embedding(vocab_size, n_features)
        self.pos_embedding = torch.nn.Parameter(torch.zeros(1, block_size, n_features))
        # BERT is trained on two sentences
        self.sentence_number_embedding = torch.nn.Parameter(
            torch.zeros(1, 2, n_features)
        )
        self.norm = torch.nn.LayerNorm(n_features, eps=1e-12)
        if drop_p > 0.0:
            self.drop = torch.nn.Dropout(drop_p)
        else:  # pragma: no cover
            self.drop = None
        torch.nn.init.ones_(self.norm.weight)
        torch.nn.init.zeros_(self.norm.bias)
        torch.nn.init.normal_(self.text_embedding.weight, std=0.02)
        torch.nn.init.normal_(self.pos_embedding, std=0.02)
        torch.nn.init.normal_(self.sentence_number_embedding, std=0.02)

    def forward(self, x, sentence_nums: Optional[torch.Tensor] = None):
        bs, seq_len = x.shape
        if seq_len > self.block_size:  # pragma: no cover
            raise ValueError(
                f"passed sequence of length {seq_len} is too long, only {self.block_size} is allowed"
            )

        x = self.text_embedding(x)
        x = x + self.pos_embedding[:, :seq_len]
        if sentence_nums is None:
            x = x + self.sentence_number_embedding[:, :1]  # broadcast "sentence A"
        else:  # pragma: no cover
            raise NotImplementedError("Feeding two sentences is not yet implemented")
        x = self.norm(x)
        if self.drop is not None:
            x = self.drop(x)
        return x


class FastGELU(torch.nn.Module):
    # This is for HF GPT models. We may or may not offer a config option to switch
    # the speedup is rather dubious
    def forward(self, x):
        # sqrt(2/pi) = 0.7978845608028654
        return (
            0.5
            * x
            * (
                1.0
                + torch.tanh(0.7978845608028654 * (x + 0.044715 * torch.pow(x, 3.0)))
            )
        )


class MLP(torch.nn.Sequential):
    # N.B.: timm also has a dropout layer between gelu and fc2
    #       but vit and deit seem to use no dropout and we want to be compatible
    #       with (A. Karpathy's mingpt implementation of) GPT2
    def __init__(self, n_channels, n_hidden, drop_p=0.0):
        modules = [
            torch.nn.Linear(n_channels, n_hidden),
            torch.nn.GELU(),
            torch.nn.Linear(n_hidden, n_channels),
        ]
        if drop_p > 0.0:
            modules.append(torch.nn.Dropout(drop_p))
        super().__init__(*modules)
        torch.nn.init.normal_(self[0].weight, std=0.02)
        torch.nn.init.zeros_(self[0].bias)
        torch.nn.init.normal_(self[2].weight, std=0.02)
        torch.nn.init.zeros_(self[2].bias)


class Attention(torch.nn.Module):
    # big attention has dropout, too, sometimes qkv w/o bias
    def __init__(
        self,
        n_channels,
        n_heads,
        drop_attn_p=0.0,
        drop_out_p=0.0,
        block_size=128,
        causal=False,
    ):
        super().__init__()
        self.n_channels = n_channels
        self.n_heads = n_heads
        self.scale = (n_channels // n_heads) ** -0.5
        self.qkv = torch.nn.Linear(n_channels, 3 * n_channels)
        if causal:
            self.register_buffer(
                "mask", torch.tril(torch.ones(1, 1, block_size, block_size)).log()
            )
        else:
            self.mask = None
        if drop_attn_p > 0.0:
            self.dropout_attn = torch.nn.Dropout(drop_attn_p)
        else:
            self.dropout_attn = None
        self.proj = torch.nn.Linear(n_channels, n_channels)
        if drop_out_p > 0.0:
            self.dropout_out = torch.nn.Dropout(drop_out_p)
        else:
            self.dropout_out = None
        torch.nn.init.normal_(self.qkv.weight, std=0.02)
        torch.nn.init.zeros_(self.qkv.bias)
        torch.nn.init.normal_(self.proj.weight, std=0.02)
        torch.nn.init.zeros_(self.proj.bias)

    def forward(self, x):
        bs, n_locations, n_channels = x.shape
        n_heads = self.n_heads

        # this is a trick to do q, k, v in one large linear for efficiency
        # unbind splits a tensor into a tuple of tensors
        q, k, v = self.qkv(x).view(bs, n_locations, 3, n_heads, -1).unbind(dim=2)

        logits = torch.einsum("bthc,bshc->bhts", q, k)
        logits *= self.scale  # normalize against staturation
        if self.mask is not None:
            logits += (
                self.mask
            )  # this relies on logits being less than inf, but I guess that is OK
        attn = logits.softmax(-1)
        if self.dropout_attn is not None:
            attn = self.dropout_attn(attn)
        output = torch.einsum("bhts,bshc->bthc", attn, v)  # target source
        output = output.reshape(bs, n_locations, n_channels)  # recombine
        output = self.proj(output)
        if self.dropout_out is not None:
            output = self.dropout_out(output)
        return output


class Block(torch.nn.Module):
    # works for GPT and DeiT
    def __init__(
        self,
        n_channels,
        n_heads,
        n_hidden,
        drop_attn_p=0.0,
        drop_attn_out_p=0.0,
        drop_mlp_p=0.0,
        block_size=None,
        causal=False,
        ln_eps=1e-6,
    ):
        super().__init__()
        self.norm1 = torch.nn.LayerNorm(n_channels, eps=ln_eps)
        self.attn = Attention(
            n_channels,
            n_heads,
            drop_attn_p=drop_attn_p,
            drop_out_p=drop_attn_out_p,
            causal=causal,
            block_size=block_size,
        )
        self.norm2 = torch.nn.LayerNorm(n_channels, eps=ln_eps)
        self.mlp = MLP(n_channels, n_hidden, drop_p=drop_mlp_p)
        torch.nn.init.ones_(self.norm1.weight)
        torch.nn.init.zeros_(self.norm1.bias)
        torch.nn.init.ones_(self.norm2.weight)
        torch.nn.init.zeros_(self.norm2.bias)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class BERTBlock(Block):
    # same ingredients as Block, just slightly different forward
    def forward(self, x):
        x = self.norm1(x + self.attn(x))
        x = self.norm2(x + self.mlp(x))
        return x


class DeiTTiny(torch.nn.Module):
    def __init__(self, n_classes=1000):
        super().__init__()
        input_size = 224
        patch_size = 16
        n_channels = 192  # 64 * 3
        n_heads = 3
        n_hidden = 768
        n_blocks = 12
        self.patch_embedding = VisionPatchEmbedding(input_size, patch_size, n_channels)
        self.blocks = torch.nn.Sequential(
            *[
                Block(n_channels, n_heads, n_hidden, ln_eps=1e-6)
                for i in range(n_blocks)
            ]
        )
        self.norm = torch.nn.LayerNorm(n_channels, eps=1e-6)
        self.head = torch.nn.Linear(n_channels, n_classes)
        torch.nn.init.ones_(self.norm.weight)
        torch.nn.init.zeros_(self.norm.bias)
        torch.nn.init.normal_(self.head.weight, std=0.02)  # scale with size?
        torch.nn.init.zeros_(self.head.bias)

    def forward(self, x):
        x = self.patch_embedding(x)
        x = self.blocks(x)
        x = self.norm(x)
        x = self.head(x[:, 0])
        return x


class DeiTBase(torch.nn.Module):
    def __init__(self, n_classes=1000):
        super().__init__()
        input_size = 224
        patch_size = 16
        n_channels = 768
        n_heads = 12
        n_hidden = 3072
        n_blocks = 12
        self.patch_embedding = VisionPatchEmbedding(input_size, patch_size, n_channels)
        self.blocks = torch.nn.Sequential(
            *[
                Block(n_channels, n_heads, n_hidden, ln_eps=1e-6)
                for i in range(n_blocks)
            ]
        )
        self.norm = torch.nn.LayerNorm(n_channels, eps=1e-6)
        self.head = torch.nn.Linear(n_channels, n_classes)
        torch.nn.init.ones_(self.norm.weight)
        torch.nn.init.zeros_(self.norm.bias)
        torch.nn.init.normal_(self.head.weight, std=0.02)  # scale with size?
        torch.nn.init.zeros_(self.head.bias)

    def forward(self, x):
        x = self.patch_embedding(x)
        x = self.blocks(x)
        x = self.norm(x)
        x = self.head(x[:, 0])
        return x


class GPT(torch.nn.Module):
    def __init__(
        self, *, vocab_size, block_size, n_hidden, n_heads, n_blocks, n_mlp_hidden=None
    ):
        super().__init__()
        if n_mlp_hidden is None:
            n_mlp_hidden = 4 * n_hidden
        self.embedding = TextEmbedding(
            block_size=block_size,
            n_features=n_hidden,
            vocab_size=vocab_size,
            drop_p=0.1,
        )
        self.blocks = torch.nn.Sequential(
            *[
                Block(
                    n_hidden,
                    n_heads,
                    n_mlp_hidden,
                    drop_attn_p=0.1,
                    drop_attn_out_p=0.1,
                    drop_mlp_p=0.1,
                    block_size=block_size,
                    causal=True,
                    ln_eps=1e-5,
                )
                for i in range(n_blocks)
            ]
        )
        self.norm = torch.nn.LayerNorm(n_hidden, eps=1e-5)
        self.head = torch.nn.Linear(n_hidden, vocab_size, bias=False)
        torch.nn.init.ones_(self.norm.weight)
        torch.nn.init.zeros_(self.norm.bias)
        torch.nn.init.normal_(self.head.weight, std=0.02)  # scale with size?

    def forward(self, x):
        x = self.embedding(x)
        x = self.blocks(x)
        x = self.norm(x)
        x = self.head(x)
        return x


class BERTPooler(torch.nn.Module):
    def __init__(self, n_hidden):
        super().__init__()
        self.fc = torch.nn.Linear(n_hidden, n_hidden, bias=True)
        torch.nn.init.normal_(self.fc.weight, std=0.02)  # scale with size?
        torch.nn.init.zeros_(self.fc.bias)

    def forward(self, x):
        x = x[..., 0, :]  # put output at CLS token
        x = self.fc(x)
        x = torch.tanh(x)
        return x


class BERT(torch.nn.Module):
    def __init__(
        self, *, vocab_size, block_size, n_hidden, n_heads, n_blocks, n_mlp_hidden=None
    ):
        super().__init__()
        if n_mlp_hidden is None:
            n_mlp_hidden = 4 * n_hidden
        self.embedding = BERTEmbedding(
            block_size=block_size,
            n_features=n_hidden,
            vocab_size=vocab_size,
            drop_p=0.1,
        )
        self.blocks = torch.nn.Sequential(
            *[
                BERTBlock(
                    n_hidden,
                    n_heads,
                    n_mlp_hidden,
                    drop_attn_p=0.1,
                    drop_attn_out_p=0.1,
                    drop_mlp_p=0.1,
                    block_size=block_size,
                    causal=False,
                    ln_eps=1e-12,
                )
                for i in range(n_blocks)
            ]
        )
        self.pooler = BERTPooler(n_hidden)

    def forward(self, x):
        x = self.embedding(x)
        x = self.blocks(x)
        x = self.pooler(x)
        return x
