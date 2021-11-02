import pytest
import toroidal.models
import transformers
import torch
import mingpt.model


def test_gpt_mingpt():
    cfg = mingpt.model.GPTConfig(65, 128, n_layer=8, n_head=8, n_embd=512)
    ref_model = mingpt.model.GPT(cfg)
    model = toroidal.models.GPT(
        n_heads=8, n_hidden=512, n_blocks=8, block_size=128, vocab_size=65
    )

    sd = ref_model.state_dict()
    sd["embedding.pos_embedding"] = sd.pop("pos_emb")
    sd["embedding.text_embedding.weight"] = sd.pop("tok_emb.weight")

    bl_num = 0
    while f"blocks.{bl_num}.attn.query.weight" in sd:
        prefix = f"blocks.{bl_num}.attn"
        for par in ["weight", "bias"]:
            qkv_par = torch.cat(
                [sd.pop(f"{prefix}.{part}.{par}") for part in ["query", "key", "value"]]
            )
            sd[f"{prefix}.qkv.{par}"] = qkv_par
        bl_num += 1

    for k in list(sd.keys()):
        if ".ln" in k:
            sd[k.replace(".ln", ".norm")] = sd.pop(k)
        elif ".mask" in k:
            sd[k] = model.state_dict()[k]
    sd["norm.weight"] = sd.pop("ln_f.weight")
    sd["norm.bias"] = sd.pop("ln_f.bias")

    model.load_state_dict(sd)

    for training in (True, False):
        model.train(training)
        ref_model.train(training)

        inp = torch.arange(128).view(1, -1) % 65
        torch.manual_seed(1234)
        ref_out = ref_model(inp)
        torch.manual_seed(1234)
        out = model(inp)

        torch.testing.assert_allclose(ref_out[0], out)


def test_gpt_transformers():
    cfg = transformers.GPT2Config()
    ref_model = transformers.GPT2LMHeadModel(cfg)
    model = toroidal.models.GPT(
        vocab_size=cfg.vocab_size,
        block_size=cfg.n_positions,
        n_hidden=cfg.n_embd,
        n_heads=cfg.n_head,
        n_blocks=cfg.n_layer,
    )

    for bl in model.blocks:
        bl.mlp[1] = toroidal.models.FastGELU()

    sd = ref_model.state_dict()

    for k in list(sd.keys()):
        if k.startswith("transformer."):
            sd[k.replace("transformer.", "")] = sd.pop(k)

    sd["embedding.pos_embedding"] = sd.pop("wpe.weight")[None]
    sd["embedding.text_embedding.weight"] = sd.pop("wte.weight")

    for k in list(sd.keys()):
        if k.startswith("h."):
            sd[k.replace("h.", "blocks.")] = sd.pop(k)

    for k, v in model.state_dict().items():
        if k.endswith(".mask"):
            sd[k] = v.clone()  # HF uses -10.000 instead of -inf

    for k in list(sd.keys()):
        if ".ln_" in k:
            sd[k.replace(".ln_", ".norm")] = sd.pop(k)
        elif ".mlp.c_fc" in k:
            sd[k.replace(".mlp.c_fc", ".mlp.0")] = sd.pop(k).t()
        elif ".mlp.c_proj" in k:
            sd[k.replace(".mlp.c_proj", ".mlp.2")] = sd.pop(k).t()
        elif ".attn.c_proj" in k:
            sd[k.replace(".attn.c_proj", ".attn.proj")] = sd.pop(k).t()
        elif ".attn.c_attn" in k:
            sd[k.replace(".attn.c_attn", ".attn.qkv")] = sd.pop(k).t()
        elif k.startswith("lm_head."):
            sd[k.replace("lm_head.", "head.")] = sd.pop(k)
        elif k.startswith("ln_f."):
            sd[k.replace("ln_f.", "norm.")] = sd.pop(k)

    for k in list(sd.keys()):
        if k.endswith(".attn.bias") or k.endswith(".attn.masked_bias"):
            del sd[k]

    model.load_state_dict(sd)

    for training in (True, False):
        model.train(training)
        ref_model.train(training)

        inp = torch.arange(1024).view(1, -1)
        torch.manual_seed(1234)
        ref_out = ref_model(inp)
        torch.manual_seed(1234)
        out = model(inp)

        torch.testing.assert_allclose(ref_out.logits, out)


if __name__ == "__main__":
    pytest.main([__file__])
