import pytest
import toroidal.models
import transformers
import torch


def test_bert_transformers():
    model = toroidal.models.BERT(
        vocab_size=119547, block_size=512, n_hidden=768, n_heads=12, n_blocks=12
    )

    model_ref = transformers.BertModel.from_pretrained("bert-base-multilingual-cased")

    sd = model_ref.state_dict()
    del sd["embeddings.position_ids"]

    sd["embedding.text_embedding.weight"] = sd.pop("embeddings.word_embeddings.weight")
    sd["embedding.pos_embedding"] = sd.pop("embeddings.position_embeddings.weight")[
        None
    ]
    sd["embedding.sentence_number_embedding"] = sd.pop(
        "embeddings.token_type_embeddings.weight"
    )[None]
    sd["embedding.norm.weight"] = sd.pop("embeddings.LayerNorm.weight")
    sd["embedding.norm.bias"] = sd.pop("embeddings.LayerNorm.bias")

    sd["pooler.fc.weight"] = sd.pop("pooler.dense.weight")
    sd["pooler.fc.bias"] = sd.pop("pooler.dense.bias")

    bl_num = 0
    while f"encoder.layer.{bl_num}.attention.self.query.weight" in sd:
        prefix = f"encoder.layer.{bl_num}.attention.self"
        for par in ["weight", "bias"]:
            qkv_par = torch.cat(
                [sd.pop(f"{prefix}.{part}.{par}") for part in ["query", "key", "value"]]
            )
            sd[f"blocks.{bl_num}.attn.qkv.{par}"] = qkv_par
        for newname, hfname in (
            ("norm1", "attention.output.LayerNorm"),
            ("attn.proj", "attention.output.dense"),
            ("norm2", "output.LayerNorm"),
            ("mlp.0", "intermediate.dense"),
            ("mlp.2", "output.dense"),
        ):
            for par in ("weight", "bias"):
                sd[f"blocks.{bl_num}.{newname}.{par}"] = sd.pop(
                    f"encoder.layer.{bl_num}.{hfname}.{par}"
                )

        mask_key = f"blocks.{bl_num}.attn.mask"
        mask = model.state_dict().get(mask_key)
        if mask is not None:
            sd[mask_key] = mask

        bl_num += 1

    model.load_state_dict(sd)

    for training in (True, False):
        model.train(training)
        model_ref.train(training)

        inp = torch.tensor(
            [
                [
                    101,
                    24446,
                    10638,
                    14555,
                    99662,
                    10901,
                    10301,
                    12558,
                    15765,
                    31869,
                    119,
                    102,
                ]
            ]
        )
        torch.manual_seed(1234)
        ref_out = model_ref(inp)
        torch.manual_seed(1234)
        out_pooled = model(inp)
        torch.manual_seed(1234)
        pooler = model.pooler
        model.pooler = torch.nn.Identity()
        out_seq = model(inp)
        model.pooler = pooler

        torch.testing.assert_allclose(ref_out.last_hidden_state, out_seq)
        torch.testing.assert_allclose(ref_out.pooler_output, out_pooled)


if __name__ == "__main__":
    pytest.main([__file__])
