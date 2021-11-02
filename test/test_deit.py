import pytest
import toroidal.models
import timm
import torch


def test_deit_transformers():

    classes = [
        (toroidal.models.DeiTBase, timm.models.deit_base_patch16_224),
        (toroidal.models.DeiTTiny, timm.models.deit_tiny_patch16_224),
    ]
    for cls, cls_ref in classes:
        model_ref = cls_ref(pretrained=True)

        model = cls()

        sd = model_ref.state_dict()
        sd["patch_embedding.class_token"] = sd.pop("cls_token")
        sd["patch_embedding.pos_embedding"] = sd.pop("pos_embed")
        sd["patch_embedding.patch_to_vec.weight"] = sd.pop("patch_embed.proj.weight")
        sd["patch_embedding.patch_to_vec.bias"] = sd.pop("patch_embed.proj.bias")

        for k in list(sd.keys()):
            if ".mlp.fc1" in k:
                sd[k.replace(".mlp.fc1", ".mlp.0")] = sd.pop(k)
            elif ".mlp.fc2" in k:
                sd[k.replace(".mlp.fc2", ".mlp.2")] = sd.pop(k)

        model.load_state_dict(sd)

        for training in (True, False):
            model.train(training)
            model_ref.train(training)

            inp = torch.randn(1, 3, 224, 224)
            torch.manual_seed(1234)
            ref_out = model_ref(inp)
            torch.manual_seed(1234)
            out = model(inp)

            torch.testing.assert_allclose(ref_out, out)


if __name__ == "__main__":
    pytest.main([__file__])
