![](docs/source/_static/logo/toroidal-logo-rendered.svg)

# toroidal - a lightweight transformer library for PyTorch

Toroidal transformers are of smaller size and lower weight than the more common E-I types. This is the software equivalent.

This is a small and educational project, not big and professional like the `transformers` library.

- Simplicity! We only cover very popular transformer types and keep implementations things simple and beautiful.

- Hightlight similarities. We try not to copy-paste code between almost identical implementations.

- PyTorch first and only. We do not provide everything for everyone, but focus on PyTorch and use PyTorchy coding style. Ideally, our models work well with TorchScript.

## Important Note

For the time being, we will emphasize beautiful code over backward compatibility, so things will break. Don't use it for things you cannot fix.

## FAQ

Why not [minGPT](https://github.com/karpathy/minGPT/)? I love minGPT, but I needed BERT.

Why not [transformers](https://github.com/huggingface/transformers/)? I wanted small, but if you have to ask, you should use them instead.
