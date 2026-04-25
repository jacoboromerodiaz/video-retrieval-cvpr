## TODO

- [ ] batch size > 1

## Known Issues

### V-JEPA 2 hub cache patch

After the first run, `torch.hub` caches the vjepa2 repo locally with a hardcoded
internal Meta URL that breaks weight downloads. Patch it:

```bash
sed -i '' 's|http://localhost:8300|https://dl.fbaipublicfiles.com/vjepa2|g' \
    ~/.cache/torch/hub/facebookresearch_vjepa2_main/src/hub/backbones.py
```

