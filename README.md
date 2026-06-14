# ValGeoFuseNet main network

This folder contains only the main network definition for submission.

Use it inside the original project root, where the transformer backbone package
is available:

```python
from valgeofusenet import ValGeoFuseNet

model = ValGeoFuseNet(
    in_channels=1,
    out_channels=29,
    patch_size=4,
    depths=[2, 2, 8],
    num_heads=[4, 8, 16],
    embed_dim=[128, 256, 512],
    topk_small_ratio=0.1,
    topk_large_ratio=0.4,
    use_cross_fusion=True,
)
```

The forward pass returns `(segmentation_logits, sdf_logits)`. The final
experiments use `lambda_reg = 2.0` in the loss, not in the model constructor.
