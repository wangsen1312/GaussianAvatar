import torch
from diff_gaussian_rasterization import GaussianRasterizer, GaussianRasterizationSettings

def main():
    assert torch.cuda.is_available(), "Run this on a GPU node."

    H, W = 64, 64
    device = "cuda"

    # Minimal camera/settings (identity transforms)
    world_view = torch.eye(4, device=device, dtype=torch.float32)
    proj = torch.eye(4, device=device, dtype=torch.float32)
    cam_center = torch.zeros(3, device=device, dtype=torch.float32)

    settings = GaussianRasterizationSettings(
        image_height=H,
        image_width=W,
        tanfovx=1.0,
        tanfovy=1.0,
        bg=torch.zeros(3, device=device, dtype=torch.float32),
        scale_modifier=1.0,
        viewmatrix=world_view,
        projmatrix=proj,
        sh_degree=0,
        campos=cam_center,
        prefiltered=False,
        debug=False,
        antialiasing=True,   # keep same as your training code
    )

    rasterizer = GaussianRasterizer(raster_settings=settings)

    # Minimal gaussian inputs
    N = 8
    means3D = torch.randn(N, 3, device=device, dtype=torch.float32) * 0.1
    means2D = torch.zeros(N, 3, device=device, dtype=torch.float32)  # many impls expect Nx3
    opacities = torch.ones(N, 1, device=device, dtype=torch.float32) * 0.8
    scales = torch.ones(N, 3, device=device, dtype=torch.float32) * 0.02
    rotations = torch.tensor([[1.0, 0.0, 0.0, 0.0]], device=device, dtype=torch.float32).repeat(N, 1)

    # Use precomputed colors to keep it simple
    colors_precomp = torch.rand(N, 3, device=device, dtype=torch.float32)
    shs = None
    cov3D_precomp = None

    out = rasterizer(
        means3D=means3D,
        means2D=means2D,
        shs=shs,
        colors_precomp=colors_precomp,
        opacities=opacities,
        scales=scales,
        rotations=rotations,
        cov3D_precomp=cov3D_precomp,
    )

    # Print what came back
    if not isinstance(out, (tuple, list)):
        print("Returned a single object:", type(out))
        t = out
        print(" shape:", getattr(t, "shape", None),
              "dtype:", getattr(t, "dtype", None),
              "device:", getattr(t, "device", None))
        return

    print(f"Returned tuple/list with len = {len(out)}")
    for i, t in enumerate(out):
        if torch.is_tensor(t):
            print(f"[{i}] Tensor shape={tuple(t.shape)} dtype={t.dtype} device={t.device} "
                  f"min={t.min().item():.4g} max={t.max().item():.4g}")
        else:
            print(f"[{i}] {type(t)}: {t}")

if __name__ == "__main__":
    main()
