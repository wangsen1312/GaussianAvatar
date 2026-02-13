# knn_smoke_check.py
import torch
import diff_gaussian_rasterization as dgr
from simple_knn._C import distCUDA2

def brute_force_nn_dist2(xyz: torch.Tensor) -> torch.Tensor:
    d = torch.cdist(xyz, xyz)
    d.fill_diagonal_(float("inf"))
    nn = d.min(dim=1).values
    return nn * nn

def main():
    print("torch:", torch.__version__, "cuda build:", torch.version.cuda)
    print("cuda available:", torch.cuda.is_available())
    if not torch.cuda.is_available():
        raise SystemExit("Run inside a GPU allocation.")

    print("GPU:", torch.cuda.get_device_name(0))

    # Larger run (performance / smoke)
    N = 20000
    xyz = torch.randn(N, 3, device="cuda", dtype=torch.float32)
    out = distCUDA2(xyz).reshape(-1)
    torch.cuda.synchronize()
    print("distCUDA2 OK:", out.shape, "min/max:", out.min().item(), out.max().item())

    # Numerical check on small subset
    M = 512
    xyz_small = xyz[:M].contiguous()
    out_small = distCUDA2(xyz_small).reshape(-1).contiguous()
    ref_small = brute_force_nn_dist2(xyz_small)

    max_abs_err = (out_small - ref_small).abs().max().item()
    max_rel_err = ((out_small - ref_small).abs() / (ref_small.abs() + 1e-8)).max().item()

    print(f"Check M={M}: max_abs_err={max_abs_err:.3e}, max_rel_err={max_rel_err:.3e}")


    # Just constructing settings + touching CUDA is already a good sign
    # (The real forward call needs many correctly-shaped inputs.)
    settings = dgr.GaussianRasterizationSettings(
        image_height=64,
        image_width=64,
        tanfovx=1.0,
        tanfovy=1.0,
        bg=torch.zeros(3, device="cuda"),
        scale_modifier=1.0,
        viewmatrix=torch.eye(4, device="cuda"),
        projmatrix=torch.eye(4, device="cuda"),
        sh_degree=0,
        campos=torch.zeros(3, device="cuda"),
        antialiasing=False,
        prefiltered=False,
        debug=False,
    )

    rasterizer = dgr.GaussianRasterizer(settings)
    torch.cuda.synchronize()
    print("diff_gaussian_rasterization settings/rasterizer init OK on GPU")

if __name__ == "__main__":
    main()
