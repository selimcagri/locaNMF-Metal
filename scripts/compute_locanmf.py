import os, sys, time, argparse, numpy as np, scipy.io as sio, torch
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import locanmf.LocaNMF as L
from locanmf.video import LowRankVideo, RegionMetadata
from locanmf.LocaNMF import extract_region_metadata, factor_region_videos, rank_linesearch

def loadmat(p): return sio.loadmat(p, squeeze_me=True, struct_as_record=False)

ap = argparse.ArgumentParser()
ap.add_argument("--data_dir", required=True)      # contains Vc_Uc.mat and atlas.mat
ap.add_argument("--vcuc", default="Vc_Uc.mat")
ap.add_argument("--atlas", default="atlas.mat")
ap.add_argument("--ds", type=int, default=1)      # ds=1 (no downsample)
ap.add_argument("--loc_thresh", type=int, default=20)
ap.add_argument("--rank_min", type=int, default=70)
ap.add_argument("--rank_max", type=int, default=120)
ap.add_argument("--rank_step", type=int, default=5)
ap.add_argument("--device", default="cuda")       # "cuda" | "cpu"
args = ap.parse_args()

D = args.data_dir
arr   = loadmat(os.path.join(D, args.vcuc))
atlas = loadmat(os.path.join(D, args.atlas))
Uc    = np.asarray(arr["Uc"]).astype(np.float32)              # (Y,X,r)
Vc    = np.asarray(arr["Vc"]).astype(np.float32)              # (r,T)
mask  = np.asarray(arr["brainmask"]).astype(bool)             # (Y,X)
atlas_i = np.asarray(atlas["atlas"]).astype(np.int32)         # (Y,X)
areanames = atlas.get("areanames", None)

# Optional: per-component energy normalize Vc (compensate in U)
row_std = np.sqrt(np.mean(Vc**2, axis=1, keepdims=True)) + 1e-8
Vc /= row_std; Uc *= row_std.reshape(1,1,-1)

# Downsample (we’ll keep ds=1 for full fidelity)
if args.ds > 1: Vc = Vc[:, ::args.ds]

Y,X,r = Uc.shape; mask_flat = mask.reshape(-1)
U_flat = Uc.reshape(-1, r)[mask_flat, :]
n_pix, T = U_flat.shape[0], Vc.shape[1]

device = torch.device(args.device if (args.device=="cuda" and torch.cuda.is_available()) else "cpu")
lv = LowRankVideo((n_pix, r, T), device=device)
lv.set(torch.from_numpy(U_flat.T).to(device), torch.from_numpy(Vc).to(device))

sup_np, dist_np, labels_np = extract_region_metadata(mask, atlas_i, min_size=100)
region_meta = RegionMetadata(sup_np.shape[0], sup_np.shape[1:], device=device)
region_meta.support._data  = torch.from_numpy(sup_np.astype(bool)).to(device)
region_meta.distance._data = torch.from_numpy(dist_np.astype(np.float32)).to(device)
region_meta.labels._data   = torch.from_numpy(labels_np.astype(np.int64)).to(device)
region_videos = factor_region_videos((U_flat, Vc), sup_np, max_rank=args.rank_max, device=device)

comps = rank_linesearch(
    lv, region_meta, region_videos,
    rank_range=(args.rank_min, args.rank_max, args.rank_step),
    maxiter_rank=(args.rank_max-args.rank_min)//args.rank_step + 1,
    maxiter_lambda=6, maxiter_hals=6,
    lambda_step=1.35, lambda_init=1e-6,
    loc_thresh=args.loc_thresh, r2_thresh=0.99,
    nnt=False, verbose=[True, False, False], sample_prop=(1,1),
    device=device
)

# Save a single portable bundle (like the demo notebook already does)
K = int(comps.num_components)
A = comps.spatial.data.detach().cpu().numpy().T              # (P,K)
C = comps.temporal.data.detach().cpu().numpy().astype(np.float32)  # (K,T)
A_cube = np.full((Y, X, K), np.nan, np.float32)
A_cube.reshape(-1, K)[mask_flat, :] = A.astype(np.float32)

out_path = os.path.join(D, f"locanmf_decomp_loc{args.loc_thresh}.mat")
sio.savemat(out_path, {
    "C": C,
    "A": A_cube,
    "areas": labels_np[comps.regions.data.cpu().numpy().astype(int)].astype(np.int32),
    "areanames": areanames
}, do_compression=False)
print("Saved:", out_path)
