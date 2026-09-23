from __future__ import annotations
import os, json, glob, subprocess, shutil, tempfile, argparse, sys
from pathlib import Path
import numpy as np
import laspy

# ========= PARAMS =========
CHUNK_READ     = 200_000_000
WRITE_COLOR    = "zeros"
# ----- sampling/chunking -----
GRID_SIZE      = "0.02"
CHUNK_RANGE    = ("20", "20")
CHUNK_STRIDE   = ("10", "10")
NUM_WORKERS    = "8"

##global variables
GRIDNETHD_ROOT = None
SPLIT_JSON = None
OUT_ROOT = None
POINTCEPT_ROOT = None
TMP_ROOT = None
DINO_V2_PROJ = None
TARGET_CLASSES = 11
# ===================================

# ----- Remap 0..22 + 255 -> 0..10 + 255 -----
REMAP_SPECS = [
    ([0,1,2,3,4], 0),
    ([5],         1),
    ([6,7],       2),
    ([8,9,10,11], 3),
    ([14],        4),
    ([15],        5),
    ([16],        6),
    ([17,18],     7),
    ([19],        8),
    ([20],        9),
    ([21],       10),
    ([12,13,255],255),
]
REMAP = {int(k): int(v) for ks, v in REMAP_SPECS for k in ks}
MAP_11_TO_7 = np.asarray([0, 1, 1, 2, 3, 3, 3, 5, 5, 6, 4], dtype=np.int64)

def remap_labels(seg: np.ndarray) -> np.ndarray:
    out = np.full(seg.shape, 255, dtype=np.int64)
    for u in np.unique(seg):
        out[seg == u] = REMAP.get(int(u), 255)
    if TARGET_CLASSES == 7:
        valid = out != 255
        out[valid] = MAP_11_TO_7[out[valid]]
    return out

def find_lidar_file(zone_dir: Path) -> Path:
    lid = zone_dir / "lidar"
    cands = list(lid.glob("*.las")) + list(lid.glob("*.laz"))
    if not cands:
        raise FileNotFoundError(f"No .las/.laz in {lid}")
    cands.sort(key=lambda p: p.stat().st_size, reverse=True)
    return cands[0]


def write_zone_to_root(dest_root: Path, split: str, zone: str,
                       x: np.ndarray, y: np.ndarray, z: np.ndarray,
                       seg: np.ndarray, rgb: np.ndarray | None):
    out_dir = Path(dest_root) / split / zone
    out_dir.mkdir(parents=True, exist_ok=True)

    min_x, min_y, min_z = x.min(), y.min(), z.min()
    coord = np.stack([
        (x - min_x).astype(np.float32),
        (y - min_y).astype(np.float32),
        (z - min_z).astype(np.float32)
    ], 1)

    np.save(out_dir / "coord.npy", coord)
    np.save(out_dir / "segment.npy", seg.astype(np.int64))
    np.save(out_dir / "supervision_weight.npy",
            np.ones(seg.shape, dtype=np.float32))

    if WRITE_COLOR != "no":
        if rgb is None and WRITE_COLOR == "zeros":
            rgb = np.zeros((coord.shape[0], 3), dtype=np.uint8)
        if rgb is not None:
            if rgb.dtype != np.uint8:
                rgb = np.clip(rgb, 0, 255).astype(np.uint8)
            np.save(out_dir / "color.npy", rgb)

    print(f"[OK] tmp/{split}/{zone}  N={coord.shape[0]}  color={'yes' if rgb is not None else 'no'}")


def prepare_zone_into_tmp(
    zone: str,
    split: str,
    tmp_ds_root: Path,
):
    zone_dir = Path(GRIDNETHD_ROOT) / zone
    las_path = find_lidar_file(zone_dir)
    print(f"[PREP] split={split} zone={zone}  lidar={las_path}")

    xs, ys, zs, segs, rgbs = [], [], [], [], []
    with laspy.open(las_path) as f:
        has_rgb = all(k in f.header.point_format.dimension_names for k in ("red","green","blue"))
        for ch in f.chunk_iterator(CHUNK_READ):
            xs.append(np.asarray(ch.x))
            ys.append(np.asarray(ch.y))
            zs.append(np.asarray(ch.z, dtype=np.float32))
            if "ground_truth" in ch.point_format.dimension_names:
                seg = np.asarray(getattr(ch, "ground_truth"), dtype=np.int64)
            else:
                raise RuntimeError("No label per point (ground_truth).")
            segs.append(remap_labels(seg))

            if WRITE_COLOR != "no":
                if has_rgb:
                    r = np.asarray(ch.red, dtype=np.uint16)
                    g = np.asarray(ch.green, dtype=np.uint16)
                    b = np.asarray(ch.blue, dtype=np.uint32)
                    rgb = np.stack([r, g, b], 1).astype(np.uint32)
                    rgb = (rgb >> 8).astype(np.uint8) #uint 16 to 8
                    rgbs.append(rgb)
                elif WRITE_COLOR == "zeros":
                    rgbs.append(np.zeros((len(ch.x), 3), dtype=np.uint8))

    x = np.concatenate(xs, 0); y = np.concatenate(ys, 0); z = np.concatenate(zs, 0)
    seg = np.concatenate(segs, 0)
    rgb = np.concatenate(rgbs, 0) if rgbs else None

    write_zone_to_root(tmp_ds_root, split, zone, x, y, z, seg, rgb)


# ===== DINOv2 =====
def project_dinov2_for_zone(zone: str, split: str, tmp_ds_root: Path):
    # DINO projection is optional for the LiDAR-only PTv3 baseline.
    from lidar_projection import project_dinov2_to_lidar
    out_dir = Path(tmp_ds_root) / split / zone
    out_dir.mkdir(parents=True, exist_ok=True)
    out_npy = str(out_dir / "dinov2.npy")
    print(f"[DINOv2] {zone} -> {out_npy}")
    project_dinov2_to_lidar(
        zone_name=zone,
        dataset_root=GRIDNETHD_ROOT,
        output_npy_path=out_npy,
        camera_file_name="cameras_pose.txt",
        calib_file_name="camera_calibration.xml",
        batch_size=5_000_000,
        buffer_size=4,
        threshold=1.0,
    )
    print(f"[DINOv2] Saved {out_npy}")


# ===== Chunking =====

def run_chunking_at_root(dataset_root: Path, split_name: str):
    cmd = [
        sys.executable,
        f"{POINTCEPT_ROOT}/pointcept/datasets/preprocessing/sampling_chunking_data.py",
        "--dataset_root", str(dataset_root),
        "--grid_size", GRID_SIZE,
        "--chunk_range", *CHUNK_RANGE,
        "--chunk_stride", *CHUNK_STRIDE,
        "--split", split_name,
        "--num_workers", NUM_WORKERS,
    ]
    print("[RUN]", " ".join(cmd))
    subprocess.run(cmd, check=True)


def find_sampling_output_dir(tmp_root: Path, split: str) -> Path:
    cands = [d for d in Path(tmp_root).iterdir() if d.is_dir() and d.name.startswith(f"{split}_")]
    if cands:
        cands.sort(key=lambda p: p.stat().st_mtime, reverse=True)
        return cands[0]
    # 2) Fallback: recherche récursive (au cas où)
    cands = [d for d in Path(tmp_root).rglob(f"{split}_*") if d.is_dir()]
    if not cands:
        raise FileNotFoundError(f"No folder {tmp_root} pour split={split}")
    cands.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return cands[0]


def move_sampling_output_contents_to_final(output_dir: Path, final_split_root: Path, zone: str):
    final_split_root.mkdir(parents=True, exist_ok=True)
    moved = []
    for child in output_dir.iterdir():
        target_name = f"{zone}__{child.name}"
        target = final_split_root / target_name
        i = 1
        while target.exists():
            target = final_split_root / f"{zone}__{child.stem}__{i}{child.suffix if child.is_file() else ''}"
            i += 1
        shutil.move(str(child), str(target))
        moved.append(target)
    try:
        shutil.rmtree(output_dir)
        print(f"[CLEAN] output samples deleted: {output_dir}")
    except Exception as e:
        print(f"[WARN] Impossible to delete {output_dir}: {e}")
    return moved


def move_chunks_to_final(chunk_dirs: list[Path], final_split_root: Path, zone: str):
    final_split_root.mkdir(parents=True, exist_ok=True)
    moved = []
    for d in chunk_dirs:
        target = final_split_root / f"{zone}__{d.name}"
        i = 1
        while target.exists():
            target = final_split_root / f"{zone}__{d.name}__{i}"
            i += 1
        shutil.move(str(d), str(target))
        moved.append(target)
    return moved


def process_zone(split: str, zone: str):
    print(f"===== ZONE: {zone} ({split}) =====")
    print(TMP_ROOT)
    zone_tmp_root = Path(TMP_ROOT) / f"{split}__{zone}"
    if zone_tmp_root.exists():
        shutil.rmtree(zone_tmp_root)
    (zone_tmp_root / split).mkdir(parents=True, exist_ok=True)

    prepare_zone_into_tmp(zone, split, zone_tmp_root)

    print(DINO_V2_PROJ)
    if DINO_V2_PROJ:
        project_dinov2_for_zone(zone, split, zone_tmp_root)

    run_chunking_at_root(zone_tmp_root, split)

    sampling_out_dir = find_sampling_output_dir(zone_tmp_root, split)

    final_split_root = Path(OUT_ROOT) / f"{split}_final"
    moved = move_sampling_output_contents_to_final(sampling_out_dir, final_split_root, zone)
    print(f"[MOVE] {len(moved)} from {sampling_out_dir} to {final_split_root}")

    try:
        shutil.rmtree(zone_tmp_root)
        print(f"[CLEAN] Temp deleted: {zone_tmp_root}")
    except Exception as e:
        print(f"[WARN] Impossible to delete {zone_tmp_root}: {e}")


def zone_already_processed(split: str, zone: str) -> bool:
    """Return True only when this zone has chunks in the final split."""
    final_split_root = Path(OUT_ROOT) / f"{split}_final"
    return final_split_root.is_dir() and any(
        final_split_root.glob(f"{zone}__*")
    )

def main():
    parser = argparse.ArgumentParser(description="Process GridNet-HD dataset splits.")
    parser.add_argument("--gridnethd_root", type=str, required=True,
                        help="Path to the root folder of the GridNet-HD dataset.")
    parser.add_argument("--split_json", type=str, required=True,
                        help="Path to the split.json file.")
    parser.add_argument("--out_root", type=str, required=True,
                        help="Output directory for the processed data.")
    parser.add_argument("--pointcept_root", type=str, required=True,
                        help="Path to the Pointcept root directory.")
    parser.add_argument("--temporary_root", type=str, required=True,
                        help="Path to a temporary root directory.")
    parser.add_argument("--dino_projection", type=int, required=True,
                        help="Projecting dino feratures into the point cloud? 0:False, 1:True")
    parser.add_argument("--target_classes", type=int, choices=(7, 11), default=11,
                        help="Write the official 11 classes or project 7-class mapping")
    parser.add_argument("--resume", action="store_true",
                        help="Skip zones that already have chunks in the final split")

    args = parser.parse_args()

    global GRIDNETHD_ROOT
    GRIDNETHD_ROOT = args.gridnethd_root
    global SPLIT_JSON
    SPLIT_JSON = args.split_json
    global OUT_ROOT
    OUT_ROOT = args.out_root
    global POINTCEPT_ROOT
    POINTCEPT_ROOT = args.pointcept_root
    global TMP_ROOT
    TMP_ROOT = args.temporary_root
    global DINO_V2_PROJ
    DINO_V2_PROJ = args.dino_projection
    global TARGET_CLASSES
    TARGET_CLASSES = args.target_classes
    Path(OUT_ROOT).mkdir(parents=True, exist_ok=True)

    with open(SPLIT_JSON, "r") as f:
        split = json.load(f)

    split_keys = [k for k in ("train", "val", "test") if split.get(k)]
    for k in split_keys:
        zones = split.get(k, [])
        print(f"\n=== SPLIT: {k}  ({len(zones)} zones) ===")
        for zone in zones:
            if args.resume and zone_already_processed(k, zone):
                print(f"[SKIP] split={k} zone={zone} already processed")
                continue
            process_zone(k, zone)
    print("\n[DONE]")

if __name__ == "__main__":
    main()
