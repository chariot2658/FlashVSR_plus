"""Stitch the tile videos of a `--no-stitch` run and write raw RGB24 frames to stdout.

    python -m vsrlib.stitch_tiles <out_dir>/tiles.json | ffmpeg -f rawvideo -pix_fmt rgb24 -s WxH ...

The frames are the native stitched size (final_dims in tiles.json), or --resize
W H, which matches the CLI's --output-height/--output-width downscale (torch's
"area" mode: adaptive average pooling). Colour conversion is left to the ffmpeg
that reads them. The blend is the same feather-mask average as
tiling.stitch_video_tiles, but rounded rather than truncated to uint8.
Deliberately torch-free: importing torch costs ~0.9 GiB of commit, and this
runs next to a GPU process.
"""
import argparse
import itertools
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor

import imageio
import numpy as np


def feather_mask(h, w, overlap):
    # Same ramps as tiling.create_feather_mask_numpy.
    mask = np.ones((h, w, 1), dtype=np.float32)
    ramp = np.linspace(0, 1, overlap, dtype=np.float32)
    mask[:, :overlap, :] *= ramp[np.newaxis, :, np.newaxis]
    mask[:, -overlap:, :] *= np.flip(ramp)[np.newaxis, :, np.newaxis]
    mask[:overlap, :, :] *= ramp[:, np.newaxis, np.newaxis]
    mask[-overlap:, :, :] *= np.flip(ramp)[:, np.newaxis, np.newaxis]
    return mask


def area_resize_axis(x, out_len, axis):
    """torch.nn.functional.interpolate(mode="area") along one axis: output i
    averages input [floor(i*n/out), ceil((i+1)*n/out))."""
    n = x.shape[axis]
    if n == out_len:
        return x
    i = np.arange(out_len)
    start = (i * n) // out_len
    end = -((-(i + 1) * n) // out_len)
    count = end - start
    shape = [1] * x.ndim
    shape[axis] = out_len
    # Windows are a few pixels wide: add up shifted gathers, masking past each end.
    out = np.take(x, start, axis=axis)
    for k in range(1, int(count.max())):
        idx = np.minimum(start + k, n - 1)
        term = np.take(x, idx, axis=axis)
        if (count <= k).any():
            term *= (count > k).astype(np.float32).reshape(shape)
        out += term
    out /= count.astype(np.float32).reshape(shape)
    return out


def stitched_frames(manifest_path, chunk_size=4, resize=None):
    """Yield the stitched frames (H, W, 3) uint8, exactly frame_count of them."""
    with open(manifest_path, encoding="utf-8") as f:
        m = json.load(f)
    tiles_dir = os.path.join(os.path.dirname(os.path.abspath(manifest_path)), "tiles")
    final_w, final_h = m["final_dims"]
    scale, overlap, total = m["scale"], m["overlap"], m["frame_count"]

    placements = []
    weight = np.zeros((final_h, final_w, 1), dtype=np.float32)
    for x1, y1, x2, y2 in m["coords"]:
        th, tw = (y2 - y1) * scale, (x2 - x1) * scale
        mask = feather_mask(th, tw, overlap * scale)
        oy, ox = y1 * scale, x1 * scale
        weight[oy:oy + th, ox:ox + tw] += mask
        placements.append((oy, ox, th, tw, mask))
    weight[weight == 0] = 1.0
    inv_weight = (1.0 / weight).astype(np.float32)

    readers = [imageio.get_reader(os.path.join(tiles_dir, name)) for name in m["tiles"]]
    # Each tile decodes in its own ffmpeg process; pulling the next chunk of every
    # tile on a thread keeps them all decoding instead of one pipe at a time.
    pool = ThreadPoolExecutor(max_workers=len(readers))
    try:
        iters = [r.iter_data() for r in readers]
        canvas = np.empty((chunk_size, final_h, final_w, 3), dtype=np.float32)

        def take(it, n):
            frames = list(itertools.islice(it, n))
            return np.stack(frames) if frames else np.empty((0, 0, 0, 3), np.uint8)

        def fetch(n):
            return [pool.submit(take, it, n) for it in iters] if n else []

        pending = fetch(min(chunk_size, total))
        for start in range(0, total, chunk_size):
            n = min(chunk_size, total - start)
            chunks = [f.result() for f in pending]
            pending = fetch(min(chunk_size, total - start - n))
            c = canvas[:n]
            c.fill(0.0)
            for k, (frames, (oy, ox, th, tw, mask)) in enumerate(zip(chunks, placements)):
                if len(frames) != n:
                    raise RuntimeError(f"tile {m['tiles'][k]} ended early at frame {start + len(frames)} of {total}")
                tile = frames.astype(np.float32)
                if tile.shape[1:3] != (th, tw):
                    raise RuntimeError(f"tile {m['tiles'][k]} is {tile.shape[2]}x{tile.shape[1]}, expected {tw}x{th}")
                tile *= mask
                c[:, oy:oy + th, ox:ox + tw] += tile
            c *= inv_weight
            for frame in c:
                if resize is not None:
                    # Separable: a rectangle's average is the average of its row averages.
                    frame = area_resize_axis(area_resize_axis(frame, resize[1], 0), resize[0], 1)
                yield np.rint(np.clip(frame, 0.0, 255.0)).astype(np.uint8)
    finally:
        pool.shutdown(wait=True, cancel_futures=True)
        for r in readers:
            try:
                r.close()
            except Exception:
                pass


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("manifest", help="tiles.json written by a --no-stitch run")
    ap.add_argument("--resize", type=int, nargs=2, metavar=("W", "H"), help="area-downscale to WxH")
    ap.add_argument("--chunk-size", type=int, default=4, help="frames blended at once (RAM: ~66 MB each at 2880x1920)")
    args = ap.parse_args()
    out = sys.stdout.buffer
    try:
        for frame in stitched_frames(args.manifest, args.chunk_size, args.resize):
            out.write(frame.tobytes())
        out.flush()
    except BrokenPipeError:
        sys.exit(1)


if __name__ == "__main__":
    main()
