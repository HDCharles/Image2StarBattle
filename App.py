"""
Star Battle → puzz.link converter
Run with: streamlit run app.py
"""

import io
import numpy as np
import streamlit as st
from collections import deque, Counter
from PIL import Image, ImageDraw

# ── Detection constants (user-adjustable via sidebar) ─────────────────────────
DEFAULT_GRIDLINE_THRESHOLD = 170
DEFAULT_GRIDLINE_COVERAGE  = 0.65
DEFAULT_BOLD_THRESHOLD     = 100
DEFAULT_BOLD_FRAC          = 0.50
DEFAULT_SAMPLE_INSET       = 5
VOID_CELL_MEAN_LOW         = 220   # cells with mean brightness in this range are
VOID_CELL_MEAN_HIGH        = 252   # treated as shaded/void and blanked before detection


# ── Core detection functions ───────────────────────────────────────────────────

def pil_to_gray_array(pil_img: Image.Image) -> np.ndarray:
    im = pil_img.convert("RGBA")
    bg = Image.new("RGBA", im.size, (255, 255, 255, 255))
    flat = Image.alpha_composite(bg, im).convert("L")
    return np.array(flat)


def find_gridlines(a: np.ndarray, axis: int, threshold: int, coverage: float) -> list[int]:
    dim_span = a.shape[1 - axis]
    min_count = int(dim_span * coverage)
    dark = (a < threshold).sum(axis=axis if axis == 1 else 1) if axis == 1 \
        else (a < threshold).sum(axis=1)
    positions, cluster = [], []
    for i, v in enumerate(dark):
        if v >= min_count:
            cluster.append(i)
        else:
            if cluster:
                positions.append(int(np.mean(cluster)))
                cluster = []
    if cluster:
        positions.append(int(np.mean(cluster)))
    return positions


def blank_shaded_cells(a: np.ndarray, R: list, C: list) -> np.ndarray:
    """Overwrite gray/shaded cells with white so they don't affect wall detection."""
    a = a.copy()
    for r in range(len(R) - 1):
        for c in range(len(C) - 1):
            cell = a[R[r]+2:R[r+1]-2, C[c]+2:C[c+1]-2]
            m = cell.mean()
            if VOID_CELL_MEAN_LOW < m < VOID_CELL_MEAN_HIGH:
                a[R[r]:R[r+1], C[c]:C[c+1]] = 255
    return a


def blank_solid_cells(a: np.ndarray, R: list, C: list) -> np.ndarray:
    """Overwrite solid black cells (void cells) with white."""
    a = a.copy()
    for r in range(len(R) - 1):
        for c in range(len(C) - 1):
            cell = a[R[r]+3:R[r+1]-3, C[c]+3:C[c+1]-3]
            if cell.size > 0 and (cell < 80).mean() > 0.35:
                a[R[r]:R[r+1], C[c]:C[c+1]] = 255
    return a


def detect_walls(a: np.ndarray, R: list, C: list,
                 bold_thresh: int, bold_frac: float, inset: int):
    N_rows, N_cols = len(R) - 1, len(C) - 1
    VW, HW = set(), set()
    for c in range(1, N_cols):
        x = C[c]
        for r in range(N_rows):
            strip = a[R[r]+inset:R[r+1]-inset, x-1:x+2]
            if strip.size > 0 and (strip < bold_thresh).mean() > bold_frac:
                VW.add((r, c))
    for r in range(1, N_rows):
        y = R[r]
        for c in range(N_cols):
            strip = a[y-1:y+2, C[c]+inset:C[c+1]-inset]
            if strip.size > 0 and (strip < bold_thresh).mean() > bold_frac:
                HW.add((r, c))
    return VW, HW


def flood_fill(N_rows: int, N_cols: int, VW: set, HW: set):
    def walled(r, c, r2, c2):
        return (r, max(c, c2)) in VW if r == r2 else (max(r, r2), c) in HW
    region = [[-1]*N_cols for _ in range(N_rows)]
    cur = 0
    for r in range(N_rows):
        for c in range(N_cols):
            if region[r][c] == -1:
                q = deque([(r, c)])
                region[r][c] = cur
                while q:
                    cr, cc = q.popleft()
                    for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
                        nr, nc = cr+dr, cc+dc
                        if 0<=nr<N_rows and 0<=nc<N_cols and region[nr][nc]==-1:
                            if not walled(cr, cc, nr, nc):
                                region[nr][nc] = cur
                                q.append((nr, nc))
                cur += 1
    return region, cur


def encode_puzzlink(N_cols: int, N_rows: int, VW: set, HW: set) -> str:
    digits  = "0123456789abcdefghijklmnopqrstuv"
    weights = [16, 8, 4, 2, 1]
    vbits = [0] * ((N_cols-1)*N_rows)
    for r, c in VW:
        vbits[r*(N_cols-1)+(c-1)] = 1
    hbits = [0] * (N_cols*(N_rows-1))
    for r, c in HW:
        hbits[(r-1)*N_cols+c] = 1
    def pack(bits):
        s = ""
        for i in range(0, len(bits), 5):
            g = bits[i:i+5]
            while len(g) < 5: g.append(0)
            s += digits[sum(b*w for b,w in zip(g, weights))]
        return s
    return pack(vbits) + pack(hbits)


def verify_round_trip(N_cols: int, N_rows: int, bstr: str, VW: set, HW: set) -> int:
    digits = "0123456789abcdefghijklmnopqrstuv"
    twi = [16, 8, 4, 2, 1]
    pos1 = min(((N_cols-1)*N_rows+4)//5, len(bstr))
    pos2 = min(((N_cols*(N_rows-1)+4)//5)+pos1, len(bstr))
    border, idx = {}, 0
    for i in range(pos1):
        ca = digits.index(bstr[i])
        for w in range(5):
            if idx < (N_cols-1)*N_rows:
                border[idx] = 1 if (ca & twi[w]) else 0
                idx += 1
    idx = (N_cols-1)*N_rows
    for i in range(pos1, pos2):
        ca = digits.index(bstr[i])
        for w in range(5):
            if idx < 2*N_cols*N_rows - N_cols - N_rows:
                border[idx] = 1 if (ca & twi[w]) else 0
                idx += 1
    VW2, HW2 = {}, {}
    for i, val in border.items():
        if i < (N_cols-1)*N_rows:
            r=i//(N_cols-1); c=i%(N_cols-1)
            VW2[(r,c+1)]=val
        else:
            ii=i-(N_cols-1)*N_rows; r=ii//N_cols; c=ii%N_cols
            HW2[(r+1,c)]=val
    mismatches = 0
    for r in range(N_rows):
        for c in range(1, N_cols):
            if VW2.get((r,c),0) != (1 if (r,c) in VW else 0): mismatches += 1
    for r in range(1, N_rows):
        for c in range(N_cols):
            if HW2.get((r,c),0) != (1 if (r,c) in HW else 0): mismatches += 1
    return mismatches


def make_debug_image(pil_img: Image.Image, R: list, C: list,
                     VW: set, HW: set) -> Image.Image:
    out = pil_img.convert("RGB").copy()
    draw = ImageDraw.Draw(out)
    for r, c in VW:
        draw.line([(C[c], R[r]), (C[c], R[r+1])], fill=(220, 30, 30), width=3)
    for r, c in HW:
        draw.line([(C[c], R[r]), (C[c+1], R[r])], fill=(220, 30, 30), width=3)
    return out


def pil_to_bytes(img: Image.Image, fmt="PNG") -> bytes:
    buf = io.BytesIO()
    img.save(buf, format=fmt)
    return buf.getvalue()


def run_detection(pil_img: Image.Image, stars: int,
                  gridline_threshold: int, gridline_coverage: float,
                  bold_threshold: int, bold_frac: float, inset: int):
    a = pil_to_gray_array(pil_img)
    warnings = []

    # Find gridlines on raw array
    R = find_gridlines(a, axis=0, threshold=gridline_threshold, coverage=gridline_coverage)
    C = find_gridlines(a, axis=1, threshold=gridline_threshold, coverage=gridline_coverage)
    N_rows, N_cols = len(R)-1, len(C)-1

    if N_rows < 4 or N_cols < 4:
        return None, "Grid too small — check image or lower the gridline threshold."

    if N_rows != N_cols:
        warnings.append(f"Grid is {N_cols}×{N_rows} (not square). Expected NxN for Star Battle.")

    # Preprocess: blank shaded + solid cells
    a_clean = blank_solid_cells(blank_shaded_cells(a, R, C), R, C)

    VW, HW = detect_walls(a_clean, R, C, bold_threshold, bold_frac, inset)
    region, n_regions = flood_fill(N_rows, N_cols, VW, HW)
    sizes = sorted(Counter(v for row in region for v in row).values())

    if n_regions != N_rows:
        warnings.append(
            f"Found {n_regions} regions but expected {N_rows}. "
            "Try adjusting bold threshold — some walls may be missed or doubled."
        )

    bstr = encode_puzzlink(N_cols, N_rows, VW, HW)
    mismatches = verify_round_trip(N_cols, N_rows, bstr, VW, HW)
    if mismatches:
        warnings.append(f"Encoding round-trip had {mismatches} mismatches.")

    puzzlink_url  = f"https://puzz.link/p?starbattle/{N_cols}/{N_rows}/{stars}/{bstr}"
    penpa_url     = puzzlink_url   # puzz.link loads directly into Penpa-edit
    debug_img     = make_debug_image(pil_img, R, C, VW, HW)

    return {
        "grid":         f"{N_cols}×{N_rows}",
        "n_regions":    n_regions,
        "region_sizes": sizes,
        "puzzlink_url": puzzlink_url,
        "penpa_url":    penpa_url,
        "debug_img":    debug_img,
        "warnings":     warnings,
        "rt_ok":        mismatches == 0,
    }, None


# ── Streamlit UI ───────────────────────────────────────────────────────────────

st.set_page_config(page_title="Star Battle → puzz.link", layout="wide")

st.title("⭐ Star Battle → puzz.link")
st.caption(
    "Upload a star battle puzzle image. The app detects region boundaries and generates "
    "a puzz.link URL you can open in Penpa-edit, then convert to SudokuPad."
)

# ── Sidebar: settings ─────────────────────────────────────────────────────────
with st.sidebar:
    st.header("Settings")
    stars = st.radio("Stars per row / col / region", [1, 2, 3], index=1, horizontal=True)
    st.divider()
    st.subheader("Advanced")
    gridline_threshold = st.slider(
        "Gridline darkness threshold", 80, 220, DEFAULT_GRIDLINE_THRESHOLD,
        help="Lower for faint/dashed gridlines"
    )
    gridline_coverage = st.slider(
        "Gridline coverage fraction", 0.40, 0.95, DEFAULT_GRIDLINE_COVERAGE, 0.05,
        help="Fraction of image width/height a line must span"
    )
    bold_frac = st.slider(
        "Bold wall sensitivity", 0.10, 0.90, DEFAULT_BOLD_FRAC, 0.05,
        help="Lower = more walls detected; raise if getting too many false walls"
    )
    inset = st.slider("Sample inset (px)", 1, 15, DEFAULT_SAMPLE_INSET)

# ── Main area ────────────────────────────────────────────────────────────────
uploaded = st.file_uploader(
    "Upload puzzle image",
    type=["png", "jpg", "jpeg", "webp", "bmp"],
    label_visibility="collapsed",
)

st.caption("💡 Tip: you can also paste an image from clipboard using your OS — "
           "save it first then upload, or drag-and-drop from your file manager.")

if uploaded:
    pil_img = Image.open(uploaded)

    col_in, col_out = st.columns([1, 1])

    with col_in:
        st.subheader("Input")
        st.image(pil_img, use_container_width=True)
        run = st.button("🔍 Detect regions", type="primary", use_container_width=True)

    if run:
        with st.spinner("Detecting grid and region walls…"):
            result, err = run_detection(
                pil_img, stars,
                gridline_threshold, gridline_coverage,
                DEFAULT_BOLD_THRESHOLD, bold_frac, inset,
            )

        if err:
            st.error(f"Detection failed: {err}")
        else:
            with col_out:
                st.subheader("Debug overlay")
                st.image(result["debug_img"], use_container_width=True,
                         caption="Red lines = detected region walls")
                st.download_button(
                    "⬇ Download debug image",
                    data=pil_to_bytes(result["debug_img"]),
                    file_name="debug_walls.png",
                    mime="image/png",
                    use_container_width=True,
                )

            # Warnings
            for w in result["warnings"]:
                st.warning(w)

            # Status
            status_parts = [
                f"**Grid:** {result['grid']}",
                f"**Regions:** {result['n_regions']}",
                f"**Sizes:** {result['region_sizes']}",
                f"**Round-trip:** {'✅ OK' if result['rt_ok'] else '❌ mismatch'}",
            ]
            st.info("  ·  ".join(status_parts))

            # URLs
            st.subheader("Results")

            puzzlink = result["puzzlink_url"]
            st.text_input("puzz.link URL", value=puzzlink, key="puzzlink")
            st.markdown(f"[🔗 Open in Penpa-edit]({puzzlink}){{target='_blank'}}")

            st.divider()
            st.subheader("→ SudokuPad workflow")
            st.markdown("""
1. Click **Open in Penpa-edit** above — the puzzle loads with Star Battle mode
2. Fix any misdetected walls using Penpa-edit's editor
3. Copy the URL from your browser's address bar (it updates as you edit)
4. Paste that URL into [marktekfan's SudokuPad converter ↗](https://marktekfan.github.io/sudokupad-penpa-import/)
5. Click Convert → open the resulting link in SudokuPad
""")
            st.link_button(
                "Open marktekfan converter",
                "https://marktekfan.github.io/sudokupad-penpa-import/",
                use_container_width=True,
            )