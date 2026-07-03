"""
Star Battle → puzz.link converter
Run with: streamlit run app.py
"""

import io
import numpy as np
import streamlit as st
import streamlit.components.v1 as components
from collections import deque, Counter
from PIL import Image, ImageDraw

from streamlit_paste_button import paste_image_button

# ── Detection constants ────────────────────────────────────────────────────────
DEFAULT_GRIDLINE_THRESHOLD = 170
DEFAULT_GRIDLINE_COVERAGE  = 0.65
DEFAULT_BOLD_THRESHOLD     = 100
DEFAULT_BOLD_FRAC          = 0.50
DEFAULT_SAMPLE_INSET       = 5
SHADED_CELL_GRAY_LOW       = 180
SHADED_CELL_GRAY_HIGH      = 240
SHADED_CELL_MIN_FRAC       = 0.25


# ── Core detection functions ───────────────────────────────────────────────────

def pil_to_gray_array(pil_img: Image.Image) -> np.ndarray:
    im = pil_img.convert("RGBA")
    bg = Image.new("RGBA", im.size, (255, 255, 255, 255))
    flat = Image.alpha_composite(bg, im).convert("L")
    return np.array(flat)


def find_gridlines(a: np.ndarray, axis: int, threshold: int, coverage: float) -> list[int]:
    dim_span = a.shape[1 - axis]
    min_count = int(dim_span * coverage)
    dark = (a < threshold).sum(axis=1) if axis == 0 else (a < threshold).sum(axis=0)
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


def fill_missing_lines(positions: list[int]) -> tuple[list[int], int]:
    if len(positions) < 2:
        return positions, 0
    spacings = [positions[i+1] - positions[i] for i in range(len(positions)-1)]
    median_gap = float(np.median(spacings))
    filled, n_inserted = [positions[0]], 0
    for i, gap in enumerate(spacings):
        if gap > median_gap * 1.6:
            n_to_insert = round(gap / median_gap) - 1
            for k in range(1, n_to_insert + 1):
                filled.append(int(positions[i] + gap * k / (n_to_insert + 1)))
                n_inserted += 1
        filled.append(positions[i+1])
    return sorted(set(filled)), n_inserted


def blank_shaded_cells(a: np.ndarray, R: list, C: list, border: int = 3) -> np.ndarray:
    a = a.copy()
    for r in range(len(R) - 1):
        for c in range(len(C) - 1):
            cell = a[R[r]+2:R[r+1]-2, C[c]+2:C[c+1]-2]
            if cell.size == 0:
                continue
            gray_frac = ((cell > SHADED_CELL_GRAY_LOW) & (cell < SHADED_CELL_GRAY_HIGH)).mean()
            if gray_frac >= SHADED_CELL_MIN_FRAC:
                a[R[r]+border:R[r+1]-border, C[c]+border:C[c+1]-border] = 255
    return a


def blank_solid_cells(a: np.ndarray, R: list, C: list) -> np.ndarray:
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
    infos = []

    R = find_gridlines(a, axis=0, threshold=gridline_threshold, coverage=gridline_coverage)
    C = find_gridlines(a, axis=1, threshold=gridline_threshold, coverage=gridline_coverage)

    R, r_inserted = fill_missing_lines(R)
    C, c_inserted = fill_missing_lines(C)
    if r_inserted:
        infos.append(f"Auto-inserted {r_inserted} missing row line(s) from gap detection.")
    if c_inserted:
        infos.append(f"Auto-inserted {c_inserted} missing column line(s) from gap detection.")

    N_rows, N_cols = len(R)-1, len(C)-1

    if N_rows < 4 or N_cols < 4:
        return None, (
            f"Grid too small ({N_cols}×{N_rows}) — try lowering the "
            "Gridline coverage fraction (dashed grids need ~0.35)."
        )

    if N_rows != N_cols:
        warnings.append(
            f"Grid is {N_cols}×{N_rows} (not square). "
            "Star Battle requires NxN. Try lowering gridline coverage."
        )

    a_clean = blank_solid_cells(blank_shaded_cells(a, R, C), R, C)
    VW, HW = detect_walls(a_clean, R, C, bold_threshold, bold_frac, inset)
    region, n_regions = flood_fill(N_rows, N_cols, VW, HW)
    sizes = sorted(Counter(v for row in region for v in row).values())

    if n_regions != N_rows:
        warnings.append(
            f"Found {n_regions} regions but expected {N_rows}. "
            "Check the debug overlay — try adjusting Bold wall sensitivity."
        )

    bstr = encode_puzzlink(N_cols, N_rows, VW, HW)
    mismatches = verify_round_trip(N_cols, N_rows, bstr, VW, HW)
    if mismatches:
        warnings.append(f"Encoding round-trip had {mismatches} mismatches.")

    puzzlink_url = f"https://puzz.link/p?starbattle/{N_cols}/{N_rows}/{stars}/{bstr}"
    penpa_url    = f"https://swaroopg92.github.io/penpa-edit/?p={puzzlink_url}"
    debug_img    = make_debug_image(pil_img, R, C, VW, HW)

    return {
        "grid":         f"{N_cols}×{N_rows}",
        "n_regions":    n_regions,
        "region_sizes": sizes,
        "puzzlink_url": puzzlink_url,
        "penpa_url":    penpa_url,
        "debug_img":    debug_img,
        "warnings":     warnings,
        "infos":        infos,
        "rt_ok":        mismatches == 0,
    }, None


# ── Streamlit UI ───────────────────────────────────────────────────────────────

st.set_page_config(page_title="Star Battle → puzz.link", layout="wide")

st.title("⭐ Star Battle → puzz.link")
st.caption(
    "Upload or paste a star battle puzzle image. "
    "The app detects region boundaries and generates a puzz.link URL."
)

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("Settings")
    stars = st.radio("Stars per row / col / region", [1, 2, 3], index=1, horizontal=True)

    st.divider()
    st.subheader("Gridline detection")
    st.caption("**Dashed-grid puzzles:** lower coverage to ~0.35.")
    gridline_threshold = st.slider("Darkness threshold", 80, 220, DEFAULT_GRIDLINE_THRESHOLD,
        help="Pixel value below which a pixel is dark.")
    gridline_coverage = st.slider("Coverage fraction", 0.20, 0.95, DEFAULT_GRIDLINE_COVERAGE, 0.05,
        help="Fraction of image width/height a line must span. Solid grids: ~0.65. Dashed: ~0.35.")

    st.divider()
    st.subheader("Wall detection")
    bold_frac = st.slider("Bold wall sensitivity", 0.10, 0.90, DEFAULT_BOLD_FRAC, 0.05,
        help="Fraction of a cell-edge strip that must be dark to count as a bold wall.")
    inset = st.slider("Sample inset (px)", 1, 15, DEFAULT_SAMPLE_INSET)

    st.divider()
    with st.expander("Troubleshooting guide"):
        st.markdown("""
**"Found 1 region"** — gridlines not detected. Lower Coverage to 0.35 for dashed grids.

**Too few regions** — some bold walls missed. Lower Bold wall sensitivity.

**Too many regions** — false walls detected. Raise Bold wall sensitivity.

**Non-square grid** — a line was missed or doubled. Try a slightly different coverage value.

Use the debug overlay to see exactly what was detected.
""")

# ── Main ──────────────────────────────────────────────────────────────────────

# Image input: file upload + optional clipboard paste
col_up, col_paste = st.columns([3, 1])
with col_up:
    uploaded = st.file_uploader(
        "Upload puzzle image",
        type=["png", "jpg", "jpeg", "webp", "bmp"],
        label_visibility="collapsed",
    )
with col_paste:
    st.markdown("<br>", unsafe_allow_html=True)
    paste_result = paste_image_button("📋 Paste", key="clipboard_paste",
                                      background_color="#555",
                                      hover_background_color="#333")
    pasted = paste_result.image_data

# Paste takes priority over file upload
pil_img = pasted if pasted is not None else (
    Image.open(uploaded) if uploaded else None
)

st.caption("💡 PNG/WebP screenshots work best. Drag-and-drop also works.")

if pil_img is not None:
    col_in, col_out = st.columns(2)

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
                         caption="Red = detected region walls")
                st.download_button(
                    "⬇ Download debug image",
                    data=pil_to_bytes(result["debug_img"]),
                    file_name="debug_walls.png",
                    mime="image/png",
                    use_container_width=True,
                )

            for msg in result["infos"]:
                st.info(f"ℹ️ {msg}")
            for msg in result["warnings"]:
                st.warning(msg)

            status_parts = [
                f"**Grid:** {result['grid']}",
                f"**Regions:** {result['n_regions']}",
                f"**Sizes:** {result['region_sizes']}",
                f"**Round-trip:** {'✅ OK' if result['rt_ok'] else '❌ mismatch'}",
            ]
            st.info("  ·  ".join(status_parts))

            puzzlink = result["puzzlink_url"]
            penpa    = result["penpa_url"]

            # ── Copy puzz.link button ─────────────────────────────────────────
            escaped = puzzlink.replace("'", "\\'")
            components.html(f"""
<button onclick="navigator.clipboard.writeText('{escaped}').then(()=>{{
    this.textContent='✅ Copied!';
    setTimeout(()=>this.textContent='📋 Copy puzz.link',1500);
}})" style="width:100%;padding:9px;background:#0e7c44;color:white;border:none;
border-radius:6px;cursor:pointer;font-size:15px;font-weight:600;">
📋 Copy puzz.link
</button>""", height=48)

            # ── Path 1: Looks good ────────────────────────────────────────────
            st.subheader("✅ Borders look correct?")
            st.markdown(
                "Open in Penpa-edit, click **Share**, copy the address bar URL, "
                "paste into the marktekfan converter and click **Convert**."
            )
            col_a, col_b = st.columns(2)
            with col_a:
                st.link_button("1. Open in Penpa-edit →", penpa, use_container_width=True)
            with col_b:
                st.link_button(
                    "2. Open marktekfan converter →",
                    "https://marktekfan.github.io/sudokupad-penpa-import/",
                    use_container_width=True,
                )

            # ── Path 2: Needs fixing ──────────────────────────────────────────
            st.divider()
            with st.expander("🔧 Borders need fixing? — Penpa-edit correction steps"):
                st.markdown("""
1. Click **Open in Penpa-edit** above
2. Tap the **Problem** tab (top-left of the toolbar)
3. Tap **Edge** mode → sub-mode **Normal**
4. Tap edges to add or remove borders until the regions look right
5. Tap **Share** in the toolbar — the address bar URL updates to a Penpa URL
6. Copy that URL and paste it into the marktekfan converter, then click **Convert**
""")
                st.link_button(
                    "Open marktekfan converter →",
                    "https://marktekfan.github.io/sudokupad-penpa-import/",
                    use_container_width=True,
                )

            # ── Raw URL ───────────────────────────────────────────────────────
            with st.expander("🔗 Raw puzz.link URL"):
                st.code(puzzlink, language=None)
