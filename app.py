import base64
import html
import io
import time
from datetime import datetime
from pathlib import Path

import matplotlib.cm as cm
import numpy as np
import pandas as pd
import scipy.ndimage as ndi
import streamlit as st
import torch
from PIL import Image, ImageDraw
import segmentation_models_pytorch as smp

# These values are deliberately kept separate from the model preprocessing.
IMAGE_WIDTH = 512
IMAGE_HEIGHT = 288
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
DEFAULT_THRESHOLD = 0.5
SEVERITY_LIMITS = {"negligible": 0.5, "small": 3.0, "medium": 10.0}
PRESETS = {
    "Sensitive": {"threshold": 0.15, "minimum_area": 12, "minimum_confidence": 30.0},
    "Balanced": {"threshold": 0.40, "minimum_area": 40, "minimum_confidence": 50.0},
    "Strict": {"threshold": 0.70, "minimum_area": 80, "minimum_confidence": 70.0},
}
PRESET_DESCRIPTIONS = {
    "Sensitive": "More likely to notice small objects, but may show more false alarms.",
    "Balanced": "A practical middle ground for everyday checks.",
    "Strict": "Fewer false alarms, but may miss small objects.",
}
DEFAULT_PRESET = "Balanced"
MIN_REGION_AREA = PRESETS[DEFAULT_PRESET]["minimum_area"]
MIN_REGION_CONFIDENCE = PRESETS[DEFAULT_PRESET]["minimum_confidence"]
MAX_IMAGE_DIMENSION = 4096
WEIGHTS_PATH = Path(__file__).parent / "deeplabv3p_resnet34.pt"

# Shared visual palette for the safety-dashboard interface.
SUCCESS = "#16A34A"
WARNING = "#F59E0B"
DANGER = "#DC2626"
INFO = "#0EA5E9"
NAVY = "#0F172A"
SLATE = "#475569"
MUTED = "#64748B"
SURFACE = "#FFFFFF"
BACKGROUND = "#F8FAFC"
LINE = "#E2E8F0"

# Risk is a transparent, intentionally simple prioritisation aid, not a safety probability.
RISK_WEIGHTS = {
    "size": 0.25,
    "path": 0.35,
    "vertical_position": 0.15,
    "confidence": 0.25,
}
RISK_SIZE_REFERENCE_PERCENT = 10.0
PATH_NEAR_MARGIN = 0.12
PATH_POLYGON = ((0.38, 0.38), (0.62, 0.38), (0.86, 1.0), (0.14, 1.0))

st.set_page_config(
    page_title="Road Anomaly Detection",
    page_icon="🛣️",
    layout="wide",
)


def inject_css() -> None:
    """Apply the shared visual language without changing Streamlit behavior."""
    st.markdown(
        f"""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
        :root {{
            --navy: {NAVY};
            --slate: {SLATE};
            --muted: {MUTED};
            --surface: {SURFACE};
            --background: {BACKGROUND};
            --line: {LINE};
            --info: {INFO};
        }}
        html, body, [class*="css"] {{
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
        }}
        .stApp {{ background: var(--background); color: var(--navy); }}
        [data-testid="stAppViewContainer"] .main .block-container {{
            max-width: 1200px;
            padding-top: 2rem;
            padding-bottom: 3rem;
        }}
        [data-testid="stHeader"] {{ background: transparent; }}
        [data-testid="stCaptionContainer"], .stCaption {{
            color: var(--slate) !important;
            font-size: .9rem !important;
        }}
        .hero {{
            background: linear-gradient(120deg, #0F172A 0%, #164E63 100%);
            color: #F8FAFC; border-radius: 18px; padding: 2rem 2.2rem;
            margin-bottom: 1.3rem; box-shadow: 0 14px 30px rgba(15,23,42,.14);
        }}
        .hero h1 {{ color: white; font-size: 2.35rem; margin: .25rem 0 .4rem; letter-spacing: -.04em; }}
        .hero p {{ color: #DCEAF3; margin: 0; font-size: 1rem; }}
        .prototype-badge {{
            display: inline-block; border: 1px solid rgba(255,255,255,.35);
            border-radius: 999px; padding: .25rem .65rem; font-size: .75rem;
            font-weight: 700; letter-spacing: .04em; text-transform: uppercase;
        }}
        .steps {{ display: flex; gap: .85rem; margin: .7rem 0 1.7rem; }}
        .step-card {{
            background: var(--surface); border: 1px solid var(--line); border-radius: 14px;
            padding: 1rem 1.1rem; flex: 1; box-shadow: 0 4px 14px rgba(15,23,42,.05);
        }}
        .step-card strong {{ display: block; color: var(--navy); margin-bottom: .25rem; }}
        .step-card span {{ color: var(--slate); font-size: .9rem; line-height: 1.4; }}
        .step-icon {{ font-size: 1.35rem; margin-right: .4rem; }}
        .result-card, [data-testid="stMetric"], [data-testid="stDataFrame"] {{
            background: var(--surface); border: 1px solid var(--line);
            border-radius: 14px; box-shadow: 0 5px 16px rgba(15,23,42,.06);
        }}
        .result-card {{ padding: 1.2rem 1.35rem; margin: .6rem 0 1rem; }}
        .result-card h2 {{ margin: 0; color: var(--navy); }}
        [data-testid="stMetric"] {{ padding: 1rem; }}
        [data-testid="stMetricLabel"] {{ color: var(--slate) !important; font-size: .85rem; }}
        [data-testid="stMetricValue"] {{ color: var(--navy) !important; font-weight: 800; }}
        .stButton > button, .stDownloadButton > button {{
            border-radius: 10px; min-height: 2.6rem; font-weight: 600;
            border: 1px solid #CBD5E1; transition: all .15s ease;
        }}
        .stButton > button:hover, .stDownloadButton > button:hover {{
            border-color: var(--info); color: var(--info); transform: translateY(-1px);
        }}
        [data-baseweb="tab-list"] {{ gap: .4rem; }}
        [data-baseweb="tab"] {{ border-radius: 999px; padding: .45rem 1rem; }}
        [aria-selected="true"][data-baseweb="tab"] {{
            background: #E0F2FE; color: #075985; font-weight: 700;
        }}
        [data-testid="stDataFrame"] {{ padding: .35rem; overflow: hidden; }}
        [data-testid="stSidebar"] {{ border-right: 1px solid var(--line); }}
        [data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3 {{ color: var(--navy); }}
        .sidebar-section {{
            color: var(--muted); font-size: .72rem; font-weight: 800;
            letter-spacing: .1em; text-transform: uppercase; margin: 1.2rem 0 .45rem;
        }}
        div[data-testid="stAlert"] {{ border-radius: 12px; }}
        footer {{ visibility: hidden; }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_header() -> None:
    """Render the calm, compact onboarding header."""
    st.markdown(
        """
        <section class="hero">
            <span class="prototype-badge">Experimental prototype</span>
            <h1>🛣️ Road Anomaly Detection</h1>
            <p>A clear visual check for unusual regions on road images.</p>
        </section>
        <div class="steps">
            <div class="step-card"><strong><span class="step-icon">📤</span>1. Upload</strong>
                <span>Choose one or more road images.</span></div>
            <div class="step-card"><strong><span class="step-icon">🔎</span>2. Analyze</strong>
                <span>We highlight areas worth a closer look.</span></div>
            <div class="step-card"><strong><span class="step-icon">✅</span>3. Decide</strong>
                <span>Review the result and choose a safe next step.</span></div>
        </div>
        """,
        unsafe_allow_html=True,
    )


@st.cache_resource
def load_model(device_name: str):
    """Load the original architecture and checkpoint once per device."""
    if not WEIGHTS_PATH.exists():
        raise FileNotFoundError(
            "The model file deeplabv3p_resnet34.pt is missing beside app.py."
        )
    model = smp.DeepLabV3Plus(
        encoder_name="resnet34",
        encoder_weights=None,
        in_channels=3,
        classes=1,
    )
    try:
        state = torch.load(WEIGHTS_PATH, map_location="cpu", weights_only=True)
    except TypeError:
        state = torch.load(WEIGHTS_PATH, map_location="cpu")
    model.load_state_dict(state, strict=True)
    return model.to(device_name).eval()


def preprocess(image: Image.Image) -> torch.Tensor:
    """Keep the training preprocessing unchanged: 512x288, ImageNet normalized."""
    image = image.convert("RGB").resize(
        (IMAGE_WIDTH, IMAGE_HEIGHT), Image.Resampling.BILINEAR
    )
    arr = np.asarray(image, dtype=np.float32) / 255.0
    arr = (arr - MEAN) / STD
    return torch.from_numpy(arr.transpose(2, 0, 1)).unsqueeze(0)


@st.cache_data(show_spinner=False)
def predict_probabilities(image_bytes: bytes, device_name: str) -> np.ndarray:
    """Cache only the model output so UI slider changes do not run inference again."""
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    model = load_model(device_name)
    with torch.inference_mode():
        probabilities = torch.sigmoid(
            model(preprocess(image).to(device_name))
        )[0, 0].cpu().numpy()
    return probabilities.astype(np.float32)


def safe_image(image_bytes: bytes) -> Image.Image:
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    if max(image.size) > MAX_IMAGE_DIMENSION:
        scale = MAX_IMAGE_DIMENSION / max(image.size)
        image = image.resize(
            (max(1, int(image.width * scale)), max(1, int(image.height * scale))),
            Image.Resampling.LANCZOS,
        )
    return image


def make_overlay(rgb: np.ndarray, mask: np.ndarray, color: tuple, opacity: float) -> np.ndarray:
    overlay = rgb.astype(np.float32).copy()
    color_array = np.array(color, dtype=np.float32)
    overlay[mask] = (1 - opacity) * overlay[mask] + opacity * color_array
    return np.clip(overlay, 0, 255).astype(np.uint8)


def find_regions(mask: np.ndarray, probabilities: np.ndarray, minimum_area: int) -> list[dict]:
    labels, count = ndi.label(mask, structure=np.ones((3, 3), dtype=np.uint8))
    regions = []
    for label_number in range(1, count + 1):
        ys, xs = np.where(labels == label_number)
        area = len(xs)
        regions.append(
            {
                "label": len(regions) + 1,
                "area": area,
                "percent": area / mask.size * 100,
                "confidence": float(probabilities[labels == label_number].mean() * 100),
                "box": (int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())),
            }
        )
    return regions


def path_polygon(shape: tuple[int, int]) -> list[tuple[int, int]]:
    height, width = shape
    return [(round(x * width), round(y * height)) for x, y in PATH_POLYGON]


def path_zone_for(region: dict, shape: tuple[int, int]) -> str:
    """Classify a region using its centre and the lower-centre driving path."""
    height, width = shape
    x1, y1, x2, y2 = region["box"]
    centre_x = ((x1 + x2) / 2) / width
    centre_y = ((y1 + y2) / 2) / height
    left_edge = PATH_POLYGON[0][0] + (PATH_POLYGON[3][0] - PATH_POLYGON[0][0]) * centre_y
    right_edge = PATH_POLYGON[1][0] + (PATH_POLYGON[2][0] - PATH_POLYGON[1][0]) * centre_y
    if left_edge <= centre_x <= right_edge and centre_y >= PATH_POLYGON[0][1]:
        return "In your path"
    expanded_left = left_edge - PATH_NEAR_MARGIN
    expanded_right = right_edge + PATH_NEAR_MARGIN
    if expanded_left <= centre_x <= expanded_right and centre_y >= 0.25:
        return "Near your path"
    return "Off the road side"


def risk_score_for(region: dict, shape: tuple[int, int]) -> tuple[float, str]:
    """Return a review-priority score from measurable region properties."""
    height, width = shape
    _, y1, _, y2 = region["box"]
    vertical = np.clip(((y1 + y2) / 2) / height, 0.0, 1.0)
    size_score = min(region["percent"] / RISK_SIZE_REFERENCE_PERCENT, 1.0)
    path_score = {
        "In your path": 1.0,
        "Near your path": 0.55,
        "Off the road side": 0.2,
    }[region["path_zone"]]
    confidence_score = np.clip(region["confidence"] / 100.0, 0.0, 1.0)
    score = 100 * (
        RISK_WEIGHTS["size"] * size_score
        + RISK_WEIGHTS["path"] * path_score
        + RISK_WEIGHTS["vertical_position"] * vertical
        + RISK_WEIGHTS["confidence"] * confidence_score
    )
    label = "Low" if score < 35 else "Medium" if score < 65 else "High" if score < 85 else "Critical"
    return round(float(score), 1), label


def enrich_regions(regions: list[dict], shape: tuple[int, int]) -> list[dict]:
    for index, region in enumerate(regions, start=1):
        region["label"] = index
        region["position"] = position_for(region, shape)
        region["path_zone"] = path_zone_for(region, shape)
        region["risk_score"], region["risk_label"] = risk_score_for(region, shape)
    return sorted(regions, key=lambda item: item["risk_score"], reverse=True)


def position_for(region: dict, shape: tuple[int, int]) -> str:
    x1, y1, x2, y2 = region["box"]
    height, width = shape
    x = ((x1 + x2) / 2) / width
    y = ((y1 + y2) / 2) / height
    horizontal = "left" if x < 1 / 3 else "right" if x > 2 / 3 else "center"
    vertical = "top" if y < 1 / 3 else "bottom" if y > 2 / 3 else "middle"
    return f"{horizontal}, {vertical}"


def annotated_image(image: Image.Image, probabilities: np.ndarray, threshold: float,
                    minimum_area: int, minimum_confidence: float, color: tuple,
                    opacity: float, show_hidden: bool, show_path: bool
                    ) -> tuple[np.ndarray, np.ndarray, list[dict], int]:
    rgb = np.asarray(image, dtype=np.uint8)
    mask_small = probabilities > threshold
    mask = np.asarray(
        Image.fromarray(mask_small).resize(image.size, Image.Resampling.NEAREST),
        dtype=bool,
    )
    overlay = make_overlay(rgb, mask, color, opacity)
    all_regions = enrich_regions(
        find_regions(mask_small, probabilities, minimum_area), probabilities.shape
    )
    visible_regions = [
        region for region in all_regions
        if show_hidden or (
            region["area"] >= minimum_area
            and region["confidence"] >= minimum_confidence
        )
    ]
    hidden_count = len(all_regions) - len(visible_regions)
    scale_x, scale_y = image.width / IMAGE_WIDTH, image.height / IMAGE_HEIGHT
    overlay_image = Image.fromarray(overlay)
    if show_path:
        path_layer = Image.new("RGBA", overlay_image.size, (0, 0, 0, 0))
        path_draw = ImageDraw.Draw(path_layer)
        path_draw.polygon(
            [(round(x * scale_x), round(y * scale_y)) for x, y in path_polygon(probabilities.shape)],
            fill=(60, 130, 220, 35),
            outline=(60, 130, 220, 130),
        )
        overlay_image = Image.alpha_composite(overlay_image.convert("RGBA"), path_layer).convert("RGB")
    draw = ImageDraw.Draw(overlay_image)
    risk_colors = {"Low": (245, 190, 40), "Medium": (234, 120, 25), "High": (210, 45, 35), "Critical": (145, 15, 25)}
    for region in visible_regions:
        x1, y1, x2, y2 = region["box"]
        box = (int(x1 * scale_x), int(y1 * scale_y), int((x2 + 1) * scale_x), int((y2 + 1) * scale_y))
        region_color = risk_colors[region["risk_label"]]
        draw.rectangle(box, outline=region_color, width=max(2, round(min(image.size) / 250)))
        text = f"Anomaly {region['label']} · {region['risk_label']}"
        text_box = draw.textbbox((0, 0), text)
        text_width = text_box[2] - text_box[0]
        text_height = text_box[3] - text_box[1]
        label_box = (box[0], max(0, box[1] - text_height - 8), box[0] + text_width + 12, box[1])
        draw.rounded_rectangle(label_box, radius=4, fill=(25, 30, 40))
        draw.text((label_box[0] + 6, label_box[1] + 3), text, fill=(255, 255, 255))
    return np.asarray(overlay_image), mask, visible_regions, hidden_count


def verdict_for(percent: float) -> tuple[str, str, str]:
    if percent < SEVERITY_LIMITS["negligible"]:
        return "No anomaly detected", "Negligible", "#15803d"
    if percent < SEVERITY_LIMITS["medium"]:
        return "Possible anomaly detected", "Small" if percent < SEVERITY_LIMITS["small"] else "Medium", "#c2410c"
    return "Large anomaly detected", "Large", "#b91c1c"


def heatmap(probabilities: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    rgba = cm.get_cmap("inferno")(probabilities)
    image = Image.fromarray(np.uint8(rgba[:, :, :3] * 255)).resize(size, Image.Resampling.BILINEAR)
    return np.asarray(image)


def png_bytes(array: np.ndarray) -> bytes:
    output = io.BytesIO()
    Image.fromarray(array).save(output, format="PNG")
    return output.getvalue()


def recommended_action(risk_score: float, regions: list[dict]) -> str:
    if not regions or risk_score < 35:
        return "Continue normally, while keeping your usual attention on the road."
    if risk_score < 65:
        return "Slow down and stay alert. Check the highlighted area when it is safe."
    return "Slow down and avoid the highlighted area. Change lane or stop only if it is safe."


def apply_preset() -> None:
    values = PRESETS[st.session_state["preset"]]
    st.session_state["threshold"] = values["threshold"]
    st.session_state["minimum_area"] = values["minimum_area"]
    st.session_state["minimum_confidence"] = values["minimum_confidence"]


def html_report(image: Image.Image, overlay: np.ndarray, verdict: str, severity: str,
                regions: list[dict], settings: dict) -> bytes:
    original = base64.b64encode(png_bytes(np.asarray(image))).decode()
    marked = base64.b64encode(png_bytes(overlay)).decode()
    rows = "".join(
        f"<tr><td>Anomaly {r['label']}</td><td>{r['percent']:.2f}%</td>"
        f"<td>{r['confidence']:.1f}%</td><td>{r['position']}</td></tr>"
        for r in regions
    )
    document = f"""<!doctype html><html><head><meta charset="utf-8"><title>Road anomaly report</title>
    <style>body{{font-family:Arial;max-width:1000px;margin:2rem auto}} img{{max-width:48%;margin-right:1%}}
    table{{border-collapse:collapse;width:100%}}td,th{{border:1px solid #ccc;padding:8px;text-align:left}}</style></head>
    <body><h1>Road anomaly report</h1><p>{datetime.now().strftime('%Y-%m-%d %H:%M')}</p>
    <h2>{html.escape(verdict)} — {html.escape(severity)}</h2><p>{len(regions)} anomaly region(s) found.</p>
    <img src="data:image/png;base64,{original}" alt="Original"><img src="data:image/png;base64,{marked}" alt="Overlay">
    <h2>Detected regions</h2><table><tr><th>Region</th><th>Size</th><th>Confidence</th><th>Position</th></tr>{rows}</table>
    <h2>Settings</h2><pre>{html.escape(str(settings))}</pre></body></html>"""
    return document.encode("utf-8")


def render_result(name: str, image: Image.Image, probabilities: np.ndarray, settings: dict) -> dict:
    started = time.perf_counter()
    hidden_state_key = f"show-hidden-{name}"
    show_hidden = settings["show_hidden"] or st.session_state.get(hidden_state_key, False)
    overlay, mask, regions, hidden_count = annotated_image(
        image, probabilities, settings["threshold"], settings["minimum_area"],
        settings["minimum_confidence"], settings["color"], settings["opacity"],
        show_hidden, settings["show_path"],
    )
    percent = float((probabilities > settings["threshold"]).mean() * 100)
    verdict, severity, color = verdict_for(percent)
    for region in regions:
        region["position"] = position_for(region, probabilities.shape)
    elapsed = time.perf_counter() - started

    st.markdown(
        f'<div class="result-card" style="border-left:7px solid {color};">'
        f"<h2>{verdict}</h2><b>Severity: {severity}</b><br>"
        f"{'The image looks clear at this setting.' if not regions else f'We found {len(regions)} region(s) that may need a closer look.'}"
        "</div>",
        unsafe_allow_html=True,
    )
    m1, m2, m3 = st.columns(3)
    m1.metric("Flagged image area", f"{percent:.2f}%")
    m2.metric("Anomalies found", len(regions))
    m3.metric("Inference time", f"{elapsed:.2f}s")
    risk_score = max((region["risk_score"] for region in regions), default=0.0)
    risk_label = "Low" if risk_score < 35 else "Medium" if risk_score < 65 else "High" if risk_score < 85 else "Critical"
    st.progress(risk_score / 100, text=f"Risk priority: {risk_score:.0f}/100 ({risk_label})")
    st.markdown(f"**Recommended action:** {recommended_action(risk_score, regions)}")
    st.caption(
        "Risk combines region size, whether it is in the driving path, vertical position, "
        "and model confidence. It is a review aid, not a probability of danger."
    )
    if hidden_count:
        if show_hidden:
            st.info(f"Showing {hidden_count} tiny or low-confidence detection(s) that were normally hidden.")
        else:
            hidden_label = "detection" if hidden_count == 1 else "detections"
            if st.button(
                f"{hidden_count} doubtful {hidden_label} hidden. Click to show.",
                key=f"show-hidden-button-{name}",
            ):
                st.session_state[hidden_state_key] = True
                st.rerun()

    tabs = st.tabs(["Result", "Mask", "Heatmap", "Side by side"])
    with tabs[0]:
        st.image(overlay, caption="Detected regions (outlined and numbered)", use_container_width=True)
        if regions:
            table = pd.DataFrame([
                {
                    "Anomaly": f"Anomaly {r['label']}",
                    "Risk": f"{r['risk_score']:.0f}/100 ({r['risk_label']})",
                    "Size (% image)": round(r["percent"], 2),
                    "Confidence": f"{r['confidence']:.1f}%",
                    "Position": r["position"],
                    "Driving path": r["path_zone"],
                }
                for r in regions
            ])
            st.dataframe(table, hide_index=True, use_container_width=True)
        else:
            st.info("No region was large enough to display. Try a lower threshold or minimum size.")
    with tabs[1]:
        st.image(mask.astype(np.uint8) * 255, caption="Binary mask", use_container_width=True)
    with tabs[2]:
        st.image(heatmap(probabilities, image.size), caption="Raw model probabilities (darker = lower, brighter = higher)",
                 use_container_width=True)
    with tabs[3]:
        left, right = st.columns(2)
        left.image(image, caption="Original", use_container_width=True)
        right.image(overlay, caption="Overlay", use_container_width=True)

    base_name = Path(name).stem or "anomaly"
    report_regions = [dict(r) for r in regions]
    report = html_report(image, overlay, verdict, severity, report_regions, settings)
    download_columns = st.columns(3)
    with download_columns[0]:
        st.download_button("Download mask", png_bytes(mask.astype(np.uint8) * 255),
                           f"{base_name}_mask.png", "image/png", key=f"mask-{name}",
                           use_container_width=True)
    with download_columns[1]:
        st.download_button("Download overlay", png_bytes(overlay),
                           f"{base_name}_overlay.png", "image/png", key=f"overlay-{name}",
                           use_container_width=True)
    with download_columns[2]:
        st.download_button("Download report", report, f"{base_name}_report.html", "text/html",
                           key=f"report-{name}", use_container_width=True)
    return {
        "File name": name,
        "Verdict": verdict,
        "Flagged (%)": round(percent, 2),
        "Anomalies": len(regions),
        "Risk score": risk_score,
        "Risk level": risk_label,
        "Inference (s)": round(elapsed, 2),
    }


def main() -> None:
    if "history" not in st.session_state:
        st.session_state.history = []
    if "preset" not in st.session_state:
        st.session_state.preset = DEFAULT_PRESET
        apply_preset()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    inject_css()
    render_header()
    with st.sidebar:
        st.header("Settings")
        st.markdown("**How closely should we check?**")
        preset = st.radio(
            "Choose a checking style",
            list(PRESETS),
            key="preset",
            on_change=apply_preset,
            label_visibility="collapsed",
        )
        st.caption(PRESET_DESCRIPTIONS[preset])
        st.markdown(
            "**Sensitive:** more small spots, more false alarms  \n"
            "**Balanced:** practical everyday setting  \n"
            "**Strict:** fewer false alarms, may miss small objects"
        )
        expert_mode = st.checkbox(
            "Expert mode",
            value=False,
            help="Show the underlying controls when you need precise tuning.",
        )
        if expert_mode:
            threshold = st.slider(
                "How sure must the model be to color a pixel",
                0.1,
                0.9,
                key="threshold",
                step=0.05,
                help="Example: 0.70 colors only pixels the model considers very likely to be anomalous.",
            )
            minimum_area = st.slider(
                "Ignore very small spots",
                1,
                500,
                key="minimum_area",
                step=1,
                help="Example: 40 ignores connected spots smaller than 40 model pixels.",
            )
            minimum_confidence = st.slider(
                "Hide doubtful detections",
                0.0,
                100.0,
                key="minimum_confidence",
                step=5.0,
                help="Example: 60 hides regions whose average model confidence is below 60%.",
            )
            opacity = st.slider(
                "Overlay opacity",
                0.1,
                1.0,
                0.55,
                0.05,
                help="Example: 0.55 makes the highlight visible while keeping the road easy to see.",
            )
            color = st.color_picker(
                "Overlay color",
                "#FF2D2D",
                help="Example: choose orange if red is hard to see on your image.",
            )
        else:
            preset_values = PRESETS[preset]
            threshold = preset_values["threshold"]
            minimum_area = preset_values["minimum_area"]
            minimum_confidence = preset_values["minimum_confidence"]
            opacity = 0.55
            color = "#FF2D2D"
        show_hidden = st.checkbox("Show hidden detections", value=False)
        show_path = st.checkbox("Show driving path", value=True)
        if threshold < 0.25:
            st.warning("Very sensitive mode: more false alarms are likely.")
        st.caption(f"Running on: **{device.upper()}**")
        st.divider()
        st.info("Small training dataset (60 images). Experimental prototype; not for road-safety or autonomous-driving decisions.")

    st.write("**What does this app detect?** Possible unusual regions on a road image. The binary model does not identify object types.")
    uploaded = st.file_uploader("Choose road images", type=["png", "jpg", "jpeg", "webp"],
                                accept_multiple_files=True)
    examples = Path(__file__).parent / "examples"
    if examples.exists():
        example_files = list(examples.glob("*.png")) + list(examples.glob("*.jpg")) + list(examples.glob("*.jpeg"))
        if st.button("Try an example") and example_files:
            uploaded = [example_files[0]]
    if not uploaded:
        st.warning("Limitations: this is an experimental prototype trained on only 60 images. It can miss objects or raise false alarms and must not guide real road-safety decisions.")
        if st.session_state.history:
            st.subheader("Session history")
            st.dataframe(pd.DataFrame(st.session_state.history), hide_index=True, use_container_width=True)
        return

    settings = {
        "preset": preset,
        "threshold": threshold,
        "opacity": opacity,
        "minimum_area": minimum_area,
        "minimum_confidence": minimum_confidence,
        "show_hidden": show_hidden,
        "show_path": show_path,
        "color": tuple(int(color[i:i + 2], 16) for i in (1, 3, 5)),
    }
    progress = st.progress(0)
    summaries = []
    for index, file in enumerate(uploaded):
        name = file.name if hasattr(file, "name") else Path(file).name
        data = file.getvalue() if hasattr(file, "getvalue") else Path(file).read_bytes()
        st.subheader(f"📷 {name}")
        try:
            image = safe_image(data)
            with st.spinner("Analyzing image..."):
                probabilities = predict_probabilities(data, device)
            summaries.append(render_result(name, image, probabilities, settings))
        except (OSError, ValueError, RuntimeError) as exc:
            st.error(f"Could not analyze {name}: {exc}")
        progress.progress((index + 1) / len(uploaded))

    if summaries:
        st.subheader("Summary")
        summary_df = pd.DataFrame(summaries)
        st.dataframe(summary_df, hide_index=True, use_container_width=True)
        st.download_button("Download results as CSV", summary_df.to_csv(index=False).encode("utf-8"),
                           "road_anomaly_results.csv", "text/csv")
        st.session_state.history = (st.session_state.history + summaries)[-50:]

    st.markdown("---")
    st.caption("Model: DeepLabV3+ / ResNet34 · Experimental results only; not a substitute for professional road-safety decisions.")


if __name__ == "__main__":
    main()
