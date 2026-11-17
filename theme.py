"""
theme.py - shared flat design system (light / dark) for the three agents.

* No gradients, no emojis. Actions use Material icons (":material/name:").
* Dark mode is a user toggle (st.toggle in the sidebar, key "dark_mode").
  Streamlit cannot switch its built-in theme at run time, so the colours are
  applied through CSS variables and every table / card / chart is drawn with
  this module's helpers so that it follows the toggle.
"""
from __future__ import annotations

import html as _html

import pandas as pd
import streamlit as st

# ----------------------------------------------------------------- palettes
LIGHT = dict(
    bg="#f6f6f4", surface="#fcfcfb", raised="#ffffff", text="#0b0b0b", muted="#52514e",
    border="#d8d7d2", accent="#2a78d6", on_accent="#ffffff",
    good="#0b7a3b", warn="#a85d00", bad="#c0302f", info="#2a78d6",
    # categorical series (validated reference palette, fixed order)
    series=["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"],
    grid="#e4e3de",
)
DARK = dict(
    bg="#121211", surface="#1a1a19", raised="#222221", text="#f0efec", muted="#c3c2b7",
    border="#3a3a38", accent="#3987e5", on_accent="#0b0b0b",
    good="#4cc27c", warn="#e0a23a", bad="#ee7a79", info="#3987e5",
    series=["#3987e5", "#d95926", "#199e70", "#c98500", "#d55181", "#0ca30c", "#9085e9", "#e66767"],
    grid="#333331",
)


MODES = ["System", "Light", "Dark"]
MODE_LABELS = {"System": ":material/contrast: System", "Light": ":material/light_mode: Light",
               "Dark": ":material/dark_mode: Dark"}


def mode() -> str:
    m = st.session_state.get("theme_mode") or "System"
    return m if m in MODES else "System"


def system_is_dark() -> bool:
    """Theme reported by the browser / operating system (None until the browser has replied)."""
    try:
        return st.context.theme.type == "dark"
    except Exception:
        return False


def is_dark() -> bool:
    m = mode()
    if m == "Dark":
        return True
    if m == "Light":
        return False
    return system_is_dark()


def theme_switch() -> None:
    """Sidebar control: follow the system (default) or force light / dark."""
    st.radio("Theme", MODES, key="theme_mode", format_func=lambda m: MODE_LABELS[m])
    if mode() == "System":
        st.caption(f"Following your system: {'dark' if system_is_dark() else 'light'}")


def pal() -> dict:
    return DARK if is_dark() else LIGHT


def tone_color(tone: str | None) -> str:
    p = pal()
    return {"good": p["good"], "warn": p["warn"], "bad": p["bad"], "info": p["info"],
            None: p["muted"], "muted": p["muted"], "accent": p["accent"]}.get(tone, p["muted"])


# --------------------------------------------------------------------- CSS
def apply() -> None:
    p = pal()
    css = f"""
<style>
:root {{
  --bg:{p['bg']}; --surface:{p['surface']}; --raised:{p['raised']}; --text:{p['text']};
  --muted:{p['muted']}; --border:{p['border']}; --accent:{p['accent']}; --on-accent:{p['on_accent']};
  --good:{p['good']}; --warn:{p['warn']}; --bad:{p['bad']};
  color-scheme:{'dark' if is_dark() else 'light'};
}}
.stApp, [data-testid="stAppViewContainer"], [data-testid="stMain"] {{ background:var(--bg) !important; color:var(--text); }}
[data-testid="stHeader"], [data-testid="stBottom"], [data-testid="stBottom"] > div {{ background:var(--bg) !important; }}
[data-testid="stSidebar"], [data-testid="stSidebar"] > div {{ background:var(--surface) !important; }}
[data-testid="stSidebar"] {{ border-right:1px solid var(--border); }}

/* text */
.stApp h1,.stApp h2,.stApp h3,.stApp h4,.stApp h5,.stApp h6,
.stApp p,.stApp li,.stApp label,.stApp legend,
[data-testid="stMarkdownContainer"],[data-testid="stMarkdownContainer"] *,
[data-testid="stWidgetLabel"] *,[data-testid="stCaptionContainer"],
[data-testid="stSidebarNavLink"] span,[data-testid="stSidebarNavLink"] {{ color:var(--text); }}
[data-testid="stCaptionContainer"], [data-testid="stCaptionContainer"] *, small {{ color:var(--muted) !important; }}
.stApp a {{ color:var(--accent); }}
.stApp code {{ background:var(--raised); color:var(--text); border:1px solid var(--border); }}
hr {{ border-color:var(--border) !important; }}

/* sidebar navigation */
[data-testid="stSidebarNavLink"][aria-current="page"] {{ background:var(--raised); border:1px solid var(--border); }}

/* inputs */
[data-baseweb="input"], [data-baseweb="base-input"], [data-baseweb="textarea"], [data-baseweb="select"] > div,
.stTextInput input, .stNumberInput input, .stTextArea textarea, .stDateInput input {{
  background:var(--raised) !important; color:var(--text) !important; border-color:var(--border) !important;
}}
[data-baseweb="input"], [data-baseweb="base-input"], [data-baseweb="textarea"] {{ border:1px solid var(--border) !important; border-radius:6px; }}
input::placeholder, textarea::placeholder {{ color:var(--muted) !important; opacity:.8; }}
[data-baseweb="select"] * {{ color:var(--text) !important; }}
[data-baseweb="select"] svg, [data-baseweb="input"] svg {{ fill:var(--muted) !important; }}
[data-testid="stNumberInputStepUp"], [data-testid="stNumberInputStepDown"] {{ background:var(--raised) !important; color:var(--text) !important; }}
[data-baseweb="popover"], [data-baseweb="popover"] > div, [data-baseweb="menu"], [data-baseweb="calendar"],
[data-baseweb="popover"] ul {{ background:var(--raised) !important; color:var(--text) !important; }}
[data-baseweb="popover"] li, [data-baseweb="popover"] [role="option"] {{ background:var(--raised) !important; color:var(--text) !important; }}
[data-baseweb="popover"] li:hover, [data-baseweb="popover"] [role="option"]:hover,
[data-baseweb="popover"] [aria-selected="true"] {{ background:var(--border) !important; }}
[data-baseweb="calendar"] * {{ color:var(--text) !important; background-color:transparent; }}
[data-baseweb="calendar"] [aria-selected="true"], [data-baseweb="calendar"] [aria-selected="true"] * {{ background:var(--accent) !important; color:var(--on-accent) !important; }}
[data-baseweb="tag"] {{ background:var(--border) !important; }}

/* buttons */
button[kind="secondary"], button[kind="secondaryFormSubmit"], button[kind="tertiary"] {{
  background:var(--raised) !important; color:var(--text) !important; border:1px solid var(--border) !important;
  border-radius:6px; box-shadow:none !important; }}
button[kind="secondary"]:hover, button[kind="secondaryFormSubmit"]:hover {{ border-color:var(--accent) !important; color:var(--accent) !important; }}
button[kind="secondary"] *, button[kind="secondaryFormSubmit"] *, button[kind="tertiary"] * {{ color:inherit !important; }}
button[kind="primary"], button[kind="primaryFormSubmit"] {{
  background:var(--accent) !important; color:var(--on-accent) !important; border:1px solid var(--accent) !important;
  border-radius:6px; box-shadow:none !important; }}
button[kind="primary"] *, button[kind="primaryFormSubmit"] * {{ color:var(--on-accent) !important; }}
button[kind="primary"]:hover, button[kind="primaryFormSubmit"]:hover {{ opacity:.9; }}

/* tabs */
[data-baseweb="tab-list"] {{ border-bottom:1px solid var(--border); gap:4px; }}
[data-baseweb="tab"] {{ background:transparent !important; }}
[data-baseweb="tab"] p, [data-baseweb="tab"] span {{ color:var(--muted) !important; font-weight:500; }}
[data-baseweb="tab"][aria-selected="true"] p, [data-baseweb="tab"][aria-selected="true"] span {{ color:var(--text) !important; }}
[data-baseweb="tab-highlight"] {{ background:var(--accent) !important; height:3px; }}
[data-baseweb="tab-border"] {{ background:var(--border) !important; }}

/* expander, forms, chat */
[data-testid="stExpander"] details, [data-testid="stForm"] {{ background:var(--surface); border:1px solid var(--border) !important; border-radius:8px; }}
[data-testid="stExpander"] summary, [data-testid="stExpander"] summary * {{ color:var(--text) !important; }}
[data-testid="stChatMessage"] {{ background:var(--surface); border:1px solid var(--border); border-radius:8px; }}
[data-testid="stChatMessage"] * {{ color:var(--text); }}
[data-testid="stChatInput"], [data-testid="stChatInput"] > div, [data-testid="stChatInput"] textarea {{ background:var(--raised) !important; color:var(--text) !important; border-color:var(--border) !important; }}
[data-testid="stChatInput"] button {{ background:var(--accent) !important; }}
[data-testid="stChatInput"] button svg {{ fill:var(--on-accent) !important; }}
[data-testid="stChatMessageAvatarUser"], [data-testid="stChatMessageAvatarAssistant"] {{ background:var(--accent) !important; }}
[data-testid="stChatMessageAvatarUser"] *, [data-testid="stChatMessageAvatarAssistant"] * {{ color:var(--on-accent) !important; fill:var(--on-accent) !important; }}
[data-testid="stCheckbox"] label span, [data-testid="stRadio"] label p, [data-testid="stToggle"] label p {{ color:var(--text) !important; }}
[data-testid="stFileUploader"] section {{ background:var(--raised) !important; border:1px dashed var(--border); }}
[data-testid="stFileUploader"] * {{ color:var(--text) !important; }}
[data-testid="stFileUploader"] button {{ background:var(--surface) !important; }}

/* custom components */
.ui-hero {{ background:var(--surface); border:1px solid var(--border); border-left:6px solid var(--accent);
  padding:18px 22px; border-radius:8px; margin:0 0 18px 0; }}
.ui-hero h1 {{ margin:0; font-size:1.7rem; padding:0; color:var(--text); }}
.ui-hero p {{ margin:4px 0 0 0; color:var(--muted); }}
.ui-card {{ background:var(--surface); border:1px solid var(--border); border-radius:8px; padding:14px 16px; height:100%; }}
.ui-card .l {{ color:var(--muted); font-size:.82rem; text-transform:uppercase; letter-spacing:.04em; }}
.ui-card .v {{ color:var(--text); font-size:1.7rem; font-weight:700; line-height:1.25; word-break:break-word; }}
.ui-card .s {{ color:var(--muted); font-size:.82rem; }}
.ui-callout {{ background:var(--surface); border:1px solid var(--border); border-left:5px solid var(--muted);
  padding:10px 14px; border-radius:6px; margin:8px 0; color:var(--text); }}
.ui-badge {{ display:inline-block; padding:1px 9px; border-radius:4px; font-size:.78rem; font-weight:600;
  border:1px solid currentColor; }}
.ui-table-wrap {{ overflow-x:auto; border:1px solid var(--border); border-radius:8px; background:var(--surface); }}
table.ui-table {{ border-collapse:collapse; width:100%; font-size:.92rem; }}
table.ui-table th {{ text-align:left; padding:8px 12px; background:var(--raised); color:var(--muted);
  border-bottom:1px solid var(--border); font-weight:600; white-space:nowrap; }}
table.ui-table td {{ padding:7px 12px; border-bottom:1px solid var(--border); color:var(--text); }}
table.ui-table tr:last-child td {{ border-bottom:none; }}
table.ui-table td.num, table.ui-table th.num {{ text-align:right; font-variant-numeric:tabular-nums; }}
.ui-bar {{ background:var(--border); border-radius:4px; height:8px; width:100%; }}
.ui-bar > div {{ height:8px; border-radius:4px; background:var(--accent); }}
</style>
"""
    st.markdown(css, unsafe_allow_html=True)


# --------------------------------------------------------------- components
def hero(title: str, subtitle: str = "") -> None:
    st.markdown(f'<div class="ui-hero"><h1>{_html.escape(title)}</h1><p>{_html.escape(subtitle)}</p></div>',
                unsafe_allow_html=True)


def card(label: str, value: str, sub: str = "", tone: str | None = None) -> str:
    colour = f"color:{tone_color(tone)};" if tone else ""
    size = "font-size:1.15rem;" if len(str(value)) > 14 else ""
    return (f'<div class="ui-card"><div class="l">{_html.escape(label)}</div>'
            f'<div class="v" style="{colour}{size}">{_html.escape(str(value))}</div>'
            f'<div class="s">{_html.escape(sub)}</div></div>')


def cards(items: list[tuple]) -> None:
    """items: (label, value, sub, tone) tuples rendered in one row."""
    cols = st.columns(len(items))
    for col, it in zip(cols, items):
        label, value = it[0], it[1]
        sub = it[2] if len(it) > 2 else ""
        tone = it[3] if len(it) > 3 else None
        col.markdown(card(label, value, sub, tone), unsafe_allow_html=True)


def callout(text: str, tone: str | None = None) -> None:
    """Flat note with a coloured left edge. `text` is plain text (escaped)."""
    st.markdown(f'<div class="ui-callout" style="border-left-color:{tone_color(tone)}">{_html.escape(text)}</div>',
                unsafe_allow_html=True)


def badge(text: str, tone: str | None = None) -> str:
    return f'<span class="ui-badge" style="color:{tone_color(tone)}">{_html.escape(text)}</span>'


def html_table(df: pd.DataFrame, num_cols: tuple = (), tone_col: str | None = None, tones: dict | None = None) -> None:
    """Theme-aware table (the canvas based st.dataframe ignores the theme switch)."""
    if df is None or df.empty:
        st.caption("No rows to show.")
        return
    head = "".join(f'<th class="{"num" if c in num_cols else ""}">{_html.escape(str(c))}</th>' for c in df.columns)
    body = ""
    for _, row in df.iterrows():
        cells = ""
        for c in df.columns:
            v = row[c]
            txt = _html.escape("" if pd.isna(v) else str(v))
            style = ""
            if tone_col == c and tones:
                style = f' style="color:{tone_color(tones.get(v))};font-weight:600"'
            if len(txt) <= 24:
                style += ' style="white-space:nowrap"' if not style else ""
                style = style.replace('style="color', 'style="white-space:nowrap;color') if "color" in style else style
            cells += f'<td class="{"num" if c in num_cols else ""}"{style}>{txt}</td>'
        body += f"<tr>{cells}</tr>"
    st.markdown(f'<div class="ui-table-wrap"><table class="ui-table"><tr>{head}</tr>{body}</table></div>',
                unsafe_allow_html=True)


def bar(pct: float) -> str:
    pct = max(0.0, min(100.0, pct))
    return f'<div class="ui-bar"><div style="width:{pct:.0f}%"></div></div>'


def style_fig(fig, height: int = 360, legend: bool = True):
    """Apply the flat theme to a Plotly figure."""
    p = pal()
    fig.update_layout(
        height=height, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color=p["text"], size=13), margin=dict(t=56, b=20, l=10, r=10),
        colorway=p["series"], showlegend=legend,
        legend=dict(orientation="h", yanchor="bottom", y=1.0, xanchor="right", x=1, font=dict(color=p["muted"])),
        title=dict(font=dict(size=15, color=p["text"])),
        hoverlabel=dict(bgcolor=p["raised"], font=dict(color=p["text"]), bordercolor=p["border"]),
    )
    fig.update_xaxes(gridcolor=p["grid"], zerolinecolor=p["grid"], linecolor=p["border"],
                     tickfont=dict(color=p["muted"]), title_font=dict(color=p["muted"]))
    fig.update_yaxes(gridcolor=p["grid"], zerolinecolor=p["grid"], linecolor=p["border"],
                     tickfont=dict(color=p["muted"]), title_font=dict(color=p["muted"]))
    return fig


def show(fig, key: str | None = None) -> None:
    st.plotly_chart(fig, width="stretch", key=key, config={"displayModeBar": False})
