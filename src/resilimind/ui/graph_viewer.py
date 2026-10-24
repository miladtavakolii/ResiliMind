import math
import logging
import streamlit as st
import plotly.graph_objects as go
from typing import Any, Dict, List, Optional, Set, Tuple

from resilimind.core.agents.common import resilience_graph

logger = logging.getLogger(__name__)

# Complete configuration for the 6 resilience domains
DOMAIN_CONFIG: Dict[str, Dict[str, Any]] = {
    "Personal_Resilience": {
        "name_fa": "تاب‌آوری فردی",
        "name_en": "Personal Resilience",
        "color": "#38bdf8",          # Sky Blue
        "border": "#0284c7",
        "icon": "🧘",
        "badge_bg": "rgba(56, 189, 248, 0.15)",
        "badge_border": "rgba(56, 189, 248, 0.4)",
        "badge_color": "#38bdf8",
        "description": "توانمندی‌های درونی، تنظیم هیجان، خودکارآمدی و چشم‌انداز آینده فردی"
    },
    "Social_Resilience": {
        "name_fa": "تاب‌آوری اجتماعی",
        "name_en": "Social Resilience",
        "color": "#34d399",          # Emerald Green
        "border": "#10b981",
        "icon": "🤝",
        "badge_bg": "rgba(52, 211, 153, 0.15)",
        "badge_border": "rgba(52, 211, 153, 0.4)",
        "badge_color": "#34d399",
        "description": "پایگاه امن خانوادگی، شبکه حمایت دوستان و سرمایه اجتماعی"
    },
    "Economic_Resilience": {
        "name_fa": "تاب‌آوری اقتصادی",
        "name_en": "Economic Resilience",
        "color": "#fbbf24",          # Amber / Gold
        "border": "#d97706",
        "icon": "💼",
        "badge_bg": "rgba(251, 191, 36, 0.15)",
        "badge_border": "rgba(251, 191, 36, 0.4)",
        "badge_color": "#fbbf24",
        "description": "انطباق‌پذیری مالی، امنیت حرفه‌ای و تاب‌آوری در برابر شوک‌های معیشتی"
    },
    "Physical_Resilience": {
        "name_fa": "تاب‌آوری جسمانی",
        "name_en": "Physical Resilience",
        "color": "#fb923c",          # Coral Orange
        "border": "#ea580c",
        "icon": "🏃",
        "badge_bg": "rgba(251, 146, 60, 0.15)",
        "badge_border": "rgba(251, 146, 60, 0.4)",
        "badge_color": "#fb923c",
        "description": "پاسخ‌های تنشی بدن، سطح انرژی، کیفیت خواب و ریتم‌های بیولوژیک"
    },
    "Political_Resilience": {
        "name_fa": "تاب‌آوری سیاسی-مدنی",
        "name_en": "Political Resilience",
        "color": "#c084fc",          # Purple
        "border": "#9333ea",
        "icon": "🏛️",
        "badge_bg": "rgba(192, 132, 252, 0.15)",
        "badge_border": "rgba(192, 132, 252, 0.4)",
        "badge_color": "#c084fc",
        "description": "مصونیت شناختی در برابر اخبار جعلی، احساس عاملیت و مشارکت شهروندی"
    },
    "Spiritual_Cultural_Resilience": {
        "name_fa": "تاب‌آوری معنوی-فرهنگی",
        "name_en": "Spiritual & Cultural Resilience",
        "color": "#f472b6",          # Rose / Pink
        "border": "#db2777",
        "icon": "🕊️",
        "badge_bg": "rgba(244, 114, 182, 0.15)",
        "badge_border": "rgba(244, 114, 182, 0.4)",
        "badge_color": "#f472b6",
        "description": "معنابخشی به رنج‌ها، امید اگزیستانسیال و پیوستگی با هویت فرهنگی"
    }
}

STATUS_META: Dict[str, Dict[str, str]] = {
    "GREEN": {"label": "🟢 پایدار", "color": "#10b981", "badge_bg": "rgba(16, 185, 129, 0.15)"},
    "YELLOW": {"label": "🟡 هشدار", "color": "#f59e0b", "badge_bg": "rgba(245, 158, 11, 0.15)"},
    "RED": {"label": "🔴 بحرانی", "color": "#ef4444", "badge_bg": "rgba(239, 68, 68, 0.15)"}
}

# Concise 2-3 word Persian titles for network display to prevent label collision
# (The full comprehensive scientific name remains in hover cards and detail matrix)
SHORT_NODE_TITLES: Dict[str, str] = {
    "IND_PER_01": "خودکارآمدی فردی",
    "IND_PER_02": "تنظیم هیجان",
    "IND_PER_03": "چشم‌انداز آینده",
    "IND_POL_01": "مصونیت شناختی",
    "IND_POL_02": "عاملیت سیاسی",
    "IND_ECO_01": "انطباق مالی",
    "IND_ECO_02": "امنیت شغلی",
    "IND_PHY_01": "پاسخ تنشی بدن",
    "IND_PHY_02": "ریتم بیولوژیک",
    "IND_SOC_01": "انسجام خانواده",
    "IND_SOC_02": "حمایت دوستان",
    "IND_SPI_01": "معنابخشی معنوی",
    "IND_SPI_02": "هویت فرهنگی"
}


def get_radial_text_position(x: float, y: float) -> str:
    """Calculates outward radial textposition to project labels away from circle center."""
    angle_deg = math.degrees(math.atan2(y, x)) % 360
    # Categorize into 8 sectors
    if 337.5 <= angle_deg or angle_deg < 22.5:
        return "middle right"
    elif 22.5 <= angle_deg < 67.5:
        return "top right"
    elif 67.5 <= angle_deg < 112.5:
        return "top center"
    elif 112.5 <= angle_deg < 157.5:
        return "top left"
    elif 157.5 <= angle_deg < 202.5:
        return "middle left"
    elif 202.5 <= angle_deg < 247.5:
        return "bottom left"
    elif 247.5 <= angle_deg < 292.5:
        return "bottom center"
    else: # 292.5 <= angle_deg < 337.5
        return "bottom right"


def build_resilience_network_figure(
    active_nodes: List[str],
    assessments_by_node: Dict[str, Dict[str, Any]],
    domain_filter: str = "همه حوزه‌ها"
) -> go.Figure:
    """
    Builds a native Plotly Circular Network Graph with deterministic domain clustering,
    directional flow arrows, non-overlapping concise labels, and active node highlights.
    """
    domain_order = list(DOMAIN_CONFIG.keys())
    num_domains = len(domain_order)
    angle_step = 2 * math.pi / num_domains

    # 1. Deterministic Circular Polar Layout with radial & angular staggering
    pos: Dict[str, Tuple[float, float]] = {}
    domain_label_pos: Dict[str, Tuple[float, float]] = {}

    for d_idx, dom in enumerate(domain_order):
        nodes_in_dom = [n for n, d in resilience_graph.nodes(data=True) if d.get("domain") == dom]
        center_angle = d_idx * angle_step
        num_nodes = len(nodes_in_dom)

        # Place domain title on outer perimeter ring (R=1.75) safely beyond node labels
        domain_label_pos[dom] = (1.75 * math.cos(center_angle), 1.75 * math.sin(center_angle))

        for n_idx, node_id in enumerate(nodes_in_dom):
            if num_nodes == 1:
                angle = center_angle
                radius = 1.05
            elif num_nodes == 2:
                # 2 nodes: alternate radii and spread by angle
                offset = (-0.20 if n_idx == 0 else 0.20) * angle_step
                angle = center_angle + offset
                radius = 0.95 if n_idx == 0 else 1.12
            else:
                # 3 nodes: middle node pushed out, flanks pulled in
                offset = (n_idx - (num_nodes - 1) / 2.0) * (angle_step * 0.32)
                angle = center_angle + offset
                radius = 1.15 if n_idx == 1 else 0.94

            pos[node_id] = (radius * math.cos(angle), radius * math.sin(angle))

    active_set = set(active_nodes)
    fig = go.Figure()

    # 2. Add Network Edges
    for u, v, data in resilience_graph.edges(data=True):
        if domain_filter != "همه حوزه‌ها":
            u_dom = resilience_graph.nodes.get(u, {}).get("domain", "")
            v_dom = resilience_graph.nodes.get(v, {}).get("domain", "")
            filtered_dom_key = next((k for k, v in DOMAIN_CONFIG.items() if v["name_fa"] == domain_filter), None)
            if filtered_dom_key and u_dom != filtered_dom_key and v_dom != filtered_dom_key:
                continue

        x0, y0 = pos[u]
        x1, y1 = pos[v]

        u_active = u in active_set
        v_active = v in active_set
        is_cross = "CROSS_DOMAIN" in data.get("relation_type", "")

        # Edge color & width
        if u_active and v_active:
            edge_color = "#38bdf8"
            edge_width = 3.5
            dash = "solid"
        elif u_active or v_active:
            edge_color = "#0284c7"
            edge_width = 2.4
            dash = "solid"
        elif is_cross:
            edge_color = "rgba(192, 132, 252, 0.45)"
            edge_width = 1.6
            dash = "dash"
        else:
            edge_color = "rgba(148, 163, 184, 0.25)"
            edge_width = 1.2
            dash = "solid"

        # Edge line trace
        fig.add_trace(go.Scatter(
            x=[x0, x1, None],
            y=[y0, y1, None],
            mode="lines",
            line=dict(width=edge_width, color=edge_color, dash=dash),
            hoverinfo="none",
            showlegend=False
        ))

        # Directional Arrowhead Annotation
        arrow_x = x0 + 0.82 * (x1 - x0)
        arrow_y = y0 + 0.82 * (y1 - y0)
        fig.add_annotation(
            x=arrow_x,
            y=arrow_y,
            ax=x0,
            ay=y0,
            xref="x",
            yref="y",
            axref="x",
            ayref="y",
            showarrow=True,
            arrowhead=2,
            arrowsize=1.1,
            arrowwidth=edge_width,
            arrowcolor=edge_color
        )

    # 3. Add Domain Circumference Sector Badges (Outer Orbit)
    for dom, (lx, ly) in domain_label_pos.items():
        meta = DOMAIN_CONFIG[dom]
        fig.add_annotation(
            x=lx,
            y=ly,
            text=f"<b>{meta['icon']} {meta['name_fa']}</b>",
            showarrow=False,
            font=dict(family="Vazirmatn", size=12.5, color=meta["color"]),
            bgcolor="rgba(15, 23, 42, 0.90)",
            bordercolor=meta["border"],
            borderwidth=1.5,
            borderpad=6
        )

    # 4. Add Node Traces (Grouped by Domain for Interactive Legend)
    for dom_key in domain_order:
        meta = DOMAIN_CONFIG[dom_key]
        if domain_filter != "همه حوزه‌ها" and meta["name_fa"] != domain_filter:
            continue

        nodes_in_dom = [n for n, d in resilience_graph.nodes(data=True) if d.get("domain") == dom_key]

        x_coords, y_coords = [], []
        marker_sizes, marker_colors = [], []
        marker_line_widths, marker_line_colors = [], []
        hover_texts, label_texts = [], []
        text_positions = []

        for node_id in nodes_in_dom:
            data = resilience_graph.nodes[node_id]
            x, y = pos[node_id]
            x_coords.append(x)
            y_coords.append(y)

            name_fa = data.get("name_fa", node_id)
            short_name = SHORT_NODE_TITLES.get(node_id, name_fa)
            name_en = data.get("name_en", "")
            description = data.get("description", "")
            is_active = node_id in active_set

            assessment = assessments_by_node.get(node_id)
            status_raw = assessment.get("status", "YELLOW").upper() if assessment else None
            score = assessment.get("score") if assessment else None
            confidence = int(float(assessment.get("confidence", 0.0)) * 100) if assessment else None
            reasoning = assessment.get("reasoning", "") if assessment else ""

            if is_active:
                status_color = STATUS_META.get(status_raw, {}).get("color", meta["color"])
                marker_sizes.append(36)
                marker_colors.append(meta["color"])
                marker_line_widths.append(4.5)
                marker_line_colors.append(status_color)
                label_texts.append(f"⚡ <b>{short_name}</b>")
            else:
                marker_sizes.append(22)
                marker_colors.append(meta["color"])
                marker_line_widths.append(2.0)
                marker_line_colors.append(meta["border"])
                label_texts.append(f"<b>{short_name}</b>")

            # Outward radial position for this specific node
            text_positions.append(get_radial_text_position(x, y))

            # Rich Persian Hover Card (Preserves the full scientific title and details)
            status_text = STATUS_META.get(status_raw, {}).get("label", "ثبت نشده در این نشست")
            score_text = f"{score}/100" if score is not None else "محاسبه نشده"
            conf_text = f"{confidence}%" if confidence is not None else "N/A"

            hover_html = (
                f"<b>{meta['icon']} {name_fa}</b> ({node_id})<br>"
                f"<b>حوزه:</b> {meta['name_fa']} | <b>نام علمی:</b> {name_en}<br>"
                f"<b>وضعیت در پیام اخیر:</b> {status_text}<br>"
                f"<b>امتیاز تاب‌آوری:</b> {score_text} | <b>اطمینان:</b> {conf_text}<br>"
                f"--------------------------------------------------<br>"
                f"<b>تعریف نود:</b> {description[:160]}..."
            )
            if reasoning:
                hover_html += f"<br><b>استدلال مدل:</b> {reasoning[:120]}..."

            hover_texts.append(hover_html)

        fig.add_trace(go.Scatter(
            x=x_coords,
            y=y_coords,
            mode="markers+text",
            name=f"{meta['icon']} {meta['name_fa']}",
            marker=dict(
                size=marker_sizes,
                color=marker_colors,
                line=dict(width=marker_line_widths, color=marker_line_colors),
                opacity=0.95
            ),
            text=label_texts,
            textposition=text_positions,
            textfont=dict(family="Vazirmatn", size=12.5, color="#f8fafc"),
            hoverinfo="text",
            hovertext=hover_texts,
            showlegend=True
        ))

    # 5. Styling Layout
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        hoverlabel=dict(
            bgcolor="#0f172a",
            bordercolor="#38bdf8",
            font_size=12.5,
            font_family="Vazirmatn",
            font_color="#f8fafc"
        ),
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False, range=[-2.25, 2.25]),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False, range=[-2.05, 2.05], scaleanchor="x", scaleratio=1),
        margin=dict(l=20, r=20, t=30, b=30),
        height=680,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=-0.12,
            xanchor="center",
            x=0.5,
            font=dict(family="Vazirmatn", size=11.5, color="#cbd5e1"),
            bgcolor="rgba(15, 23, 42, 0.6)",
            bordercolor="rgba(255, 255, 255, 0.1)",
            borderwidth=1
        )
    )

    return fig


def render_domain_matrix_view(active_nodes: List[str], assessments_by_node: Dict[str, Dict[str, Any]]) -> None:
    """Renders the comprehensive 6-domain glassmorphic card matrix with crystal-clear definitions."""
    st.markdown("### 🗂️ ماتریس ۶ گانه حوزه‌ها و شناسنامه نودهای تاب‌آوری")
    st.caption("در این بخش می‌توانید هر حوزه را با نودهای زیرمجموعه، تعاریف علمی، و شواهد متنی با وضوح بالا مطالعه کنید:")

    active_set = set(active_nodes)
    domain_keys = list(DOMAIN_CONFIG.keys())

    # 3-Column Layout
    col1, col2, col3 = st.columns(3)
    columns = [col1, col2, col3]

    for idx, d_key in enumerate(domain_keys):
        meta = DOMAIN_CONFIG[d_key]
        target_col = columns[idx % 3]

        nodes_in_dom = [n for n, d in resilience_graph.nodes(data=True) if d.get("domain") == d_key]

        with target_col:
            # Domain Container Header
            st.markdown(f"""
            <div style="background:rgba(255, 255, 255, 0.03); border:1.5px solid {meta['border']}; border-radius:14px; padding:14px; margin-bottom:16px;">
                <div style="display:flex; align-items:center; justify-content:space-between; margin-bottom:6px;">
                    <span style="font-size:1.15rem; font-weight:700; color:{meta['color']};">
                        {meta['icon']} {meta['name_fa']}
                    </span>
                    <span style="font-size:0.75rem; background:{meta['badge_bg']}; color:{meta['badge_color']}; padding:3px 8px; border-radius:12px; font-weight:600;">
                        {len(nodes_in_dom)} نود
                    </span>
                </div>
                <div style="font-size:0.83rem; color:#94a3b8; line-height:1.5; margin-bottom:12px;">
                    {meta['description']}
                </div>
            </div>
            """, unsafe_allow_html=True)

            # Node Cards inside Domain
            for node_id in nodes_in_dom:
                data = resilience_graph.nodes[node_id]
                name_fa = data.get("name_fa", node_id)
                name_en = data.get("name_en", "")
                description = data.get("description", "")
                is_active = node_id in active_set

                assessment = assessments_by_node.get(node_id)
                status_raw = assessment.get("status", "YELLOW").upper() if assessment else None
                score = assessment.get("score") if assessment else None
                confidence = int(float(assessment.get("confidence", 0.0)) * 100) if assessment else None
                reasoning = assessment.get("reasoning", "") if assessment else ""

                border_style = f"1.5px solid {STATUS_META.get(status_raw, {}).get('color', '#38bdf8')}" if is_active else "1px solid rgba(255, 255, 255, 0.08)"
                bg_style = "rgba(56, 189, 248, 0.06)" if is_active else "rgba(255, 255, 255, 0.02)"

                with st.container(border=True):
                    # Header
                    if is_active:
                        st.markdown(
                            f"<div style='font-size:0.8rem; color:#38bdf8; font-weight:700; margin-bottom:4px;'>"
                            f"⚡ نود فعال در مکالمه اخیر | وضعیت: {STATUS_META.get(status_raw, {}).get('label', '')}"
                            f"</div>",
                            unsafe_allow_html=True
                        )

                    st.markdown(f"**{name_fa}**")
                    st.caption(f"`{node_id}` — {name_en}")

                    if is_active and score is not None:
                        st.markdown(
                            f"<div style='font-size:0.85rem; background:rgba(255,255,255,0.05); padding:6px 10px; border-radius:8px; margin:6px 0;'>"
                            f"📊 <strong>امتیاز:</strong> <code>{score}/100</code> | <strong>اطمینان:</strong> <code>{confidence}%</code>"
                            f"</div>",
                            unsafe_allow_html=True
                        )
                        if reasoning:
                            st.caption(f"🧠 *استدلال:* {reasoning}")

                    st.markdown(f"<div style='font-size:0.88rem; line-height:1.7; color:#e2e8f0; margin:8px 0;'>{description}</div>", unsafe_allow_html=True)

                    # Expandable details
                    with st.expander("🔍 جزئیات، نشانه‌ها و راهکارها"):
                        st.markdown(f"**مرز معنایی:** {data.get('semantic_boundary', 'تعریف نشده')}")

                        cues = data.get("cues", {})
                        if cues.get("positive_keywords"):
                            st.markdown(f"🟢 **کلمات کلیدی مثبت:** {', '.join(cues['positive_keywords'])}")
                        if cues.get("negative_keywords"):
                            st.markdown(f"🔴 **کلمات کلیدی منفی:** {', '.join(cues['negative_keywords'])}")

                        interventions = data.get("interventions", {})
                        if interventions:
                            if "on_red" in interventions:
                                st.markdown(f"🚨 **مداخله بحرانی (قرمز):** {interventions['on_red']}")
                            if "on_yellow" in interventions:
                                st.markdown(f"⚠️ **مداخله هشدار (زرد):** {interventions['on_yellow']}")


def render_cross_domain_ripples_view(active_nodes: List[str]) -> None:
    """Renders comprehensive causal relationship chains and cross-domain ripple effects."""
    st.markdown("### ⚡ اطلس روابط علی و سرریزهای بین‌حوزه‌ای")
    st.caption("گراف دانش ResiliMind دارای ۱۵ یال مستقیم است که چگونگی سرریز بحران یا منابع قدرت از یک حوزه به حوزه دیگر را تبیین می‌کنند:")

    active_set = set(active_nodes)

    # 1. Active Relationships in Current Session
    active_edges = []
    other_cross_edges = []
    intra_edges = []

    for u, v, data in resilience_graph.edges(data=True):
        u_data = resilience_graph.nodes.get(u, {})
        v_data = resilience_graph.nodes.get(v, {})
        item = {
            "source_id": u,
            "source_name": u_data.get("name_fa", u),
            "source_domain": u_data.get("domain", ""),
            "target_id": v,
            "target_name": v_data.get("name_fa", v),
            "target_domain": v_data.get("domain", ""),
            "relation_type": data.get("relation_type", ""),
            "description": data.get("description", ""),
            "is_active": u in active_set or v in active_set
        }

        if item["is_active"]:
            active_edges.append(item)
        elif "CROSS_DOMAIN" in item["relation_type"]:
            other_cross_edges.append(item)
        else:
            intra_edges.append(item)

    if active_edges:
        st.markdown("#### ⚡ ۱. روابط فعال در پیام اخیر شما:")
        st.info("نودهای زیر در مکالمه شما شناسایی شده و زنجیره اثرات علی زیر را فعال کرده‌اند:")
        for edge in active_edges:
            s_meta = DOMAIN_CONFIG.get(edge["source_domain"], {"name_fa": "", "icon": "📍", "color": "#38bdf8"})
            t_meta = DOMAIN_CONFIG.get(edge["target_domain"], {"name_fa": "", "icon": "📍", "color": "#38bdf8"})

            with st.container(border=True):
                st.markdown(
                    f"**{s_meta['icon']} {edge['source_name']}** "
                    f"<span style='color:#38bdf8; font-weight:700;'>➔ ➔ ➔</span> "
                    f"**{t_meta['icon']} {edge['target_name']}**",
                    unsafe_allow_html=True
                )
                st.caption(f"🧭 نوع پیوند: `{edge['relation_type']}` | از **{s_meta['name_fa']}** به **{t_meta['name_fa']}**")
                st.markdown(f'<div class="evidence-quote">🧠 {edge["description"]}</div>', unsafe_allow_html=True)

        st.divider()

    # 2. All Cross-Domain Edges
    st.markdown("#### 🌐 ۲. پیوندهای بنیادین بین‌حوزه‌ای (Cross-Domain Bridges):")
    st.caption("این پیوندها نشان می‌دهند که چگونه ابعاد مختلف زندگی (اقتصادی، جسمانی، اجتماعی، سیاسی و معنوی) بر روان و تاب‌آوری فردی اثر متقابل دارند:")

    for edge in other_cross_edges:
        s_meta = DOMAIN_CONFIG.get(edge["source_domain"], {"name_fa": "", "icon": "📍", "color": "#cbd5e1"})
        t_meta = DOMAIN_CONFIG.get(edge["target_domain"], {"name_fa": "", "icon": "📍", "color": "#cbd5e1"})

        with st.container(border=True):
            st.markdown(
                f"**{s_meta['icon']} {edge['source_name']}** "
                f"<span style='color:#94a3b8;'>➔</span> "
                f"**{t_meta['icon']} {edge['target_name']}**",
                unsafe_allow_html=True
            )
            st.caption(f"🧭 نوع پیوند: `{edge['relation_type']}` | از حوزه **{s_meta['name_fa']}** به **{t_meta['name_fa']}**")
            st.markdown(f'<div style="font-size:0.9rem; color:#cbd5e1; line-height:1.7; margin-top:4px;">{edge["description"]}</div>', unsafe_allow_html=True)

    st.divider()

    # 3. Intra-Domain Edges
    st.markdown("#### 🔄 ۳. تقویت و تثبیت درون‌حوزه‌ای (Intra-Domain Reinforcement):")
    for edge in intra_edges:
        s_meta = DOMAIN_CONFIG.get(edge["source_domain"], {"name_fa": "", "icon": "📍"})
        with st.container(border=True):
            st.markdown(f"**{s_meta['icon']} {edge['source_name']}** ➔ **{edge['target_name']}** (`{edge['relation_type']}`)")
            st.caption(edge["description"])


def render_knowledge_graph_view() -> None:
    """Renders the executive, completely redesigned Knowledge Graph view."""
    # Top Header
    st.markdown("""
    <div class="main-header" style="background: linear-gradient(135deg, #091e3a 0%, #172554 100%);">
        <h2 style="margin:0; font-weight:700;">🌱 اطلس شبکه گراف دانش تاب‌آوری (Resilience Knowledge Atlas)</h2>
        <p style="margin:6px 0 0 0; opacity:0.85; font-size:0.95rem;">
            شبکه علی ۱۳ نودی و ۱۵ یال بین‌حوزه‌ای در ۶ بعد اصلی؛ مدل‌سازی چندبعدی پدیده‌های روانی بر مبنای گراف دانش
        </p>
    </div>
    """, unsafe_allow_html=True)

    # State extraction
    last_state = st.session_state.get("last_state") or {}
    active_nodes: List[str] = last_state.get("active_nodes", [])
    assessments: List[Dict[str, Any]] = last_state.get("assessments", [])
    assessments_by_node = {a.get("node_id", ""): a for a in assessments}

    # Active Session Alert Banner
    if active_nodes:
        st.markdown(
            f"<div style='margin-bottom:20px; padding:12px 18px; background:rgba(56,189,248,0.12); "
            f"border:1.5px solid #38bdf8; border-radius:12px; color:#f8fafc; font-size:0.95rem;'>"
            f"⚡ <strong>وضعیت فعال نشست جاری:</strong> <code>{len(active_nodes)} بعد تاب‌آوری</code> در پیام اخیر شما شناسایی شد: "
            f"<span style='color:#38bdf8; font-weight:700;'>{', '.join(active_nodes)}</span>"
            f"</div>",
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            "<div style='margin-bottom:20px; padding:10px 16px; background:rgba(255,255,255,0.03); "
            "border:1px solid rgba(255,255,255,0.08); border-radius:12px; color:#94a3b8; font-size:0.9rem;'>"
            "ℹ️ <strong>حالت پایه:</strong> تمام ۱۳ نود و پیوندهای علی در وضعیت پیش‌فرض نمایش داده شده‌اند. با ارسال پیام در تب گفتگو، نودهای مرتبط به صورت زنده هایلایت می‌شوند."
            "</div>",
            unsafe_allow_html=True
        )

    # 3-Tier Multi-Perspective Views
    view_tab1, view_tab2, view_tab3 = st.tabs([
        "🕸️ نقشه شبکه تعاملی (Network Map)",
        "🗂️ ماتریس ۶ گانه حوزه‌ها و نودها (Domain Cards)",
        "⚡ تحلیل اثرات سرریز و پیوندهای علی (Causal Ripples)"
    ])

    with view_tab1:
        col_ctrl1, col_ctrl2 = st.columns([1.5, 2.5])
        domain_options = ["همه حوزه‌ها"] + [meta["name_fa"] for meta in DOMAIN_CONFIG.values()]
        with col_ctrl1:
            selected_domain = st.selectbox("🎯 فیلتر نمایش حوزه:", domain_options, index=0)
        with col_ctrl2:
            st.markdown(
                "<div style='padding-top:25px; font-size:0.85rem; color:#94a3b8;'>"
                "💡 با نشانگر ماوس روی هر نود بروید تا کارت خلاصه ارزیابی و تعریف آن را به زبان فارسی مشاهده کنید."
                "</div>",
                unsafe_allow_html=True
            )

        # Plotly Network Map
        fig = build_resilience_network_figure(
            active_nodes=active_nodes,
            assessments_by_node=assessments_by_node,
            domain_filter=selected_domain
        )
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

    with view_tab2:
        render_domain_matrix_view(active_nodes, assessments_by_node)

    with view_tab3:
        render_cross_domain_ripples_view(active_nodes)
