"""Streamlit workspace for graph detection and dataset-specific evaluation."""

from __future__ import annotations

import csv
import io
from pathlib import Path
from typing import Any

from gnn_attack_detector.attacks.registry import LIVE_GENERATION_ATTACKS, TARGETED_ATTACKS, attack_spec
from gnn_attack_detector.platform_service import (
    collect_platform_status,
    detect_platform_graphs,
    generate_and_detect_platform_graphs,
    load_platform_accuracy_summary,
    load_platform_evaluation,
    save_uploaded_graph,
)


def _percent(value: Any) -> str:
    return "—" if value is None else f"{float(value) * 100:.2f}%"


def _dataset_name(identifier: str, status: dict[str, Any]) -> str:
    names = {item["id"]: item["name"] for item in status["profiles"]}
    name = names.get(identifier, str(identifier).removesuffix("-lcc"))
    return {"cora": "Cora", "citeseer": "Citeseer", "polblogs": "Polblogs", "pubmed": "Pubmed"}.get(name.lower(), name)


def _attack_name(identifier: str) -> str:
    if identifier == "clean":
        return "干净图"
    try:
        return attack_spec(identifier).display_name
    except ValueError:
        return identifier


def _style(st) -> None:
    st.markdown(
        """
        <style>
        :root {--gs-ink: #141326; --gs-sidebar: #211d42; --gs-blue: #6753e8; --gs-cyan: #2cc9da; --gs-plasma: #a879ff; --gs-violet: #7563ef; --gs-green: #168a72; --gs-amber: #c26a16;
            --gs-sky: #e9e6ff; --gs-surface: #f6f4fb; --gs-canvas: #f1f0f6; --gs-card: #ffffffd9; --gs-line: #ddd8ee; --gs-text: #27233f; --gs-muted: #706a86;}
        .stApp {background: radial-gradient(circle at 92% -8%, #e4dbff 0, transparent 26%),
            radial-gradient(circle at 2% 34%, #d9f9fa 0, transparent 20%),
            linear-gradient(145deg, #f5f3fa 0%, #efeff5 55%, #f9f8fc 100%); color: var(--gs-text);}
        .block-container {max-width: 1440px; padding: 3rem 3rem 4rem;}
        [data-testid="stSidebar"] {background: linear-gradient(180deg, var(--gs-sidebar) 0%, #2b2552 56%, #151629 100%);
            border-right: 1px solid #4b4475;}
        [data-testid="stSidebar"] > div:first-child {padding: 1.4rem .85rem 1.5rem;}
        [data-testid="stSidebar"] [role="radiogroup"] {gap: .35rem;}
        [data-testid="stSidebar"] [role="radiogroup"] > label {color: #b8b4d6; padding: .78rem .85rem;
            border-radius: .8rem; margin: .12rem 0; transition: background .2s ease, color .2s ease, box-shadow .2s ease;}
        [data-testid="stSidebar"] [role="radiogroup"] > label:hover {background: #ffffff12; color: #ffffff;}
        [data-testid="stSidebar"] [role="radiogroup"] > label:has(input:checked) {background: #ffffff18; color: #ffffff;
            box-shadow: inset 3px 0 0 var(--gs-cyan), 0 8px 18px #08071633;}
        [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] {color: #b8b4d6;}
        [data-testid="stSidebar"] hr {border-color: #4c4674; margin: 1.2rem 0;}
        .brand {font-size: 1.15rem; font-weight: 750; color: #ffffff; letter-spacing: -.02em; margin-bottom: .25rem;}
        .brand-mark {display: inline-flex; width: 1.85rem; height: 1.85rem; align-items: center; justify-content: center;
            background: linear-gradient(145deg, #bca4ff, #62e1e8); border: 1px solid #d8cbff; border-radius: .55rem; color: #17152f; margin-right: .45rem;
            font-weight: 800; font-size: .78rem; vertical-align: -.12rem;}
        .brand-sub {font-size: .68rem; color: #9d98c6; letter-spacing: .18em; margin: 0 0 1.45rem 2.35rem;}
        .sidebar-brand-panel {padding: .95rem .85rem 1rem; border: 1px solid #7770ae; border-radius: 1rem;
            background: linear-gradient(145deg, #ffffff16, #ffffff08); box-shadow: 0 10px 24px #08071636;}
        .sidebar-nav-label {color: #8f89bb; font-size: .65rem; font-weight: 800; letter-spacing: .16em; margin: 1.35rem .8rem .5rem;}
        .sidebar-status, .sidebar-status-chip {display: flex; align-items: center; gap: .45rem; font-size: .78rem; color: #e4e0ff;}
        .sidebar-status-dot {width: .45rem; height: .45rem; border-radius: 50%; background: #43dfba; box-shadow: 0 0 0 .22rem #43dfba2e;}
        .sidebar-metrics {display: grid; grid-template-columns: 1fr 1fr; gap: .5rem; margin-top: .8rem;}
        .sidebar-metric {border: 1px solid #7169a6; background: #ffffff12; border-radius: .7rem; padding: .55rem .65rem;}
        .sidebar-metric b {display: block; color: #8beef1; font-size: 1rem;}
        .sidebar-metric:nth-child(2) b {color: #c7adff;}
        .sidebar-metric span {color: #aaa5cf; font-size: .63rem;}
        .security-console {position: relative;}
        .hero {position: relative; background: linear-gradient(135deg, #211b3d 0%, #3b2d68 56%, #1b294d 100%);
            border: 1px solid #6756a1; border-radius: 1.25rem; padding: 2.25rem 2.5rem;
            color: #ffffff; display: flex; justify-content: space-between; align-items: center; margin-bottom: 1.25rem;
            gap: 2rem; overflow: hidden; box-shadow: 0 18px 38px #30215a3b;}
        .hero::after {content: ""; width: 18rem; height: 18rem; border: 1px solid #a879ff55; border-radius: 50%;
            position: absolute; right: -7rem; top: -8rem;}
        .hero-copy {position: relative; z-index: 1; max-width: 44rem;}
        .hero-label {font-size: .68rem; letter-spacing: .19em; color: #82e9ee; font-weight: 700;}
        .hero h1 {color: #ffffff; font-size: clamp(1.8rem, 3vw, 2.65rem); margin: .55rem 0 .75rem; padding: 0; letter-spacing: -.045em;}
        .hero p {color: #d2cbed; margin: 0; font-size: .94rem; line-height: 1.75; max-width: 39rem;}
        .hero-visual {position: relative; z-index: 1; flex-shrink: 0; display: flex; align-items: center; gap: .65rem;}
        .hero-visual svg {width: 12rem; opacity: .92;}
        .hero-signal {font-family: ui-monospace, SFMono-Regular, Consolas, monospace; font-size: .7rem; line-height: 1.7;
            color: #8beef1; border-left: 1px solid #8a79c9; padding-left: 1rem;}
        .status-rail {display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: .75rem; margin-bottom: 2rem;}
        .status-item {position: relative; overflow: hidden; background: linear-gradient(145deg, #fffaff, #f0effa); border: 1px solid #d9d2eb;
            border-radius: .9rem; padding: .9rem 1rem; box-shadow: 0 8px 22px #5b4b9717;}
        .panel-corner {position: absolute; top: .55rem; right: .6rem; width: .55rem; height: .55rem; border-top: 1px solid var(--gs-cyan); border-right: 1px solid var(--gs-cyan); opacity: .9;}
        .status-item-label {display: flex; align-items: center; gap: .5rem; color: var(--gs-muted); font-size: .74rem; margin-bottom: .35rem;}
        .status-item strong {color: var(--gs-text); font-size: 1rem; font-weight: 700;}
        .status-item small {display: block; color: #8391a5; font-size: .7rem; margin-top: .2rem;}
        .status-dot {width: .45rem; height: .45rem; border-radius: 50%; background: var(--gs-blue);}
        .status-dot.ready {background: var(--gs-green); box-shadow: 0 0 0 .22rem #05966920;}
        .module-header {display: flex; align-items: flex-end; justify-content: space-between; gap: 1rem;
            background: linear-gradient(100deg, #f2effa 0%, #fbfaff 72%); border: 1px solid #d8d0eb;
            border-radius: 1rem; padding: 1rem 1.15rem; margin: 1.75rem 0 1rem; box-shadow: 0 8px 20px #5b4b9712;}
        .module-header h2 {margin: 0; color: var(--gs-text); font-size: 1.25rem; letter-spacing: -.035em;}
        .module-header p {margin: .25rem 0 0; color: var(--gs-muted); font-size: .78rem;}
        .module-tag {color: #6953dd; font-size: .66rem; font-weight: 800; letter-spacing: .14em; white-space: nowrap;}
        .result-strip {display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: .7rem; margin: .75rem 0 1rem;}
        .result-state {position: relative; overflow: hidden; background: linear-gradient(145deg, #ffffff, #f2effa); border: 1px solid #d9d2eb;
            border-radius: .85rem; padding: .8rem .9rem; box-shadow: 0 7px 18px #5b4b9714;}
        .result-state span {display: block; color: var(--gs-muted); font-size: .72rem; margin-bottom: .35rem;}
        .result-state strong {display: block; color: var(--gs-text); font-size: 1.45rem; letter-spacing: -.04em;}
        .result-state.attacked {background: linear-gradient(145deg, #fff5f8, #fceaf1); border-color: #f2c6d3;}
        .result-state.attacked strong {color: #b4234d;}
        .result-state.clean {background: linear-gradient(145deg, #f0fffb, #e6f8f3); border-color: #bce9dc;}
        .result-state.clean strong {color: var(--gs-green);}
        .result-state.failed {background: linear-gradient(145deg, #fff9f0, #fff1da); border-color: #f1d8ac;}
        .result-state.failed strong {color: var(--gs-amber);}
        .accuracy-overview-chart {border: 1px solid #d5cceb; border-radius: .8rem; padding: .75rem; background: linear-gradient(145deg, #ffffff, #f3f0fb); box-shadow: 0 8px 20px #5b4b9710;}
        .flow-steps {display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: .6rem; margin: .75rem 0 1.25rem;}
        .flow-step {position: relative; overflow: hidden; border: 1px solid #d5cceb; border-radius: .85rem; padding: .85rem; background: linear-gradient(145deg, #ffffff, #f1effa); box-shadow: 0 7px 18px #5b4b9712;}
        .flow-step::before {content: ""; position: absolute; left: .85rem; right: .85rem; top: 0; height: 2px; background: linear-gradient(90deg, var(--gs-plasma), var(--gs-cyan));}
        .flow-step b {display: block; color: #6953dd; font-size: .68rem; letter-spacing: .12em; margin-bottom: .3rem;}
        .flow-step span {color: var(--gs-text); font-size: .84rem; font-weight: 650;}
        .defense-callout {background: linear-gradient(100deg, #eeeaff, #f6f3ff); border: 1px solid #d3c8f2; border-radius: .85rem; padding: .85rem 1rem; color: #514775; font-size: .82rem; line-height: 1.65;}
        .section-heading {display: flex; align-items: baseline; justify-content: space-between; margin: 2rem 0 .8rem;}
        .section-heading h2 {margin: 0; font-size: 1.1rem; letter-spacing: -.025em;}
        .section-heading span {color: var(--gs-muted); font-size: .75rem;}
        .page-kicker {font-size: .68rem; letter-spacing: .18em; color: #6953dd; font-weight: 750; margin-bottom: .45rem;}
        [data-testid="stMetric"], .metric-data-plate {position: relative; overflow: hidden; background: linear-gradient(145deg, #ffffff 0%, #f8f5ff 68%, #eefcff 100%);
            border: 1px solid #d8d0eb; border-radius: 1rem; padding: 1.1rem 1.2rem; min-height: 6.5rem; box-shadow: 0 9px 24px #5b4b9717;}
        [data-testid="stMetric"]::before, .metric-data-plate::before {content: ""; position: absolute; left: 1rem; right: 1rem; top: 0; height: 3px;
            background: linear-gradient(90deg, var(--gs-plasma), var(--gs-cyan)); border-radius: 0 0 4px 4px;}
        [data-testid="stMetric"]::after, .metric-data-plate::after {content: ""; position: absolute; right: -.9rem; bottom: -.9rem; width: 4rem; height: 4rem; border: 1px solid #a879ff45; border-radius: 50%;}
        [data-testid="stMetricLabel"] {color: #6f668b; font-size: .76rem;}
        [data-testid="stMetricValue"] {font-size: 1.7rem; font-weight: 760; color: #34285e; letter-spacing: -.04em;}
        [data-testid="stMetricDelta"] {font-size: .72rem;}
        [class*="st-key-entry-"] {background: linear-gradient(145deg, #ffffff, #f0edfc); border: 1px solid #d5cceb; border-radius: 1rem;
            box-shadow: 0 9px 24px #5b4b9717;}
        [class*="st-key-entry-"]:hover {border-color: #a879ff; box-shadow: 0 11px 26px #5b4b9728;}
        [class*="st-key-entry-"] > div {border-radius: .9rem; padding: .25rem;}
        .entry-number {color: var(--gs-blue); font-size: .68rem; font-weight: 750; letter-spacing: .14em;}
        [data-testid="stVerticalBlockBorderWrapper"] > div {border-radius: 1rem; border-color: #d5cceb; background: #ffffff80;}
        [data-testid="stBaseButton-primary"] {border-radius: .75rem; min-height: 2.65rem; background: linear-gradient(100deg, #6753e8, #2cc9da); border-color: #6753e8;
            box-shadow: 0 8px 18px #6753e834;}
        [data-testid="stBaseButton-primary"]:hover {background: linear-gradient(100deg, #5542c9, #1da7b7); border-color: #5542c9;}
        .stButton > button {border-radius: .75rem; min-height: 2.55rem; transition: background-color .18s ease, border-color .18s ease, color .18s ease, box-shadow .18s ease;}
        .stButton > button:hover {border-color: #a879ff; box-shadow: 0 5px 14px #6753e826;}
        [data-baseweb="tab-list"] {gap: .35rem; margin: 1.25rem 0; border-bottom: 1px solid #d5cceb;}
        [data-baseweb="tab"] {padding: .65rem 1rem; color: var(--gs-muted);}
        [aria-selected="true"][data-baseweb="tab"] {color: #6953dd; border-bottom-color: #6953dd;}
        [data-testid="stFileUploader"] {border: 1px dashed #a879ff; border-radius: .9rem; background: #f4f0ff;}
        [data-testid="stDataFrame"] {border: 1px solid #d5cceb; border-radius: .85rem; overflow: hidden; box-shadow: 0 7px 18px #5b4b9712;}
        [data-testid="stAlert"] {border-radius: .7rem; border-width: 1px;}
        .stCaption, [data-testid="stCaptionContainer"] {color: var(--gs-muted);}
        @media (prefers-reduced-motion: reduce) {*, *::before, *::after {animation-duration: .01ms !important;
            animation-iteration-count: 1 !important; transition-duration: .01ms !important; scroll-behavior: auto !important;}}
        @media (max-width: 900px) {.block-container {padding: 2rem 1.25rem 3rem;}.hero {padding: 1.7rem;}.hero-visual {display: none;}
            .status-rail, .flow-steps {grid-template-columns: repeat(2, minmax(0, 1fr));}}
        @media (max-width: 520px) {.status-rail, .flow-steps, .result-strip {grid-template-columns: 1fr;}.hero h1 {font-size: 1.7rem;}
            .module-header {align-items: flex-start; flex-direction: column;}}
        </style>
        """, unsafe_allow_html=True,
    )


def _page_heading(st, title: str, description: str) -> None:
    st.markdown('<div class="page-kicker">GRAPH SECURITY WORKSPACE</div>', unsafe_allow_html=True)
    st.title(title)
    st.caption(description)


def _status_rail(st, status: dict[str, Any]) -> None:
    ready = bool(status["ready_for_detection"])
    ready_count = sum(bool(item["ready"]) for item in status["profiles"])
    st.markdown(
        f"""<div class="status-rail" aria-label="系统状态">
        <div class="status-item"><span class="panel-corner" aria-hidden="true"></span><div class="status-item-label"><span class="status-dot {'ready' if ready else ''}"></span>检测服务</div>
        <strong>{'已就绪' if ready else '待准备'}</strong><small>共享检测器状态</small></div>
        <div class="status-item"><span class="panel-corner" aria-hidden="true"></span><div class="status-item-label"><span class="status-dot"></span>数据集就绪度</div>
        <strong>{ready_count}/{len(status['profiles'])}</strong><small>可直接进行评测</small></div>
        <div class="status-item"><span class="panel-corner" aria-hidden="true"></span><div class="status-item-label"><span class="status-dot"></span>支持攻击方法</div>
        <strong>{status['supported_attack_count']}</strong><small>已注册攻击协议</small></div>
        <div class="status-item"><span class="panel-corner" aria-hidden="true"></span><div class="status-item-label"><span class="status-dot"></span>扰动图语料</div>
        <strong>{status['attack_graph_count']:,}</strong><small>已保存完整图样本</small></div>
        </div>""",
        unsafe_allow_html=True,
    )


def _select_dataset(st, status: dict[str, Any], *, key: str, ready_only: bool = True) -> str | None:
    profiles = [item["id"] for item in status["profiles"] if item["ready"] or not ready_only]
    if not profiles:
        st.info("暂无可用数据集，请先完成数据准备。")
        return None
    return st.selectbox("数据集", profiles, format_func=lambda value: _dataset_name(value, status), key=key)


def _csv_bytes(rows: list[dict[str, Any]]) -> bytes:
    output = io.StringIO()
    if rows:
        writer = csv.DictWriter(output, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return output.getvalue().encode("utf-8-sig")


def _result_table(rows: list[dict[str, Any]], status: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        {
            "输入图": item.get("source_name", "—"),
            "数据集": _dataset_name(item.get("profile_id") or item.get("dataset", "—"), status),
            "检测结果": item.get("prediction_label", "检测失败"),
            "攻击概率": _percent(item.get("graph_attack_probability")),
            "节点数": item.get("scanned_nodes", 0),
            "边数": item.get("scanned_undirected_edges", 0),
        }
        for item in rows
    ]


def _show_detection_results(st, rows: list[dict[str, Any]], status: dict[str, Any], *, key: str) -> None:
    successful = [item for item in rows if item.get("status") == "ok"]
    attacked = sum(item.get("prediction") == "attacked" for item in successful)
    clean = len(successful) - attacked
    failed = len(rows) - len(successful)
    st.markdown(
        '<div class="module-header"><div><span class="module-tag">LIVE ANALYSIS</span>'
        '<h2>检测结果</h2><p>模型已完成对当前输入图的图级判断。</p></div>'
        f'<span class="module-tag">{len(rows)} FILES</span></div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        f'''<div class="result-strip">
        <div class="result-state attacked"><span>检测到攻击</span><strong>{attacked}</strong></div>
        <div class="result-state clean"><span>未检测到攻击</span><strong>{clean}</strong></div>
        <div class="result-state failed"><span>处理失败</span><strong>{failed}</strong></div>
        </div>''',
        unsafe_allow_html=True,
    )
    columns = st.columns(3)
    columns[0].metric("有效输入", len(successful))
    columns[1].metric("攻击占比", _percent(attacked / len(successful)) if successful else "—")
    columns[2].metric("输出文件", len(rows))
    table = _result_table(rows, status)
    st.dataframe(table, use_container_width=True, hide_index=True)
    st.caption("每行对应一张完整图。攻击概率是模型的判断分数，不是测试准确率。")
    st.download_button("导出检测结果 CSV", _csv_bytes(table), "detection-results.csv", "text/csv", key=f"{key}-export")
    for item in rows:
        if item.get("status") == "error":
            st.error(f"{item.get('source_name', '输入图')}：{item.get('error', '未知错误')}")


def _overview(st, status: dict[str, Any]) -> None:
    st.markdown('<div class="security-console">', unsafe_allow_html=True)
    st.markdown(
        """<div class="hero"><div class="hero-copy"><div class="hero-label">GRAPH SECURITY · 图安全工作台</div>
        <h1>图对抗攻击检测与防御平台</h1><p>从数据集准备到攻击检测，再到鲁棒防御评测，在一个清晰、可追溯的工作台中完成图数据安全分析。</p></div>
        <div class="hero-visual"><svg viewBox="0 0 200 125" aria-hidden="true"><g stroke="#72bfe8" stroke-width="1.5" fill="none">
        <path d="M20 70L60 22L110 48L167 20L183 86L125 108L70 94L20 70M60 22L70 94L110 48L183 86M110 48L125 108M167 20L125 108"/>
        </g><g fill="#c9f3ff" stroke="#2c83b3" stroke-width="5"><circle cx="20" cy="70" r="7"/>
        <circle cx="60" cy="22" r="8"/><circle cx="110" cy="48" r="11"/><circle cx="167" cy="20" r="6"/>
        <circle cx="183" cy="86" r="8"/><circle cx="125" cy="108" r="6"/><circle cx="70" cy="94" r="7"/></g></svg><div class="hero-signal">MODEL<br>READY<br>SECURE</div></div></div>""",
        unsafe_allow_html=True,
    )
    _status_rail(st, status)
    first, second, third, fourth = st.columns(4)
    first.metric("支持数据集", len(status["profiles"]))
    second.metric("攻击方法", status["supported_attack_count"])
    third.metric("已保存扰动图", f"{status['attack_graph_count']:,}")
    fourth.metric("检测服务", "可用" if status["ready_for_detection"] else "待准备")
    st.markdown('<div class="section-heading"><h2>工作台</h2><span>选择一个任务开始</span></div>', unsafe_allow_html=True)

    def navigate(page: str) -> None:
        st.session_state["platform-page"] = page

    entries = [
        ("01", "攻击检测", "在线生成或批量上传图数据，获取检测结果。"),
        ("02", "准确率评测", "按数据集查看准确率与攻击检测结果。"),
        ("03", "鲁棒防御模型", "选择鲁棒模型，评测输入图的节点分类表现。"),
    ]
    for column, (number, name, description) in zip(st.columns(3), entries):
        with column, st.container(border=True, key=f"entry-{number}"):
            st.markdown(f'<span class="entry-number">{number} / WORKSPACE</span>', unsafe_allow_html=True)
            st.markdown(f"### {name}")
            st.caption(description)
            st.button(f"进入{name} →", key=f"open-{number}", on_click=navigate, args=(name,), use_container_width=True)
    st.markdown('<div class="section-heading"><h2>数据集就绪度</h2><span>当前项目数据源</span></div>', unsafe_allow_html=True)
    st.dataframe(
        [{"数据集": item["name"], "节点数": item["expected_num_nodes"],
          "状态": "已准备" if item["ready"] else "待准备"} for item in status["profiles"]],
        use_container_width=True, hide_index=True,
    )
    st.markdown('</div>', unsafe_allow_html=True)


def _generation_tab(st, root: Path, status: dict[str, Any]) -> None:
    st.markdown('<div class="module-header"><div><span class="module-tag">01 / GENERATE</span><h2>在线生成并检测</h2><p>选择数据集和攻击协议，直接查看模型响应。</p></div></div>', unsafe_allow_html=True)
    with st.container(border=True):
        left, right = st.columns(2)
        with left:
            profile_id = _select_dataset(st, status, key="generate-dataset")
        with right:
            attack_name = st.selectbox("攻击方法", list(LIVE_GENERATION_ATTACKS), format_func=_attack_name, key="generate-attack")
        if profile_id is None:
            return
        left, right = st.columns(2)
        targeted = attack_name in TARGETED_ATTACKS
        with left:
            strength = st.selectbox(
                "攻击强度", [0, 5, 10, 15, 20] if targeted else [0.0, 0.05, 0.10, 0.15, 0.20],
                index=1, key=f"generate-strength-{'nodes' if targeted else 'edges'}",
                format_func=lambda value: "0 · 干净图" if value == 0 else (
                    f"{int(value)} 个目标节点" if targeted else f"{float(value) * 100:g}% 边扰动"
                ),
            )
        with right:
            count = st.number_input("图数量", min_value=1, max_value=10, value=1, disabled=strength == 0, key=f"generate-count-{'clean' if strength == 0 else 'attack'}")
        count = 1 if strength == 0 else int(count)
        st.caption("0 扰动直接检测一张干净图。在线生成结果单独保存，不计入训练、验证和测试集。")
        signature = (profile_id, attack_name, strength, count)
        if st.button("检测干净图" if strength == 0 else "生成并检测", type="primary", use_container_width=True, key="generate-submit"):
            st.session_state.pop("generated-results", None)
            try:
                with st.spinner("正在检测干净图……" if strength == 0 else "正在生成并检测，完成后自动显示结果……"):
                    rows = generate_and_detect_platform_graphs(
                        root / "configs" / "base.yaml", dataset_name=profile_id,
                        attack_name=attack_name, strength=strength, count=count, project_root=root,
                    )
                st.session_state["generated-results"] = (signature, rows)
            except Exception as exc:
                st.error(f"本次处理未完成：{exc}")
    saved = st.session_state.get("generated-results")
    if saved and saved[0] == signature:
        _show_detection_results(st, saved[1], status, key="generated")


def _upload_tab(st, root: Path, status: dict[str, Any]) -> None:
    st.markdown('<div class="module-header"><div><span class="module-tag">02 / INGEST</span><h2>上传图文件</h2><p>先绑定数据集，再提交完整图或 CSR 邻接矩阵。</p></div></div>', unsafe_allow_html=True)
    with st.container(border=True):
        profile_id = _select_dataset(st, status, key="upload-dataset")
        uploads = st.file_uploader("选择一张或多张图文件", type=["pt", "npz"], accept_multiple_files=True, key="detection-uploads")
        st.caption("请先选择图的来源数据集。完整 PT/NPZ 使用文件自身属性；只有邻接矩阵的 CSR-NPZ 使用所选数据集的可信节点属性。不同数据集请分批上传。")
        signature = (profile_id, tuple((item.name, item.file_id) for item in uploads))
        if st.button("开始批量检测", type="primary", use_container_width=True, key="upload-submit"):
            st.session_state.pop("upload-results", None)
            if not uploads:
                st.warning("请先选择文件。")
            else:
                rows = []
                with st.spinner("正在检测上传图……"):
                    for upload in uploads:
                        try:
                            path = save_uploaded_graph(root, upload.name, upload.getvalue())
                            row = detect_platform_graphs(
                                root / "configs" / "base.yaml",
                                [path],
                                project_root=root,
                                profile_id=profile_id,
                            )[0]
                        except Exception as exc:
                            row = {"status": "error", "prediction_label": "检测失败", "error": str(exc)}
                        row.update({"source_name": upload.name})
                        rows.append(row)
                st.session_state["upload-results"] = (signature, rows)
    saved = st.session_state.get("upload-results")
    if saved and saved[0] == signature:
        _show_detection_results(st, saved[1], status, key="uploaded")


def _detection(st, root: Path, status: dict[str, Any]) -> None:
    _page_heading(st, "攻击检测", "共享GCN检测完整图；在线生成和文件上传都需要先选择数据集来源。")
    st.markdown(
        '<div class="flow-steps"><div class="flow-step"><b>STEP 01</b><span>绑定数据集来源</span></div>'
        '<div class="flow-step"><b>STEP 02</b><span>生成或上传图文件</span></div>'
        '<div class="flow-step"><b>STEP 03</b><span>获得攻击概率</span></div></div>',
        unsafe_allow_html=True,
    )
    if not status["ready_for_detection"]:
        st.info("请先用 rebuild-detector 训练检测器；训练完成后会自动更新当前模型，各数据集的历史评测结果会分别保留。")
        return
    generated, uploaded = st.tabs(["在线生成", "上传图文件"])
    with generated:
        _generation_tab(st, root, status)
    with uploaded:
        _upload_tab(st, root, status)


def _evaluation(st, root: Path, status: dict[str, Any]) -> None:
    _page_heading(st, "准确率评测", "切换数据集，查看已保存测试图的实际检测结果。")
    profile_id = _select_dataset(st, status, key="evaluation-dataset", ready_only=False)
    if profile_id is None:
        return
    try:
        payload = load_platform_evaluation(
            root / status["evaluation_report"],
            prediction_path=root / status["prediction_report"] if status.get("prediction_report") else None,
            model_path=root / status["detector_model"] if status.get("detector_model") else None,
            profile_id=profile_id,
        )
    except (ValueError, OSError) as exc:
        st.warning(f"暂时无法读取评测结果：{exc}")
        return
    if not payload["available"]:
        st.info(payload.get("reason") or "尚无测试结果。")
        return
    report = payload["report"]
    if payload.get("historical"):
        st.warning("这是该数据集已保存的历史模型或候选模型评测，并非当前激活模型的结果。只有再次评测此数据集才更新此页。")
    if report.get("clean_control_warning"):
        st.warning("若未提供独立非攻击样本，干净图在各划分重复使用：以下仅为基准对照结果，不能作为真实场景泛化或正式准确率证明。")
    evaluated_at = report.get("evaluated_at_utc")
    model_hash = report.get("detector_model_hash")
    if isinstance(evaluated_at, str) and isinstance(model_hash, str):
        st.caption(f"评测时间（UTC）：{evaluated_at} · 检测器：{model_hash[:12]}")
    detail = report.get("profile_evaluations", {}).get(profile_id, {})
    metrics = detail.get("test_metrics") or report.get("metrics_by_profile", {}).get(profile_id)
    if not metrics:
        st.info(f"{_dataset_name(profile_id, status)} 尚无测试结果，请先完成该数据集的测试评测。")
        return
    matrix = metrics["confusion_matrix"]
    attacked_count = matrix["tp"] + matrix["fn"]
    clean_count = matrix["tn"] + matrix["fp"]
    first, second, third, fourth = st.columns(4)
    first.metric("测试准确率", _percent(metrics["accuracy"]))
    second.metric("攻击图检出率", _percent(metrics["recall"]) if attacked_count else "—")
    third.metric("攻击图数量", attacked_count)
    fourth.metric("干净图数量", clean_count)
    st.caption(
        f"仅统计 {_dataset_name(profile_id, status)}：共 {metrics['count']} 张图，"
        f"判断正确 {matrix['tp'] + matrix['tn']} 张。测试准确率 = 判断正确图数 ÷ 全部测试图数。"
    )
    summary = load_platform_accuracy_summary(
        root / status["evaluation_report"],
        profile_ids=tuple(item["id"] for item in status["profiles"]),
        prediction_path=(root / status["prediction_report"] if status.get("prediction_report") else None),
        model_path=(root / status["detector_model"] if status.get("detector_model") else None),
    )
    if summary["available"]:
        st.markdown(
            '<div class="module-header"><div><span class="module-tag">CROSS-DATASET VIEW</span>'
            '<h2>四个数据集综合结果</h2><p>按数据集比较检测器在测试集上的表现，橙色虚线表示 0.80 参考目标。</p></div></div>',
            unsafe_allow_html=True,
        )
        average, weighted, completed = st.columns(3)
        average.metric("数据集平均准确率", _percent(summary["macro_accuracy"]))
        weighted.metric("总体准确率（加权）", _percent(summary["weighted_accuracy"]))
        completed.metric("已评测数据集", f"{len(summary['evaluated_profiles'])}/{len(status['profiles'])}")
        accuracy_rows = []
        for candidate in summary["evaluated_profiles"]:
            candidate_payload = load_platform_evaluation(
                root / status["evaluation_report"],
                prediction_path=root / status["prediction_report"] if status.get("prediction_report") else None,
                model_path=root / status["detector_model"] if status.get("detector_model") else None,
                profile_id=candidate,
            )
            candidate_report = candidate_payload.get("report") or {}
            candidate_detail = candidate_report.get("profile_evaluations", {}).get(candidate, {})
            candidate_metrics = candidate_detail.get("test_metrics") or candidate_report.get("metrics_by_profile", {}).get(candidate)
            if candidate_metrics:
                accuracy_rows.append({
                    "数据集": _dataset_name(candidate, status),
                    "准确率": float(candidate_metrics["accuracy"]),
                    "测试图数": int(candidate_metrics.get("count", 0)),
                })
        if accuracy_rows:
            import altair as alt
            import pandas as pd

            st.markdown('<div class="accuracy-overview-chart">', unsafe_allow_html=True)
            accuracy_frame = pd.DataFrame(accuracy_rows)
            bars = alt.Chart(accuracy_frame).mark_bar(cornerRadiusEnd=5).encode(
                x=alt.X("准确率:Q", scale=alt.Scale(domain=[0, 1]), axis=alt.Axis(format=".0%", title=None)),
                y=alt.Y("数据集:N", sort="-x", title=None),
                color=alt.Color("数据集:N", scale=alt.Scale(range=["#22d3ee", "#60a5fa", "#818cf8", "#a78bfa"]), legend=None),
                tooltip=["数据集:N", alt.Tooltip("准确率:Q", format=".2%"), "测试图数:Q"],
            ).properties(height=max(150, 42 * len(accuracy_rows)))
            target = alt.Chart(pd.DataFrame({"目标": [0.8]})).mark_rule(color="#d97706", strokeDash=[6, 4], size=2).encode(x="目标:Q")
            st.altair_chart(bars + target, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)
        if summary["missing_profiles"]:
            missing = "、".join(_dataset_name(item, status) for item in summary["missing_profiles"])
            st.caption(f"尚未完成测试：{missing}。平均值只统计已经有测试报告的数据集，不把缺失数据按 0 分计算。")
        elif summary.get("historical"):
            st.caption("平均值包含与当前检测器版本不同的历史数据集报告，仅用于查看历史整体表现。")
    if clean_count:
        st.caption(f"干净图判断正确 {matrix['tn']}/{clean_count} 张；干净图为该数据集已有的参考图。")
    if not detail:
        st.info("旧报告缺少分数据集的攻击明细。运行 evaluate-detector 可更新报告，无需重新生成图或训练模型。")
        return
    by_attack = detail["attacked_complete_graph_recall"]["by_attack"]
    table = [
        {"攻击方法": _attack_name(name), "测试图数量": values["total"],
         "识别正确": values["detected"], "识别错误": values["total"] - values["detected"],
         "攻击图检出率": _percent(values["recall"])}
        for name, values in by_attack.items()
    ]
    st.markdown(
        '<div class="module-header"><div><span class="module-tag">ATTACK BREAKDOWN</span>'
        '<h2>各攻击方法的测试表现</h2><p>识别率按当前数据集、当前测试记录计算。</p></div></div>',
        unsafe_allow_html=True,
    )
    st.caption("攻击图检出率 = 正确识别为被攻击的图数 ÷ 该方法的测试图数。数量按实际测试记录显示。")
    if not table:
        st.info("该数据集尚无攻击图测试结果。")
        return
    details, chart = st.tabs(["测试明细", "检出率对比"])
    with details:
        st.dataframe(table, use_container_width=True, hide_index=True)
        st.download_button("导出当前数据集结果 CSV", _csv_bytes(table), f"{profile_id}-evaluation.csv", "text/csv", key="evaluation-export")
    with chart:
        import altair as alt
        import pandas as pd

        values = pd.DataFrame([{"攻击方法": _attack_name(name), "检出率": item["recall"], "测试图": item["total"]} for name, item in by_attack.items()])
        plot = alt.Chart(values).mark_bar(color="#247cba", cornerRadiusEnd=4).encode(
            x=alt.X("检出率:Q", scale=alt.Scale(domain=[0, 1]), axis=alt.Axis(format=".0%")),
            y=alt.Y("攻击方法:N", sort=None, title=None),
            tooltip=["攻击方法:N", alt.Tooltip("检出率:Q", format=".2%"), "测试图:Q"],
        ).properties(height=max(180, 28 * len(values)))
        st.altair_chart(plot, use_container_width=True)


def _defenses(st, root: Path, status: dict[str, Any]) -> None:
    _page_heading(st, "鲁棒防御模型", "选择鲁棒模型，查看输入图的节点分类表现。")
    st.markdown(
        '<div class="defense-callout"><strong>独立评测通道</strong>　防御评测与攻击检测相互独立。每张输入图都会使用 DeepRobust 原始划分重新训练所选模型，临时参数不会覆盖共享攻击检测器。</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="module-header"><div><span class="module-tag">ROBUSTNESS LAB</span>'
        '<h2>鲁棒防御模型实验台</h2><p>选择数据集、模型和输入图，比较防御前后的节点分类表现。</p></div></div>',
        unsafe_allow_html=True,
    )
    display = {item["id"]: item["display_name"] for item in status["defenses"]}
    modes = {item["id"]: item["execution_mode"] for item in status["defenses"]}
    mode_labels = {
        "input_fit": "每张输入图重新训练",
        "offline": "离线结构优化（很慢）",
    }
    with st.container(border=True):
        left, right = st.columns(2)
        with left:
            profile_id = _select_dataset(st, status, key="defense-dataset")
        with right:
            selected = st.selectbox(
                "鲁棒模型",
                [item["id"] for item in status["defenses"]],
                format_func=lambda value: display.get(value, value),
                key="defense-model",
            )
        profile_row = next(
            (item for item in status["profiles"] if item["id"] == profile_id), {}
        )
        defense_state = profile_row.get("defenses", {}).get(selected, {})
        ready = bool(defense_state.get("ready"))
        st.caption(f"运行方式：{mode_labels.get(modes.get(selected), modes.get(selected, '—'))}")
        if not ready:
            reason = defense_state.get("reason")
            if reason:
                st.warning(reason)
            else:
                st.warning(f"{display.get(selected, selected)} 无法在 {profile_id} 上运行；请先准备该数据集的干净档案。")
        upload = st.file_uploader("选择要进行防御评测的图", type=["pt", "npz"], key="defense-upload")
        if st.button("训练并运行防御评测", type="primary", use_container_width=True, disabled=profile_id is None or not ready):
            if upload is None:
                st.warning("请先选择图文件。")
            else:
                from gnn_attack_detector.defense_service import evaluate_robust_probe

                try:
                    path = save_uploaded_graph(root, upload.name, upload.getvalue())
                    with st.spinner("正在使用 DeepRobust 划分重新训练鲁棒模型并评测……"):
                        result = evaluate_robust_probe(root / "configs" / "base.yaml", path, project_root=root, profile_id=profile_id, defense_name=selected)
                    first, second, third = st.columns(3)
                    first.metric("干净图节点准确率", _percent(result["clean_accuracy"]))
                    second.metric("输入图节点准确率", _percent(result["input_graph_accuracy"]))
                    third.metric("准确率变化", _percent(-result["accuracy_drop"]))
                    split_sizes = result["split_sizes"]
                    st.caption(
                        "节点划分来自 DeepRobust："
                        f"训练 {split_sizes['train']} · 验证 {split_sizes['val']} · "
                        f"测试 {split_sizes['test']}"
                    )
                except Exception as exc:
                    st.error(f"防御评测未完成：{exc}")
    with st.expander("防御模型列表"):
        profile_states = profile_row.get("defenses", {})
        st.dataframe(
            [
                {
                    "模型": item["display_name"],
                    "运行方式": mode_labels.get(item["execution_mode"], item["execution_mode"]),
                    "当前数据集": "可运行" if profile_states.get(item["id"], {}).get("ready") else "待准备/受限",
                }
                for item in status["defenses"]
            ],
            use_container_width=True, hide_index=True,
        )


def render_app(project_root: str | Path) -> None:
    import streamlit as st

    root = Path(project_root).resolve()
    st.set_page_config(page_title="图安全 · 攻击检测与防御平台", page_icon="◈", layout="wide")
    _style(st)
    try:
        status = collect_platform_status(root)
    except (ValueError, OSError) as exc:
        st.error(f"无法加载平台配置：{exc}")
        return
    with st.sidebar:
        st.markdown('<div class="sidebar-brand-panel"><div class="brand"><span class="brand-mark">GS</span>图安全平台</div><div class="brand-sub">GRAPH SECURITY</div></div>', unsafe_allow_html=True)
        st.markdown('<div class="sidebar-nav-label">WORKSPACE</div>', unsafe_allow_html=True)
        page = st.radio("工作台导航", ["平台概览", "攻击检测", "准确率评测", "鲁棒防御模型"], key="platform-page", label_visibility="collapsed")
        st.divider()
        st.markdown('<div class="sidebar-nav-label">SYSTEM STATUS</div>', unsafe_allow_html=True)
        st.markdown(
            f'<div class="sidebar-status-chip sidebar-status"><span class="sidebar-status-dot"></span>检测服务 · {"已就绪" if status["ready_for_detection"] else "待准备"}</div>',
            unsafe_allow_html=True,
        )
        st.markdown(
            f'<div class="sidebar-metrics"><div class="sidebar-metric"><b>{len(status["profiles"])}</b><span>数据集</span></div>'
            f'<div class="sidebar-metric"><b>{status["supported_attack_count"]}</b><span>攻击协议</span></div></div>',
            unsafe_allow_html=True,
        )
        st.caption("浅蓝模式 · 运行状态实时读取")
    if page == "平台概览":
        _overview(st, status)
    elif page == "攻击检测":
        _detection(st, root, status)
    elif page == "准确率评测":
        _evaluation(st, root, status)
    else:
        _defenses(st, root, status)
