"""
Streamlit Visualization Renderers
==================================
Premium-quality flowchart visualizations rendered inside Streamlit
using **Graphviz** (``st.graphviz_chart``).

- ``render_repo_tree``       → hierarchical directory-tree diagram
- ``render_call_graph``      → flowchart of function calls & class containment
- ``render_dependency_graph`` → flowchart of file-level import dependencies

No extra dependencies — Graphviz is bundled with Streamlit.
"""

import os
import html as html_mod
import streamlit as st
from typing import Any, Dict, List, Set


# ═══════════════════════════════════════════════════════════════════════
# Colour palettes (softer tones for a premium look)
# ═══════════════════════════════════════════════════════════════════════

_NODE_COLORS = {
    "function": {"fill": "#EDE7F6", "border": "#7C4DFF", "font": "#4A148C"},
    "class":    {"fill": "#FCE4EC", "border": "#FF4081", "font": "#880E4F"},
    "method":   {"fill": "#E0F2F1", "border": "#1DE9B6", "font": "#004D40"},
}

_LANG_COLORS = {
    "python":     {"fill": "#E3F2FD", "border": "#1976D2", "font": "#0D47A1"},
    "javascript": {"fill": "#FFFDE7", "border": "#FBC02D", "font": "#F57F17"},
    "typescript": {"fill": "#E8EAF6", "border": "#3F51B5", "font": "#1A237E"},
    "dart":       {"fill": "#E0F7FA", "border": "#00ACC1", "font": "#006064"},
    "other":      {"fill": "#F5F5F5", "border": "#9E9E9E", "font": "#424242"},
}

_EXT_COLORS = {
    ".py":   "#3572A5", ".js":  "#F7DF1E", ".jsx": "#61DAFB",
    ".ts":   "#3178C6", ".tsx": "#61DAFB", ".dart": "#00B4AB",
    ".java": "#B07219", ".kt":  "#A97BFF", ".go":   "#00ADD8",
    ".rs":   "#DEA584", ".rb":  "#CC342D", ".c":    "#555555",
    ".cpp":  "#F34B7D", ".h":   "#555555", ".swift": "#FA7343",
    ".md":   "#083FA1", ".json": "#292929", ".yaml": "#CB171E",
    ".yml":  "#CB171E", ".toml": "#9C4121",
}

_FILE_ICONS = {
    ".py": "🐍", ".js": "🟨", ".jsx": "⚛️", ".ts": "🔷",
    ".tsx": "⚛️", ".dart": "🎯", ".java": "☕", ".kt": "🟣",
    ".c": "⚙️", ".cpp": "⚙️", ".h": "⚙️", ".go": "🐹",
    ".rs": "🦀", ".rb": "💎", ".swift": "🍎", ".md": "📝",
    ".json": "📋", ".yaml": "📋", ".yml": "📋", ".toml": "📋",
}


# ═══════════════════════════════════════════════════════════════════════
# Shared helpers
# ═══════════════════════════════════════════════════════════════════════

def _esc(text: str) -> str:
    """Escape text for use inside Graphviz HTML labels."""
    return html_mod.escape(str(text), quote=True)


def _human_size(size_bytes: int) -> str:
    """Convert bytes to a human-readable string."""
    if size_bytes < 1024:
        return f"{size_bytes} B"
    elif size_bytes < 1024 * 1024:
        return f"{size_bytes / 1024:.1f} KB"
    return f"{size_bytes / (1024 * 1024):.1f} MB"


def _legend_html(items: List[dict]) -> str:
    """
    Build a styled HTML legend bar.

    Each item: {"color": "#hex", "label": "Text"}
    """
    badges = ""
    for it in items:
        badges += (
            f'<span style="display:inline-flex;align-items:center;gap:6px;'
            f'padding:4px 14px;border-radius:20px;'
            f'background:{it["color"]}18;border:1.5px solid {it["color"]};'
            f'font-size:0.82rem;font-weight:600;color:{it["color"]}">'
            f'{it.get("icon", "●")} {it["label"]}</span>'
        )
    return (
        f'<div style="display:flex;flex-wrap:wrap;gap:10px;'
        f'margin-bottom:16px">{badges}</div>'
    )


def _stats_bar(items: List[dict]) -> str:
    """
    Build a row of metric cards.

    Each item: {"label": "...", "value": "...", "icon": "..."}
    """
    cards = ""
    for it in items:
        cards += (
            f'<div style="background:linear-gradient(135deg,#1a1a2e,#16213e);'
            f'border-radius:12px;padding:14px 22px;min-width:120px;'
            f'text-align:center;flex:1">'
            f'<div style="font-size:1.5rem">{it["icon"]}</div>'
            f'<div style="font-size:1.6rem;font-weight:700;color:#e0e0e0;'
            f'margin:2px 0">{it["value"]}</div>'
            f'<div style="font-size:0.78rem;color:#9e9e9e;text-transform:'
            f'uppercase;letter-spacing:1px">{it["label"]}</div>'
            f'</div>'
        )
    return (
        f'<div style="display:flex;flex-wrap:wrap;gap:12px;'
        f'margin-bottom:20px">{cards}</div>'
    )


# ═══════════════════════════════════════════════════════════════════════
# 1. Repository Structure — Graphviz tree
# ═══════════════════════════════════════════════════════════════════════

def render_repo_tree(tree_data: Dict[str, Any]) -> None:
    """Render a hierarchical file-tree diagram using Graphviz DOT."""
    if not tree_data:
        st.info("No repository structure data available.")
        return

    # Collect stats
    file_count, dir_count, total_size = _count_tree(tree_data)
    lang_counts = _count_languages(tree_data)
    top_langs = sorted(lang_counts.items(), key=lambda x: -x[1])[:5]

    # Stats bar
    st.markdown(
        _stats_bar([
            {"icon": "📄", "value": str(file_count), "label": "Files"},
            {"icon": "📁", "value": str(dir_count), "label": "Directories"},
            {"icon": "💾", "value": _human_size(total_size), "label": "Total Size"},
            {"icon": "🔤", "value": str(len(lang_counts)),
             "label": "Languages"},
        ]),
        unsafe_allow_html=True,
    )

    # Language breakdown
    if top_langs:
        legend_items = []
        for lang, count in top_langs:
            color = _EXT_COLORS.get(lang, "#888")
            icon = _FILE_ICONS.get(lang, "📄")
            legend_items.append({
                "color": color,
                "icon": icon,
                "label": f"{lang} ({count})",
            })
        st.markdown(_legend_html(legend_items), unsafe_allow_html=True)

    # Build Graphviz DOT
    dot_lines = [
        'digraph RepoTree {',
        '  rankdir=TB;',
        '  bgcolor="transparent";',
        '  node [shape=none, margin="0"];',
        '  edge [color="#78909C", arrowsize=0.6, penwidth=1.2];',
    ]

    _node_counter = [0]

    def _add_node(node: Dict, parent_id: str | None, depth: int):
        _node_counter[0] += 1
        nid = f"n{_node_counter[0]}"

        if node["type"] == "directory":
            label = (
                f'<<TABLE BORDER="0" CELLBORDER="0" CELLSPACING="0">'
                f'<TR><TD><FONT POINT-SIZE="14">📁</FONT></TD>'
                f'<TD><FONT FACE="Inter,Helvetica" POINT-SIZE="11" '
                f'COLOR="#1565C0"><B> {_esc(node["name"])}</B></FONT></TD>'
                f'</TR></TABLE>>'
            )
            dot_lines.append(
                f'  {nid} [label={label}];'
            )
        else:
            ext = node.get("extension", "")
            color = _EXT_COLORS.get(ext, "#616161")
            icon = _FILE_ICONS.get(ext, "📄")
            size_str = _human_size(node.get("size_bytes", 0))
            label = (
                f'<<TABLE BORDER="0" CELLBORDER="0" CELLSPACING="0">'
                f'<TR><TD><FONT POINT-SIZE="12">{icon}</FONT></TD>'
                f'<TD><FONT FACE="Inter,Helvetica" POINT-SIZE="10" '
                f'COLOR="{color}"> {_esc(node["name"])}</FONT></TD>'
                f'<TD><FONT FACE="Inter,Helvetica" POINT-SIZE="8" '
                f'COLOR="#9E9E9E"> {size_str}</FONT></TD>'
                f'</TR></TABLE>>'
            )
            dot_lines.append(
                f'  {nid} [label={label}];'
            )

        if parent_id:
            dot_lines.append(f'  {parent_id} -> {nid};')

        if node["type"] == "directory" and depth < 4:
            children = node.get("children", [])
            # Sort: dirs first, then files
            dirs = sorted(
                [c for c in children if c["type"] == "directory"],
                key=lambda c: c["name"],
            )
            files = sorted(
                [c for c in children if c["type"] == "file"],
                key=lambda c: c["name"],
            )
            for child in dirs + files:
                _add_node(child, nid, depth + 1)

    _add_node(tree_data, None, 0)
    dot_lines.append("}")

    st.graphviz_chart("\n".join(dot_lines), use_container_width=True)

    if _count_tree(tree_data)[1] > 50:
        st.caption("📌 Large repository — tree truncated at depth 4 for readability.")


def _count_tree(node: Dict) -> tuple:
    """Return (file_count, dir_count, total_size)."""
    if node["type"] == "file":
        return (1, 0, node.get("size_bytes", 0))
    files, dirs, size = 0, 1, 0
    for c in node.get("children", []):
        f, d, s = _count_tree(c)
        files += f
        dirs += d
        size += s
    return (files, dirs, size)


def _count_languages(node: Dict) -> Dict[str, int]:
    """Count files by extension."""
    counts: Dict[str, int] = {}
    if node["type"] == "file":
        ext = node.get("extension", "")
        if ext:
            counts[ext] = counts.get(ext, 0) + 1
    else:
        for c in node.get("children", []):
            for ext, count in _count_languages(c).items():
                counts[ext] = counts.get(ext, 0) + count
    return counts


# ═══════════════════════════════════════════════════════════════════════
# 2. Call Graph — Graphviz flowchart
# ═══════════════════════════════════════════════════════════════════════

def render_call_graph(graph_data: Dict[str, List[Dict[str, Any]]]) -> None:
    """Render a function-level call graph as a Graphviz flowchart."""
    nodes_data = graph_data.get("nodes", [])
    edges_data = graph_data.get("edges", [])

    if not nodes_data:
        st.info("No call graph data — the repository may have no analysable source files.")
        return

    # Stats
    func_count = sum(1 for n in nodes_data if n.get("type") == "function")
    class_count = sum(1 for n in nodes_data if n.get("type") == "class")
    method_count = sum(1 for n in nodes_data if n.get("type") == "method")
    call_edges = sum(1 for e in edges_data if e.get("type") == "calls")

    st.markdown(
        _stats_bar([
            {"icon": "⚡", "value": str(func_count), "label": "Functions"},
            {"icon": "🏛️", "value": str(class_count), "label": "Classes"},
            {"icon": "🔧", "value": str(method_count), "label": "Methods"},
            {"icon": "→", "value": str(call_edges), "label": "Calls"},
        ]),
        unsafe_allow_html=True,
    )

    # Legend
    st.markdown(
        _legend_html([
            {"color": "#7C4DFF", "icon": "⚡", "label": "Function"},
            {"color": "#FF4081", "icon": "🏛️", "label": "Class"},
            {"color": "#1DE9B6", "icon": "🔧", "label": "Method"},
            {"color": "#78909C", "icon": "—→", "label": "Calls"},
            {"color": "#FF4081", "icon": "- -→", "label": "Contains"},
        ]),
        unsafe_allow_html=True,
    )

    # Build DOT
    dot_lines = [
        'digraph CallGraph {',
        '  rankdir=TB;',
        '  bgcolor="transparent";',
        '  newrank=true;',
        '  nodesep=0.8;',
        '  ranksep=1.0;',
        '  splines=ortho;',
    ]

    # Group nodes by file using subgraphs
    file_groups: Dict[str, List[Dict]] = {}
    for n in nodes_data:
        fname = n.get("file", "unknown")
        file_groups.setdefault(fname, []).append(n)

    for idx, (fname, nodes) in enumerate(sorted(file_groups.items())):
        dot_lines.append(f'  subgraph cluster_{idx} {{')
        dot_lines.append(f'    label=<<FONT FACE="Inter,Helvetica" POINT-SIZE="10" '
                         f'COLOR="#78909C">📄 {_esc(fname)}</FONT>>;')
        dot_lines.append(f'    style=dashed;')
        dot_lines.append(f'    color="#CFD8DC";')
        dot_lines.append(f'    bgcolor="#FAFAFA";')

        for n in nodes:
            ntype = n.get("type", "function")
            colors = _NODE_COLORS.get(ntype, _NODE_COLORS["function"])
            nid = _dot_id(n["id"])

            if ntype == "class":
                shape = "doubleoctagon"
            elif ntype == "method":
                shape = "ellipse"
            else:
                shape = "box"

            line_info = f':{n.get("line", "")}' if n.get("line") else ""

            label = (
                f'<<TABLE BORDER="0" CELLBORDER="0" CELLSPACING="2">'
                f'<TR><TD><FONT FACE="Inter,Helvetica" POINT-SIZE="11" '
                f'COLOR="{colors["font"]}"><B>{_esc(n["label"])}</B></FONT></TD></TR>'
                f'<TR><TD><FONT FACE="Inter,Helvetica" POINT-SIZE="8" '
                f'COLOR="#9E9E9E">{ntype}{line_info}</FONT></TD></TR>'
                f'</TABLE>>'
            )
            dot_lines.append(
                f'    {nid} [label={label}, shape={shape}, '
                f'style="filled,rounded", fillcolor="{colors["fill"]}", '
                f'color="{colors["border"]}", penwidth=1.5];'
            )

        dot_lines.append('  }')

    # Edges
    for e in edges_data:
        src = _dot_id(e["source"])
        tgt = _dot_id(e["target"])
        etype = e.get("type", "calls")
        if etype == "contains":
            dot_lines.append(
                f'  {src} -> {tgt} [style=dashed, color="#FF4081", '
                f'penwidth=1.2, arrowsize=0.7];'
            )
        else:
            dot_lines.append(
                f'  {src} -> {tgt} [color="#7C4DFF", '
                f'penwidth=1.5, arrowsize=0.8];'
            )

    dot_lines.append("}")

    st.graphviz_chart("\n".join(dot_lines), use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════
# 3. Dependency Graph — Graphviz flowchart
# ═══════════════════════════════════════════════════════════════════════

def render_dependency_graph(
    graph_data: Dict[str, List[Dict[str, Any]]],
) -> None:
    """Render a file-level dependency graph as a Graphviz flowchart."""
    nodes_data = graph_data.get("nodes", [])
    edges_data = graph_data.get("edges", [])

    if not nodes_data:
        st.info("No dependency graph data available.")
        return

    # Stats
    lang_dist: Dict[str, int] = {}
    for n in nodes_data:
        lang = n.get("language", "other")
        lang_dist[lang] = lang_dist.get(lang, 0) + 1

    st.markdown(
        _stats_bar([
            {"icon": "📦", "value": str(len(nodes_data)), "label": "Files"},
            {"icon": "🔗", "value": str(len(edges_data)),
             "label": "Dependencies"},
            {"icon": "🌍", "value": str(len(lang_dist)), "label": "Languages"},
        ]),
        unsafe_allow_html=True,
    )

    # Legend
    legend_items = []
    for lang in sorted(lang_dist.keys()):
        colors = _LANG_COLORS.get(lang, _LANG_COLORS["other"])
        legend_items.append({
            "color": colors["border"],
            "icon": "●",
            "label": f"{lang.capitalize()} ({lang_dist[lang]})",
        })
    st.markdown(_legend_html(legend_items), unsafe_allow_html=True)

    # Build DOT
    dot_lines = [
        'digraph DependencyGraph {',
        '  rankdir=LR;',
        '  bgcolor="transparent";',
        '  nodesep=0.6;',
        '  ranksep=1.5;',
        '  splines=polyline;',
    ]

    # Group by language using subgraphs
    lang_groups: Dict[str, List[Dict]] = {}
    for n in nodes_data:
        lang = n.get("language", "other")
        lang_groups.setdefault(lang, []).append(n)

    for idx, (lang, nodes) in enumerate(sorted(lang_groups.items())):
        colors = _LANG_COLORS.get(lang, _LANG_COLORS["other"])
        dot_lines.append(f'  subgraph cluster_{idx} {{')
        dot_lines.append(
            f'    label=<<FONT FACE="Inter,Helvetica" POINT-SIZE="11" '
            f'COLOR="{colors["font"]}"><B>{lang.upper()}</B></FONT>>;'
        )
        dot_lines.append(f'    style="rounded,filled";')
        dot_lines.append(f'    color="{colors["border"]}";')
        dot_lines.append(f'    fillcolor="{colors["fill"]}40";')

        for n in nodes:
            nid = _dot_id(n["id"])
            label = (
                f'<<TABLE BORDER="0" CELLBORDER="0" CELLSPACING="2">'
                f'<TR><TD><FONT FACE="Inter,Helvetica" POINT-SIZE="11" '
                f'COLOR="{colors["font"]}"><B>{_esc(n["label"])}</B></FONT>'
                f'</TD></TR>'
                f'<TR><TD><FONT FACE="Inter,Helvetica" POINT-SIZE="8" '
                f'COLOR="#9E9E9E">{_esc(n["id"])}</FONT></TD></TR>'
                f'</TABLE>>'
            )
            dot_lines.append(
                f'    {nid} [label={label}, shape=box, '
                f'style="filled,rounded", fillcolor="{colors["fill"]}", '
                f'color="{colors["border"]}", penwidth=1.5];'
            )

        dot_lines.append('  }')

    # Edges with import labels
    for e in edges_data:
        src = _dot_id(e["source"])
        tgt = _dot_id(e["target"])
        module = e.get("module", "")
        label_part = f', label=<<FONT POINT-SIZE="8" COLOR="#78909C">{_esc(module)}</FONT>>' if module else ""
        dot_lines.append(
            f'  {src} -> {tgt} [color="#546E7A", '
            f'penwidth=1.2, arrowsize=0.7{label_part}];'
        )

    dot_lines.append("}")

    st.graphviz_chart("\n".join(dot_lines), use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════
# DOT helper
# ═══════════════════════════════════════════════════════════════════════

def _dot_id(raw: str) -> str:
    """
    Convert an arbitrary string into a valid Graphviz node id.

    Wraps the string in double-quotes and escapes inner quotes.
    """
    safe = raw.replace('\\', '\\\\').replace('"', '\\"')
    return f'"{safe}"'
