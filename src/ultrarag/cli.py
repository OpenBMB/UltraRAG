import importlib.metadata
from rich.console import Console, Group
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

# ULTRARAG_LOGO = r"""
#    _ __ ___ __  ______             ____  ___   ______   _____    ____ 
#   _ __ ___ / / / / / /__________ _/ __ \/   | / ____/  |__  /   / __ \
#  _ __ ___ / / / / / __/ ___/ __ `/ /_/ / /| |/ / __     /_ <   / / / /
# _ __ ___ / /_/ / / /_/ /  / /_/ / _, _/ ___ / /_/ /   ___/ /  / /_/ / 
#  _ __ ___\____/_/\__/_/   \__,_/_/ |_/_/  |_\____/   /____(_) \____/  
                                                           
# """.lstrip(
#     "\n"
# )

ULTRARAG_LOGO = r"""
██ ██ ██   ██   ██ ██   ████  ██████   ████   ██████
██ ██ ██ ██████ ████      ██  ██  ██  ██ ██  ██     
██ ██ ██   ██   ██     █████  ██████  ██████ ██ ███ 
██ ██ ██   ██   ██    ██ ██   ██ ██   ██ ██  ██  ██ 
 ████ ██   ████ ██     █████  ██  ██  ██ ██   ██████
""".strip("\n")

def get_version_safe(pkgname: str) -> str:
    try:
        return importlib.metadata.version(pkgname)
    except Exception:
        return "<not installed>"
    
def get_gradient_logo() -> Text:
    lines = ULTRARAG_LOGO.split("\n")
    colors = [
        (66, 133, 244),  # Blue
        (138, 180, 248), # Lighter Blue
        (217, 169, 255), # Purple
        (244, 143, 177), # Pink
    ]
    
    final_text = Text()
    
    for line in lines:
        length = len(line)
        if length == 0:
            final_text.append("\n")
            continue
            
        for i, char in enumerate(line):
            position = i / max(length, 1)
            color_idx = position * (len(colors) - 1)
            idx_1 = int(color_idx)
            idx_2 = min(idx_1 + 1, len(colors) - 1)
            factor = color_idx - idx_1
            
            r = int(colors[idx_1][0] * (1 - factor) + colors[idx_2][0] * factor)
            g = int(colors[idx_1][1] * (1 - factor) + colors[idx_2][1] * factor)
            b = int(colors[idx_1][2] * (1 - factor) + colors[idx_2][2] * factor)
            
            final_text.append(char, style=f"#{r:02x}{g:02x}{b:02x}")
        final_text.append("\n")
        
    return final_text


def make_server_banner(
    pipeline_name: str,
    show_logo: bool = True,
    doc_url: str = "https://ultrarag.openbmb.cn/",
) -> Panel:
    logo_text = get_gradient_logo() if show_logo else ""
    info_table = Table.grid(padding=(0, 1))
    info_table.add_column(style="bold", justify="center")
    info_table.add_column(style="bold cyan", justify="left")
    info_table.add_column(style="white", justify="left")
    info_table.add_row("🖥️", "Pipeline name:", pipeline_name)
    info_table.add_row("", "", "")
    info_table.add_row("📚", "Docs:", doc_url)
    info_table.add_row("", "", "")
    info_table.add_row(
        "🏎️", "FastMCP version:", Text(get_version_safe("fastmcp"), style="dim white")
    )
    info_table.add_row(
        "🤝", "MCP version:", Text(get_version_safe("mcp"), style="dim white")
    )
    return Panel(
        Group(logo_text, "", info_table),
        title="UltraRAG v3",
        title_align="left",
        border_style="dim",
        padding=(1, 4),
        expand=False,
    )


def log_server_banner(pipeline_name: str):
    console = Console(stderr=True)
    panel = make_server_banner(pipeline_name)
    console.print(Group("\n", panel, "\n"))
