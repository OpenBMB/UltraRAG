"""UltraRAG CLI - Beautiful command-line interface."""

import importlib.metadata
import platform
import sys
from typing import Optional

from rich.console import Console, Group
from rich.panel import Panel
from rich.style import Style
from rich.table import Table
from rich.text import Text

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Theme Configuration
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

GRADIENT_COLORS = [
    (255, 255, 200),  
    (255, 215, 0),   
    (255, 165, 0),   
    (255, 140, 0),  
    (255, 69, 0),   
    (200, 40, 0),
]

# Logo uses solid orange (no gradient)
LOGO_COLOR = "#f97316"

# Theme styles
STYLES = {
    "primary": Style(color="#f59e0b", bold=True),        # Amber
    "secondary": Style(color="#a38b78"),                  # Warm gray
    "success": Style(color="#fbbf24", bold=True),        # Gold
    "warning": Style(color="#f97316"),                    # Orange
    "error": Style(color="#ef4444", bold=True),          # Red
    "info": Style(color="#fb923c"),                       # Soft orange
    "muted": Style(color="#8c7a6b"),                      # Warm muted
    "accent": Style(color="#f97316", bold=True),         # Accent orange
    "highlight": Style(color="#fde68a", bold=True),      # Warm highlight
    "link": Style(color="#fb923c", underline=True),      # Orange link
}

# ASCII art logo - pixel block style (UltraRAG)
ULTRARAG_LOGO = r"""
███   ███ ███    █████████ ████████    █████   ████████    █████    ███████
███   ███ ███       ███    ███   ███  ███ ███  ███   ███  ███ ███  ███   ███
███   ███ ███       ███    ███   ███ ███   ███ ███   ███ ███   ███ ███
███   ███ ███       ███    ████████  █████████ ████████  █████████ ███  ████
███   ███ ███       ███    ███  ███  ███   ███ ███  ███  ███   ███ ███   ███
███   ███ ███       ███    ███   ███ ███   ███ ███   ███ ███   ███ ███   ███
 ███████  █████████ ███    ███   ███ ███   ███ ███   ███ ███   ███  ███████
""".strip("\n")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Utility Functions
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


def get_version_safe(pkgname: str) -> str:
    """Get package version safely, returning placeholder if not installed.

    Args:
        pkgname: Package name to get version for

    Returns:
        Package version string or "not installed" if unavailable
    """
    try:
        return importlib.metadata.version(pkgname)
    except Exception:
        return "not installed"


def interpolate_color(color1: tuple, color2: tuple, factor: float) -> str:
    """Interpolate between two RGB colors.

    Args:
        color1: First RGB color tuple
        color2: Second RGB color tuple
        factor: Interpolation factor (0-1)

    Returns:
        Hex color string
    """
    r = int(color1[0] * (1 - factor) + color2[0] * factor)
    g = int(color1[1] * (1 - factor) + color2[1] * factor)
    b = int(color1[2] * (1 - factor) + color2[2] * factor)
    return f"#{r:02x}{g:02x}{b:02x}"


def get_gradient_logo(use_large: bool = True) -> Text:
    lines = ULTRARAG_LOGO.split("\n")
    final_text = Text()

    for i, line in enumerate(lines):
        if i < len(GRADIENT_COLORS):
            r, g, b = GRADIENT_COLORS[i]
            color_hex = f"#{r:02x}{g:02x}{b:02x}"
        else:
            r, g, b = GRADIENT_COLORS[-1]
            color_hex = f"#{r:02x}{g:02x}{b:02x}"
            
        final_text.append(line + "\n", style=f"{color_hex} bold")

    return final_text


def get_gradient_text(text: str, bold: bool = True) -> Text:
    """Create gradient text using the current theme colors.

    Args:
        text: Text content
        bold: Whether to render text as bold

    Returns:
        Rich Text object with gradient-colored text
    """
    final_text = Text()
    if not text:
        return final_text

    length = len(text)
    for i, char in enumerate(text):
        position = i / max(length - 1, 1)
        color_idx = position * (len(GRADIENT_COLORS) - 1)
        idx_1 = int(color_idx)
        idx_2 = min(idx_1 + 1, len(GRADIENT_COLORS) - 1)
        factor = color_idx - idx_1

        color = interpolate_color(GRADIENT_COLORS[idx_1], GRADIENT_COLORS[idx_2], factor)
        style = f"{color} bold" if bold else color
        final_text.append(char, style=style)

    return final_text


def get_styled_text(text: str, style_name: str) -> Text:
    """Create styled text using predefined theme styles.

    Args:
        text: Text content
        style_name: Name of style from STYLES dict

    Returns:
        Rich Text object with applied style
    """
    return Text(text, style=STYLES.get(style_name, STYLES["secondary"]))


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Banner Components
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


def create_info_table(
    pipeline_name: str,
    doc_url: str = "https://ultrarag.openbmb.cn/",
    show_system_info: bool = True,
) -> Table:
    """Create information table with pipeline and version details.

    Args:
        pipeline_name: Name of the pipeline
        doc_url: Documentation URL
        show_system_info: Whether to show system information

    Returns:
        Rich Table with formatted information
    """
    table = Table.grid(padding=(0, 2))
    table.add_column(style=STYLES["accent"], justify="right", width=3)
    table.add_column(style=STYLES["secondary"], justify="left", min_width=14)
    table.add_column(justify="left")

    # Pipeline info
    table.add_row("▶", "Pipeline", Text(pipeline_name, style=STYLES["primary"]))
    table.add_row("", "", "")

    # Version info section
    ultrarag_ver = get_version_safe("ultrarag")
    fastmcp_ver = get_version_safe("fastmcp")
    mcp_ver = get_version_safe("mcp")

    table.add_row("◆", "UltraRAG", Text(f"v{ultrarag_ver}", style=STYLES["success"]))
    table.add_row("◇", "FastMCP", Text(f"v{fastmcp_ver}", style=STYLES["muted"]))
    table.add_row("◇", "MCP", Text(f"v{mcp_ver}", style=STYLES["muted"]))
    table.add_row("", "", "")

    # Documentation link
    table.add_row("➢", "Docs", Text(doc_url, style=STYLES["link"]))

    # System info (optional)
    if show_system_info:
        table.add_row("", "", "")
        py_version = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
        system_info = f"{platform.system()} {platform.machine()}"
        table.add_row("❖", "Python", Text(py_version, style=STYLES["muted"]))
        table.add_row("▤", "System", Text(system_info, style=STYLES["muted"]))

    return table


def create_status_bar(status: str = "ready", message: str = "") -> Text:
    """Create a status bar with icon and message.

    Args:
        status: Status type (ready, running, success, error, warning)
        message: Optional status message

    Returns:
        Rich Text object with status indicator
    """
    status_icons = {
        "ready": ("●", STYLES["success"]),
        "running": ("◐", STYLES["info"]),
        "success": ("✓", STYLES["success"]),
        "error": ("✗", STYLES["error"]),
        "warning": ("⚠", STYLES["warning"]),
        "info": ("ℹ", STYLES["info"]),
    }

    icon, style = status_icons.get(status, ("●", STYLES["muted"]))
    text = Text()
    text.append(f" {icon} ", style=style)
    if message:
        text.append(message, style=STYLES["secondary"])
    return text


def make_server_banner(
    pipeline_name: str,
    show_logo: bool = True,
    doc_url: str = "https://ultrarag.openbmb.cn/",
    compact: bool = False,
) -> Panel:
    """Create a formatted banner panel for UltraRAG server.

    Args:
        pipeline_name: Name of the pipeline
        show_logo: Whether to display the gradient logo (default: True)
        doc_url: Documentation URL (default: "https://ultrarag.openbmb.cn/")
        compact: Whether to use compact layout (default: False)

    Returns:
        Rich Panel object containing logo and server information
    """
    elements = []

    # Add gradient logo
    if show_logo:
        logo_text = get_gradient_logo(use_large=not compact)
        elements.append(logo_text)
        elements.append("")

    # Add tagline

    # Add info table
    info_table = create_info_table(
        pipeline_name=pipeline_name,
        doc_url=doc_url,
        show_system_info=not compact,
    )
    elements.append(info_table)

    # Create panel with styled border
    ultrarag_ver = get_version_safe("ultrarag")
    title = get_gradient_text(" UltraRAG ")
    title.append(f"v{ultrarag_ver} ", style=STYLES["muted"])

    return Panel(
        Group(*elements),
        title=title,
        title_align="left",
        subtitle=Text(" Press Ctrl+C to exit ", style=STYLES["muted"]),
        subtitle_align="right",
        border_style=Style(color="#9a3412"),  # Warm orange border
        padding=(1, 3),
        expand=False,
    )


def make_welcome_banner() -> Panel:
    """Create a welcome banner for CLI startup.

    Returns:
        Rich Panel with welcome message
    """
    logo_text = get_gradient_logo(use_large=True)

    welcome_text = Text()
    welcome_text.append("\n")
    welcome_text.append("  欢迎使用 ", style=STYLES["secondary"])
    welcome_text.append(get_gradient_text("UltraRAG"))
    welcome_text.append(" - 新一代 RAG 开发框架\n", style=STYLES["secondary"])
    welcome_text.append("\n")
    welcome_text.append("  使用 ", style=STYLES["muted"])
    welcome_text.append("ultrarag --help", style=STYLES["highlight"])
    welcome_text.append(" 查看所有可用命令\n", style=STYLES["muted"])

    return Panel(
        Group(logo_text, welcome_text),
        border_style=Style(color="#9a3412"),
        padding=(1, 2),
        expand=False,
    )


def make_command_help_panel(
    command: str,
    description: str,
    usage: str,
    options: list,
) -> Panel:
    """Create a help panel for a specific command.

    Args:
        command: Command name
        description: Command description
        usage: Usage example
        options: List of (option, description) tuples

    Returns:
        Rich Panel with command help
    """
    content = Text()

    # Command name and description
    content.append(f"  {command}\n", style=STYLES["primary"])
    content.append(f"  {description}\n\n", style=STYLES["secondary"])

    # Usage
    content.append("  Usage: ", style=STYLES["muted"])
    content.append(f"{usage}\n\n", style=STYLES["highlight"])

    # Options
    if options:
        content.append("  Options:\n", style=STYLES["muted"])
        for opt, desc in options:
            content.append(f"    {opt:<20}", style=STYLES["info"])
            content.append(f"{desc}\n", style=STYLES["secondary"])

    return Panel(
        content,
        title=Text(" Command Help ", style=STYLES["muted"]),
        border_style=Style(color="#7c2d12"),
        padding=(0, 1),
        expand=False,
    )


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Progress & Status Indicators
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


def create_spinner_frames() -> list:
    """Create spinner animation frames.

    Returns:
        List of spinner frame characters
    """
    return ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]


def create_progress_bar(
    current: int,
    total: int,
    width: int = 30,
    filled_char: str = "█",
    empty_char: str = "░",
) -> Text:
    """Create a styled progress bar.

    Args:
        current: Current progress value
        total: Total value
        width: Width of progress bar in characters
        filled_char: Character for filled portion
        empty_char: Character for empty portion

    Returns:
        Rich Text object with progress bar
    """
    if total <= 0:
        percentage = 0
    else:
        percentage = min(current / total, 1.0)

    filled_width = int(width * percentage)
    empty_width = width - filled_width

    bar = Text()
    bar.append(" ")

    # Gradient fill
    for i in range(filled_width):
        pos = i / max(width, 1)
        color_idx = pos * (len(GRADIENT_COLORS) - 1)
        idx_1 = int(color_idx)
        idx_2 = min(idx_1 + 1, len(GRADIENT_COLORS) - 1)
        factor = color_idx - idx_1
        color = interpolate_color(GRADIENT_COLORS[idx_1], GRADIENT_COLORS[idx_2], factor)
        bar.append(filled_char, style=color)

    bar.append(empty_char * empty_width, style=STYLES["muted"])
    bar.append(f" {percentage * 100:.0f}%", style=STYLES["secondary"])

    return bar


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Console Output Functions
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


def log_server_banner(pipeline_name: str) -> None:
    """Print server banner to stderr console.

    Args:
        pipeline_name: Name of the pipeline to display in banner
    """
    console = Console(stderr=True)
    panel = make_server_banner(pipeline_name)
    console.print()
    console.print(panel)
    console.print()


def log_message(
    message: str,
    level: str = "info",
    console: Optional[Console] = None,
) -> None:
    """Print a styled log message.

    Args:
        message: Message to print
        level: Log level (info, success, warning, error)
        console: Optional console instance
    """
    if console is None:
        console = Console(stderr=True)

    icons = {
        "info": ("ℹ", STYLES["info"]),
        "success": ("✓", STYLES["success"]),
        "warning": ("⚠", STYLES["warning"]),
        "error": ("✗", STYLES["error"]),
        "debug": ("◆", STYLES["muted"]),
    }

    icon, style = icons.get(level, ("●", STYLES["secondary"]))
    text = Text()
    text.append(f" {icon} ", style=style)
    text.append(message, style=STYLES["secondary"])
    console.print(text)


def log_step(
    step_number: int,
    total_steps: int,
    description: str,
    status: str = "running",
    console: Optional[Console] = None,
) -> None:
    """Print a pipeline step indicator.

    Args:
        step_number: Current step number
        total_steps: Total number of steps
        description: Step description
        status: Step status (running, success, error)
        console: Optional console instance
    """
    if console is None:
        console = Console(stderr=True)

    status_styles = {
        "running": ("◐", STYLES["info"]),
        "success": ("✓", STYLES["success"]),
        "error": ("✗", STYLES["error"]),
        "pending": ("○", STYLES["muted"]),
    }

    icon, style = status_styles.get(status, ("●", STYLES["secondary"]))

    text = Text()
    text.append(f" {icon} ", style=style)
    text.append(f"[{step_number}/{total_steps}] ", style=STYLES["muted"])
    text.append(description, style=STYLES["secondary"])

    console.print(text)


def print_divider(
    title: str = "",
    style: str = "muted",
    console: Optional[Console] = None,
) -> None:
    """Print a styled divider line.

    Args:
        title: Optional title in the middle of divider
        style: Style name from STYLES dict
        console: Optional console instance
    """
    if console is None:
        console = Console(stderr=True)

    line_style = STYLES.get(style, STYLES["muted"])

    if title:
        console.rule(title, style=line_style)
    else:
        console.print("─" * 60, style=line_style)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Exports
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

__all__ = [
    # Banner functions
    "make_server_banner",
    "make_welcome_banner",
    "make_command_help_panel",
    "log_server_banner",
    # Styling
    "get_gradient_logo",
    "get_styled_text",
    "STYLES",
    "GRADIENT_COLORS",
    # Progress & Status
    "create_status_bar",
    "create_progress_bar",
    "create_spinner_frames",
    # Logging
    "log_message",
    "log_step",
    "print_divider",
    # Utilities
    "get_version_safe",
    "create_info_table",
]
