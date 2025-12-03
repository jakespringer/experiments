#!/usr/bin/env python3
"""
╔═══════════════════════════════════════════════════════════════════════════════╗
║                           JSONL Viewer                                        ║
║                   Beautiful CLI for exploring JSONL files                     ║
╚═══════════════════════════════════════════════════════════════════════════════╝

A gorgeous terminal-based viewer for JSONL (JSON Lines) files with:
  • Auto-calculated column widths
  • Text wrapping for long content
  • Smooth keyboard navigation
  • Beautiful styling with Rich
"""

import json
import sys
import os
import signal
import textwrap
from typing import Optional
from pathlib import Path

try:
    from rich.console import Console
    from rich.table import Table
    from rich.panel import Panel
    from rich.text import Text
    from rich.style import Style
    from rich.box import ROUNDED, DOUBLE, HEAVY, MINIMAL
    from rich import box
except ImportError:
    print("Installing required package: rich")
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "rich", "--break-system-packages", "-q"])
    from rich.console import Console
    from rich.table import Table
    from rich.panel import Panel
    from rich.text import Text
    from rich.style import Style
    from rich.box import ROUNDED, DOUBLE, HEAVY, MINIMAL
    from rich import box

# Platform-specific imports for keyboard handling
if sys.platform == 'win32':
    import msvcrt
else:
    import tty
    import termios


class JSONLViewer:
    """A beautiful interactive viewer for JSONL files."""
    
    # Color palette - clean light mode
    COLORS = {
        'header_bg': '#4a7c9b',
        'header_fg': '#ffffff',
        'row_even': '#ffffff',
        'row_odd': '#f0f4f8',
        'border': '#94a3b8',
        'accent': '#2563eb',
        'highlight': '#dc2626',
        'text': '#1e293b',
        'muted': '#64748b',
        'success': '#059669',
        'warning': '#d97706',
    }
    
    def __init__(self, filepath: str):
        self.filepath = Path(filepath)
        self.console = Console()
        self.records: list[dict] = []
        self.columns: list[str] = []
        self.line_offset = 0  # Now tracks text lines, not records
        self.running = True
        self.rendered_lines: list[str] = []  # Cache of rendered table lines
        
        # Calculate available space
        self.term_width = self.console.size.width
        self.term_height = self.console.size.height
        
        # Reserve space for header and status bar
        self.visible_lines = max(5, self.term_height - 6)
        
    def load_file(self) -> bool:
        """Load and parse the JSONL file."""
        if not self.filepath.exists():
            self.show_error(f"File not found: {self.filepath}")
            return False
            
        if not self.filepath.suffix.lower() in ['.jsonl', '.json', '.ndjson']:
            self.show_warning("File doesn't have a typical JSONL extension, attempting to parse anyway...")
        
        try:
            with open(self.filepath, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f, 1):
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        record = json.loads(line)
                        if isinstance(record, dict):
                            self.records.append(record)
                            # Collect all unique columns
                            for key in record.keys():
                                if key not in self.columns:
                                    self.columns.append(key)
                        else:
                            self.show_warning(f"Line {line_num}: Not a JSON object, skipping")
                    except json.JSONDecodeError as e:
                        self.show_warning(f"Line {line_num}: Invalid JSON - {e}")
                        
            if not self.records:
                self.show_error("No valid JSON records found in file")
                return False
                
            return True
            
        except Exception as e:
            self.show_error(f"Error reading file: {e}")
            return False
    
    def calculate_column_widths(self) -> dict[str, int]:
        """Calculate optimal column widths based on content."""
        # Available width (accounting for borders and padding)
        available_width = self.term_width - (len(self.columns) * 3) - 4
        
        # Calculate max content width for each column
        max_widths = {}
        for col in self.columns:
            # Start with header length
            max_len = len(col)
            # Check content lengths
            for record in self.records:
                val = str(record.get(col, ''))
                # For very long content, we'll wrap it
                max_len = max(max_len, min(len(val), 100))
            max_widths[col] = max_len
        
        total_natural = sum(max_widths.values())
        
        if total_natural <= available_width:
            # Everything fits naturally
            return max_widths
        
        # Need to distribute space
        # Give minimum 10 chars to each column, then distribute rest proportionally
        min_width = 10
        result = {col: min_width for col in self.columns}
        remaining = available_width - (min_width * len(self.columns))
        
        if remaining > 0:
            # Distribute remaining space based on natural widths
            for col in self.columns:
                proportion = max_widths[col] / total_natural
                extra = int(remaining * proportion)
                result[col] += extra
        
        return result
    
    def wrap_text(self, text: str, width: int) -> list[str]:
        """Wrap text to fit within specified width."""
        if not text:
            return ['']
        
        # Handle newlines in original text
        lines = []
        for paragraph in str(text).split('\n'):
            if len(paragraph) <= width:
                lines.append(paragraph)
            else:
                wrapped = textwrap.wrap(paragraph, width=width, break_long_words=True, break_on_hyphens=True)
                lines.extend(wrapped if wrapped else [''])
        
        return lines if lines else ['']
    
    def format_value(self, value) -> str:
        """Format a value for display."""
        if value is None:
            return '∅'
        elif isinstance(value, bool):
            return '✓' if value else '✗'
        elif isinstance(value, (list, dict)):
            return json.dumps(value, ensure_ascii=False)
        else:
            return str(value)
    
    def build_table(self) -> Table:
        """Build the Rich table with all rows."""
        col_widths = self.calculate_column_widths()
        
        # Create table with beautiful styling
        table = Table(
            box=box.ROUNDED,
            border_style=Style(color=self.COLORS['border']),
            header_style=Style(color=self.COLORS['header_fg'], bgcolor=self.COLORS['header_bg'], bold=True),
            title=f"[bold {self.COLORS['accent']}]📄 {self.filepath.name}[/]",
            title_style=Style(color=self.COLORS['accent']),
            show_lines=True,
            pad_edge=True,
            expand=True,
        )
        
        # Add columns
        for col in self.columns:
            table.add_column(
                col,
                width=col_widths.get(col, 20),
                overflow='fold',
                style=Style(color=self.COLORS['text']),
            )
        
        # Add ALL rows
        for idx, record in enumerate(self.records):
            row_style = self.COLORS['row_even'] if idx % 2 == 0 else self.COLORS['row_odd']
            
            row_values = []
            for col in self.columns:
                value = self.format_value(record.get(col, ''))
                row_values.append(value)
            
            table.add_row(*row_values, style=Style(bgcolor=row_style))
        
        return table
    
    def pre_render_table(self):
        """Pre-render the entire table to a list of lines."""
        from io import StringIO
        
        buffer = StringIO()
        buffer_console = Console(file=buffer, force_terminal=True, width=self.term_width)
        
        table = self.build_table()
        buffer_console.print(table)
        
        self.rendered_lines = buffer.getvalue().splitlines()
    
    def _get_status_text(self) -> str:
        """Generate status bar text."""
        total_lines = len(self.rendered_lines)
        start = self.line_offset + 1
        end = min(self.line_offset + self.visible_lines, total_lines)
        
        # Scroll indicator
        if total_lines <= self.visible_lines:
            scroll_indicator = "━" * 20
        else:
            progress = self.line_offset / max(1, total_lines - self.visible_lines)
            filled = int(progress * 18)
            scroll_indicator = "█" * filled + "░" * (18 - filled)
        
        return (
            f"[{self.COLORS['accent']}]Lines {start}-{end} of {total_lines}[/] │ "
            f"[{self.COLORS['muted']}]{len(self.records)} records[/] │ "
            f"[{self.COLORS['muted']}][{scroll_indicator}][/] │ "
            f"[{self.COLORS['success']}]↑↓[/] Scroll  "
            f"[{self.COLORS['warning']}]PgUp/PgDn[/] Page  "
            f"[{self.COLORS['highlight']}]Home/End[/] Jump  "
            f"[bold red]Ctrl+C[/] Exit"
        )
    
    def show_error(self, message: str):
        """Display an error message."""
        self.console.print(Panel(
            f"[bold red]✖ Error:[/] {message}",
            border_style="red",
            box=ROUNDED
        ))
    
    def show_warning(self, message: str):
        """Display a warning message."""
        self.console.print(f"[yellow]⚠ {message}[/]")
    
    def get_key(self) -> str:
        """Get a single keypress from the user."""
        if sys.platform == 'win32':
            key = msvcrt.getch()
            if key == b'\xe0':  # Special key prefix on Windows
                key = msvcrt.getch()
                if key == b'H': return 'up'
                if key == b'P': return 'down'
                if key == b'I': return 'pageup'
                if key == b'Q': return 'pagedown'
                if key == b'G': return 'home'
                if key == b'O': return 'end'
            elif key == b'\x03':  # Ctrl+C
                return 'quit'
            return key.decode('utf-8', errors='ignore')
        else:
            fd = sys.stdin.fileno()
            old_settings = termios.tcgetattr(fd)
            try:
                tty.setraw(fd)
                ch = sys.stdin.read(1)
                if ch == '\x03':  # Ctrl+C
                    return 'quit'
                if ch == '\x1b':  # Escape sequence
                    ch2 = sys.stdin.read(1)
                    if ch2 == '[':
                        ch3 = sys.stdin.read(1)
                        if ch3 == 'A': return 'up'
                        if ch3 == 'B': return 'down'
                        if ch3 == '5':
                            sys.stdin.read(1)  # consume ~
                            return 'pageup'
                        if ch3 == '6':
                            sys.stdin.read(1)  # consume ~
                            return 'pagedown'
                        if ch3 == 'H': return 'home'
                        if ch3 == 'F': return 'end'
                        if ch3 == '1':
                            ch4 = sys.stdin.read(1)
                            if ch4 == '~': return 'home'
                            sys.stdin.read(1)  # consume ~
                            return 'home'
                        if ch3 == '4':
                            sys.stdin.read(1)  # consume ~
                            return 'end'
                return ch
            finally:
                termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
    
    def handle_input(self, key: str) -> bool:
        """Handle keyboard input. Returns False to quit."""
        max_scroll = max(0, len(self.rendered_lines) - self.visible_lines)
        
        if key == 'quit':
            return False
        elif key == 'up':
            self.line_offset = max(0, self.line_offset - 1)
        elif key == 'down':
            self.line_offset = min(max_scroll, self.line_offset + 1)
        elif key == 'pageup':
            self.line_offset = max(0, self.line_offset - self.visible_lines)
        elif key == 'pagedown':
            self.line_offset = min(max_scroll, self.line_offset + self.visible_lines)
        elif key == 'home':
            self.line_offset = 0
        elif key == 'end':
            self.line_offset = max_scroll
        elif key == 'q':
            return False
        
        return True
    
    def render(self):
        """Render the current view using buffered output for speed."""
        from io import StringIO
        
        # Build header
        buffer = StringIO()
        buffer_console = Console(file=buffer, force_terminal=True, width=self.term_width)
        
        header = Panel(
            Text.assemble(
                ("JSONL Viewer", Style(color=self.COLORS['accent'], bold=True)),
                " │ ",
                (str(self.filepath), Style(color=self.COLORS['muted'])),
            ),
            box=ROUNDED,
            border_style=Style(color=self.COLORS['border']),
            padding=(0, 2),
        )
        buffer_console.print(header)
        
        # Get visible slice of pre-rendered table
        end_line = min(self.line_offset + self.visible_lines, len(self.rendered_lines))
        visible_content = self.rendered_lines[self.line_offset:end_line]
        
        # Add table lines
        buffer.write('\n'.join(visible_content))
        buffer.write('\n')
        
        # Add status bar
        buffer_console.print(self._get_status_text())
        
        # Clear and display everything at once
        output = f"\033[2J\033[H{buffer.getvalue()}"
        sys.stdout.write(output)
        sys.stdout.flush()
    
    def run(self):
        """Main run loop."""
        # Set up signal handler for clean exit
        def signal_handler(sig, frame):
            self.running = False
            sys.stdout.write(f"\033[2J\033[H\n\033[32m✨ Goodbye!\033[0m\n\n")
            sys.stdout.flush()
            sys.exit(0)
        
        signal.signal(signal.SIGINT, signal_handler)
        
        # Load data
        if not self.load_file():
            return
        
        self.console.print(f"\n[{self.COLORS['success']}]✓ Loaded {len(self.records)} records with {len(self.columns)} columns[/]\n")
        
        # Pre-render the table once
        self.pre_render_table()
        
        # Main loop
        last_size = (self.term_width, self.term_height)
        try:
            while self.running:
                # Check for terminal resize
                new_width = self.console.size.width
                new_height = self.console.size.height
                if (new_width, new_height) != last_size:
                    self.term_width = new_width
                    self.term_height = new_height
                    self.visible_lines = max(5, self.term_height - 6)
                    self.pre_render_table()  # Re-render on resize
                    last_size = (new_width, new_height)
                
                self.render()
                key = self.get_key()
                if not self.handle_input(key):
                    break
        finally:
            sys.stdout.write(f"\033[2J\033[H\n\033[32m✨ Thanks for using JSONL Viewer!\033[0m\n\n")
            sys.stdout.flush()


def show_help():
    """Display help information."""
    console = Console()
    help_text = """
[bold cyan]JSONL Viewer[/] - A beautiful CLI for exploring JSONL files

[bold]Usage:[/]
    jsonl_viewer.py <file.jsonl>
    
[bold]Navigation:[/]
    [green]↑/↓[/]         Scroll one row up/down
    [green]PgUp/PgDn[/]   Scroll one page up/down  
    [green]Home/End[/]    Jump to beginning/end
    [green]q[/]           Quit
    [green]Ctrl+C[/]      Exit immediately

[bold]Supported formats:[/]
    • .jsonl (JSON Lines)
    • .ndjson (Newline Delimited JSON)
    • .json (will attempt to parse as JSONL)
    
[bold]Features:[/]
    • Auto-calculated column widths
    • Text wrapping for long content
    • Alternating row colors
    • Progress indicator
    • Beautiful Rich formatting
"""
    console.print(Panel(help_text, title="Help", border_style="cyan", box=ROUNDED))


def main():
    if len(sys.argv) < 2:
        show_help()
        Console().print("\n[red]Error: Please provide a JSONL file path[/]\n")
        sys.exit(1)
    
    if sys.argv[1] in ['-h', '--help', 'help']:
        show_help()
        sys.exit(0)
    
    viewer = JSONLViewer(sys.argv[1])
    viewer.run()


if __name__ == '__main__':
    main()