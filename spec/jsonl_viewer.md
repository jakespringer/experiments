# Specification: `experiments/scripts/jsonl_viewer.py`

## Purpose
An interactive terminal-based viewer for JSONL (JSON Lines) files. Uses the Rich library for beautiful table rendering with keyboard navigation, auto-calculated column widths, text wrapping, and scrolling.

## Dependencies
- `json`, `sys`, `os`, `signal`, `textwrap`, `pathlib.Path`
- `rich` (auto-installed if missing via `pip install rich --break-system-packages -q`)
- Platform-specific: `msvcrt` (Windows) or `tty`/`termios` (Unix)

---

## Class: `JSONLViewer`

### Class Constants

#### `COLORS`
- Color palette dict with keys: `header_bg`, `header_fg`, `row_even`, `row_odd`, `border`, `accent`, `highlight`, `text`, `muted`, `success`, `warning`.
- Provides a clean light-mode color scheme.

### Constructor: `__init__(self, filepath: str)`
- Initializes:
  - `filepath` as `Path`.
  - `console` as `Console()`.
  - `records`: empty list of parsed JSON dicts.
  - `columns`: empty list of column names (in discovery order).
  - `line_offset = 0`: current scroll position in rendered lines.
  - `running = True`: main loop flag.
  - `rendered_lines`: empty list for cached pre-rendered table lines.
  - `term_width`, `term_height`: from `console.size`.
  - `visible_lines = max(5, term_height - 6)`: lines available for table display.

### Method: `load_file(self) -> bool`
- Returns `False` if file doesn't exist.
- Warns if extension is not `.jsonl`, `.json`, or `.ndjson`.
- Reads file line-by-line; skips blank lines.
- Parses each line as JSON; only keeps `dict` records.
- Collects unique column names in order of first appearance.
- Warns on non-dict records and invalid JSON lines.
- Returns `True` if at least one valid record found.

### Method: `calculate_column_widths(self) -> Dict[str, int]`
- Calculates optimal column widths based on content.
- Available width = `term_width - (len(columns) × 3) - 4` (accounting for borders/padding).
- Natural width per column: `max(len(header), min(content_length, 100))`.
- If total natural width fits: uses natural widths.
- Otherwise: distributes proportionally with minimum 10 characters per column.

### Method: `wrap_text(self, text: str, width: int) -> List[str]`
- Wraps text to fit within specified width.
- Handles embedded newlines by splitting first.
- Uses `textwrap.wrap` with `break_long_words=True`.

### Method: `format_value(self, value) -> str`
- `None` → `'∅'`
- `bool` → `'✓'` or `'✗'`
- `list`/`dict` → `json.dumps(value, ensure_ascii=False)`
- Other → `str(value)`

### Method: `build_table(self) -> Table`
- Creates a Rich `Table` with styling from `COLORS`.
- Uses `box.ROUNDED`, shows lines between rows.
- Title: filename with accent color and emoji.
- Adds columns with computed widths and `overflow='fold'`.
- Adds all rows with alternating background colors.
- All values are escaped via `rich.markup.escape` to prevent markup injection.

### Method: `pre_render_table(self)`
- Pre-renders the entire table to a list of string lines.
- Uses a `StringIO` buffer with a `Console` to capture rendered output.
- Stores result in `self.rendered_lines`.

### Method: `_get_status_text(self) -> str`
- Generates status bar with: line range, record count, scroll indicator bar, navigation hints.
- Scroll indicator: `█` (filled) and `░` (empty) proportional to scroll position.

### Method: `show_error(self, message: str)`
- Displays error in a red-bordered `Panel`.

### Method: `show_warning(self, message: str)`
- Displays warning with yellow styling.

### Method: `get_key(self) -> str`
- Cross-platform key reading.
- **Windows**: Uses `msvcrt.getch()` with special key prefix `\xe0`.
- **Unix**: Sets terminal to raw mode via `tty.setraw`, reads escape sequences.
- Returns semantic strings: `'up'`, `'down'`, `'pageup'`, `'pagedown'`, `'home'`, `'end'`, `'quit'`.
- Restores terminal settings in `finally` block.

### Method: `handle_input(self, key: str) -> bool`
- Processes keyboard input. Returns `False` to quit.
- `quit` or `q` → returns `False`.
- `up` → `line_offset -= 1` (clamped to 0).
- `down` → `line_offset += 1` (clamped to max).
- `pageup` → `line_offset -= visible_lines`.
- `pagedown` → `line_offset += visible_lines`.
- `home` → `line_offset = 0`.
- `end` → `line_offset = max_scroll`.

### Method: `render(self)`
- Renders current view using buffered output.
- Builds header panel, slices pre-rendered table lines, adds status bar.
- Clears screen with ANSI escape `\033[2J\033[H` and writes everything at once.
- Writes directly to `sys.stdout` for speed.

### Method: `run(self)`
- Main run loop.
- Sets up `SIGINT` handler for clean exit.
- Calls `load_file()`.
- Pre-renders table once.
- Main loop:
  - Detects terminal resize and re-renders table on size change.
  - Calls `render()` then `get_key()` then `handle_input()`.
  - Breaks on quit signal.
- Prints goodbye message on exit.

---

## Functions

### `show_help()`
- Displays help panel with usage instructions, navigation keys, supported formats, and features.

### `main()`
- If no arguments: shows help and error message, exits with code 1.
- If `-h`/`--help`/`help`: shows help, exits with code 0.
- Otherwise: creates `JSONLViewer(sys.argv[1])` and calls `run()`.

---

## Important Behaviors
- The entire table is pre-rendered once into a list of text lines; scrolling just changes which slice is displayed.
- On terminal resize: table is re-rendered to fit new dimensions.
- Rich library is auto-installed if not present (with `--break-system-packages` flag).
- ANSI escape sequences are used for screen clearing (not Rich's built-in).
- Column values are escaped to prevent Rich markup injection from data content.
