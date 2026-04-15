#!/usr/bin/env python3
"""
╔═══════════════════════════════════════════════════════════════════════════════╗
║                          TOKEN VIEWER                                         ║
║                   A Beautiful Terminal Token Inspector                        ║
╚═══════════════════════════════════════════════════════════════════════════════╝

A terminal-based viewer for inspecting tokenized data files with:
  • Split-pane view (tokens on left, decoded text on right)
  • Memory-mapped lazy loading for large files
  • Color-coded special tokens
  • Vim/less-style navigation

Usage:
    python token_viewer.py <token_file> --tokenizer <tokenizer_name> [--dtype <dtype>]

Example:
    python token_viewer.py data.bin --tokenizer gpt2 --dtype uint16
"""

import argparse
import curses
import signal
import sys
from dataclasses import dataclass
from typing import Optional, Dict, Tuple, List

import numpy as np


# ═══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION & CONSTANTS
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class ViewerConfig:
    """Configuration for the token viewer."""
    buffer_size: int = 10000      # Tokens to buffer around viewport
    scroll_threshold: int = 100   # Lines before triggering buffer reload
    token_padding: int = 2        # Padding between token columns
    divider_char: str = "│"
    header_char: str = "═"
    corner_tl: str = "╔"
    corner_tr: str = "╗"
    corner_bl: str = "╚"
    corner_br: str = "╝"
    tee_left: str = "╠"
    tee_right: str = "╣"


# Color pair definitions
class Colors:
    NORMAL = 1
    SPECIAL = 2
    HEADER = 3
    DIVIDER = 4
    LINE_NUM = 5
    HIGHLIGHT = 6
    EOS = 7
    BOS = 8
    PAD = 9
    UNK = 10
    NEWLINE = 11
    STATUS = 12


# ═══════════════════════════════════════════════════════════════════════════════
# TOKEN BUFFER - LAZY LOADING WITH MEMORY MAPPING
# ═══════════════════════════════════════════════════════════════════════════════

class TokenBuffer:
    """
    Manages lazy loading of tokens using memory-mapped files.
    Only loads tokens near the current viewport.
    """
    
    def __init__(self, filepath: str, dtype: str, buffer_size: int = 10000):
        self.filepath = filepath
        self.dtype = np.dtype(dtype)
        self.buffer_size = buffer_size
        
        # Create memory map (read-only)
        self.mmap = np.memmap(filepath, dtype=self.dtype, mode='r')
        self.total_tokens = len(self.mmap)
        
        # Current buffer state
        self.buffer_start = 0
        self.buffer_end = 0
        self.buffer: Optional[np.ndarray] = None
        
        # Determine max token width for display
        self.max_token_value = int(np.iinfo(self.dtype).max)
        self.token_width = len(str(self.max_token_value))
        
    def load_range(self, start: int, end: int) -> None:
        """Load a range of tokens into the buffer."""
        start = max(0, start)
        end = min(self.total_tokens, end)
        
        if start >= end:
            return
            
        # Load from mmap into regular array for faster access
        self.buffer = np.array(self.mmap[start:end])
        self.buffer_start = start
        self.buffer_end = end
        
    def ensure_loaded(self, center_idx: int) -> None:
        """Ensure tokens around center_idx are loaded."""
        half_buffer = self.buffer_size // 2
        desired_start = max(0, center_idx - half_buffer)
        desired_end = min(self.total_tokens, center_idx + half_buffer)
        
        # Check if we need to reload
        if (self.buffer is None or 
            desired_start < self.buffer_start or 
            desired_end > self.buffer_end):
            self.load_range(desired_start, desired_end)
    
    def get_token(self, idx: int) -> Optional[int]:
        """Get a single token by index."""
        if idx < 0 or idx >= self.total_tokens:
            return None
            
        if (self.buffer is not None and 
            self.buffer_start <= idx < self.buffer_end):
            return int(self.buffer[idx - self.buffer_start])
        
        # Fallback to direct mmap access (slower)
        return int(self.mmap[idx])
    
    def get_tokens(self, start: int, count: int) -> List[int]:
        """Get multiple tokens starting from start."""
        self.ensure_loaded(start + count // 2)
        
        result = []
        for i in range(start, min(start + count, self.total_tokens)):
            token = self.get_token(i)
            if token is not None:
                result.append(token)
        return result
    
    def close(self):
        """Clean up resources."""
        if hasattr(self, 'mmap'):
            del self.mmap
            self.buffer = None


# ═══════════════════════════════════════════════════════════════════════════════
# TOKENIZER WRAPPER
# ═══════════════════════════════════════════════════════════════════════════════

class DemoTokenizer:
    """A simple demo tokenizer for testing without HuggingFace access."""
    
    # Common GPT-2 tokens for demo purposes
    DEMO_VOCAB = {
        13: ".", 25: ":", 198: "\n", 220: " ", 257: " a", 262: " the",
        286: " of", 318: " is", 464: "The", 625: " over", 815: " should",
        1026: "It", 1212: "This", 1332: " test", 2068: " quick",
        3290: " dog", 3586: " application", 3951: " lines", 5412: " handle",
        7586: " brown", 7651: " Special", 11241: " token", 12: "-",
        16326: " tokens", 16931: " lazy", 18045: " jumps", 19751: " viewer",
        20401: " Multiple", 21831: " fox", 50256: "<|endoftext|>",
        50257: "<|padding|>",
    }
    
    def __init__(self):
        self.eos_token_id = 50256
        self.bos_token_id = None
        self.pad_token_id = 50257
        self.unk_token_id = None
        self.additional_special_tokens_ids = []
        
    def decode(self, token_ids):
        if isinstance(token_ids, int):
            token_ids = [token_ids]
        result = []
        for tid in token_ids:
            if tid in self.DEMO_VOCAB:
                result.append(self.DEMO_VOCAB[tid])
            else:
                result.append(f"[{tid}]")
        return "".join(result)


class TokenizerWrapper:
    """Wraps HuggingFace tokenizer with caching and special token detection."""
    
    def __init__(self, tokenizer_name: str):
        if tokenizer_name == "demo":
            self.tokenizer = DemoTokenizer()
        else:
            from transformers import AutoTokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        
        self.cache: Dict[int, str] = {}
        
        # Identify special tokens
        self.special_tokens: Dict[int, str] = {}
        self.eos_token_id: Optional[int] = None
        self.bos_token_id: Optional[int] = None
        self.pad_token_id: Optional[int] = None
        self.unk_token_id: Optional[int] = None
        
        self._identify_special_tokens()
        
    def _identify_special_tokens(self):
        """Identify and cache special token IDs."""
        if self.tokenizer.eos_token_id is not None:
            self.eos_token_id = self.tokenizer.eos_token_id
            self.special_tokens[self.eos_token_id] = "<EOS>"
            
        if self.tokenizer.bos_token_id is not None:
            self.bos_token_id = self.tokenizer.bos_token_id
            self.special_tokens[self.bos_token_id] = "<BOS>"
            
        if self.tokenizer.pad_token_id is not None:
            self.pad_token_id = self.tokenizer.pad_token_id
            self.special_tokens[self.pad_token_id] = "<PAD>"
            
        if self.tokenizer.unk_token_id is not None:
            self.unk_token_id = self.tokenizer.unk_token_id
            self.special_tokens[self.unk_token_id] = "<UNK>"
            
        # Add any additional special tokens
        if hasattr(self.tokenizer, 'additional_special_tokens_ids'):
            for tid in self.tokenizer.additional_special_tokens_ids:
                if tid not in self.special_tokens:
                    try:
                        text = self.tokenizer.decode([tid])
                        self.special_tokens[tid] = f"<{text}>"
                    except:
                        self.special_tokens[tid] = f"<SPECIAL:{tid}>"
    
    def decode(self, token_id: int) -> str:
        """Decode a single token to string with caching."""
        if token_id in self.cache:
            return self.cache[token_id]
            
        try:
            # Use special token representation if applicable
            if token_id in self.special_tokens:
                text = self.special_tokens[token_id]
            else:
                text = self.tokenizer.decode([token_id])
        except Exception:
            text = f"<ERR:{token_id}>"
            
        self.cache[token_id] = text
        return text
    
    def is_special(self, token_id: int) -> bool:
        """Check if token is a special token."""
        return token_id in self.special_tokens
    
    def get_special_type(self, token_id: int) -> Optional[str]:
        """Get the type of special token."""
        if token_id == self.eos_token_id:
            return "eos"
        elif token_id == self.bos_token_id:
            return "bos"
        elif token_id == self.pad_token_id:
            return "pad"
        elif token_id == self.unk_token_id:
            return "unk"
        elif token_id in self.special_tokens:
            return "special"
        return None


# ═══════════════════════════════════════════════════════════════════════════════
# TEXT ESCAPING
# ═══════════════════════════════════════════════════════════════════════════════

def escape_text(text: str) -> Tuple[str, List[Tuple[int, int]]]:
    """
    Escape special characters for display.
    Returns escaped text and list of (start, end) positions of escape sequences.
    """
    escape_map = {
        '\n': '\\n',
        '\r': '\\r',
        '\t': '\\t',
        '\x00': '\\0',
        '\x0b': '\\v',
        '\x0c': '\\f',
        '\x1b': '\\e',
    }
    
    result = []
    escape_positions = []
    pos = 0
    
    for char in text:
        if char in escape_map:
            escaped = escape_map[char]
            escape_positions.append((pos, pos + len(escaped)))
            result.append(escaped)
            pos += len(escaped)
        elif ord(char) < 32 or ord(char) == 127:
            # Other control characters
            escaped = f'\\x{ord(char):02x}'
            escape_positions.append((pos, pos + len(escaped)))
            result.append(escaped)
            pos += len(escaped)
        else:
            result.append(char)
            pos += 1
            
    return ''.join(result), escape_positions


# ═══════════════════════════════════════════════════════════════════════════════
# VIEWER UI
# ═══════════════════════════════════════════════════════════════════════════════

class TokenViewer:
    """Main terminal UI for viewing tokens."""
    
    def __init__(self, token_buffer: TokenBuffer, tokenizer: TokenizerWrapper, 
                 config: ViewerConfig = None):
        self.buffer = token_buffer
        self.tokenizer = tokenizer
        self.config = config or ViewerConfig()
        
        self.scroll_offset = 0
        self.tokens_per_row = 8  # Will be recalculated based on terminal width
        self.text_width = 40     # Width for text display
        
    def setup_colors(self):
        """Initialize color pairs."""
        curses.start_color()
        curses.use_default_colors()
        
        # Define color pairs
        curses.init_pair(Colors.NORMAL, curses.COLOR_WHITE, -1)
        curses.init_pair(Colors.SPECIAL, curses.COLOR_MAGENTA, -1)
        curses.init_pair(Colors.HEADER, curses.COLOR_CYAN, -1)
        curses.init_pair(Colors.DIVIDER, curses.COLOR_BLUE, -1)
        curses.init_pair(Colors.LINE_NUM, curses.COLOR_YELLOW, -1)
        curses.init_pair(Colors.HIGHLIGHT, curses.COLOR_BLACK, curses.COLOR_WHITE)
        curses.init_pair(Colors.EOS, curses.COLOR_RED, -1)
        curses.init_pair(Colors.BOS, curses.COLOR_GREEN, -1)
        curses.init_pair(Colors.PAD, curses.COLOR_BLACK, -1)
        curses.init_pair(Colors.UNK, curses.COLOR_YELLOW, -1)
        curses.init_pair(Colors.NEWLINE, curses.COLOR_CYAN, -1)
        curses.init_pair(Colors.STATUS, curses.COLOR_BLACK, curses.COLOR_CYAN)
        
    def get_token_color(self, token_id: int) -> int:
        """Get the color pair for a token."""
        special_type = self.tokenizer.get_special_type(token_id)
        
        if special_type == "eos":
            return curses.color_pair(Colors.EOS) | curses.A_BOLD
        elif special_type == "bos":
            return curses.color_pair(Colors.BOS) | curses.A_BOLD
        elif special_type == "pad":
            return curses.color_pair(Colors.PAD) | curses.A_DIM
        elif special_type == "unk":
            return curses.color_pair(Colors.UNK) | curses.A_BOLD
        elif special_type == "special":
            return curses.color_pair(Colors.SPECIAL) | curses.A_BOLD
        else:
            return curses.color_pair(Colors.NORMAL)
    
    def calculate_layout(self, width: int, height: int):
        """Calculate display layout based on terminal size."""
        # Reserve space for line numbers (8 chars) and divider (3 chars)
        line_num_width = 10
        divider_width = 3
        
        # Calculate tokens panel width (left side)
        token_col_width = self.buffer.token_width + self.config.token_padding
        
        # Text panel gets about 40% of remaining space, tokens get 60%
        available = width - line_num_width - divider_width
        self.text_width = max(20, available * 2 // 5)
        tokens_width = available - self.text_width
        
        self.tokens_per_row = max(1, tokens_width // token_col_width)
        self.visible_rows = height - 4  # Reserve for header and status
        
        self.line_num_width = line_num_width
        self.tokens_panel_width = self.tokens_per_row * token_col_width
        self.divider_col = line_num_width + self.tokens_panel_width
        
    def draw_header(self, stdscr, width: int):
        """Draw the header bar."""
        header_attr = curses.color_pair(Colors.HEADER) | curses.A_BOLD
        
        # Top border
        stdscr.addstr(0, 0, self.config.corner_tl, header_attr)
        stdscr.addstr(0, 1, self.config.header_char * (width - 2), header_attr)
        try:
            stdscr.addstr(0, width - 1, self.config.corner_tr, header_attr)
        except curses.error:
            pass
        
        # Title
        title = " TOKEN VIEWER "
        title_pos = (width - len(title)) // 2
        stdscr.addstr(0, title_pos, title, header_attr | curses.A_REVERSE)
        
        # Column headers
        stdscr.addstr(1, 0, self.config.tee_left, curses.color_pair(Colors.DIVIDER))
        
        token_header = " TOKENS (base-10)"
        text_header = " DECODED TEXT"
        
        stdscr.addstr(1, 2, token_header.ljust(self.divider_col - 2), header_attr)
        stdscr.addstr(1, self.divider_col, self.config.divider_char, 
                     curses.color_pair(Colors.DIVIDER))
        stdscr.addstr(1, self.divider_col + 2, text_header, header_attr)
        
        try:
            stdscr.addstr(1, width - 1, self.config.tee_right, 
                         curses.color_pair(Colors.DIVIDER))
        except curses.error:
            pass
        
        # Separator line
        stdscr.addstr(2, 0, self.config.header_char * width, 
                     curses.color_pair(Colors.DIVIDER))
        
    def draw_status(self, stdscr, height: int, width: int):
        """Draw the status bar."""
        status_attr = curses.color_pair(Colors.STATUS)
        
        # Calculate current position info
        start_token = self.scroll_offset * self.tokens_per_row
        end_token = min(start_token + self.visible_rows * self.tokens_per_row,
                       self.buffer.total_tokens)
        
        percentage = (start_token / max(1, self.buffer.total_tokens)) * 100
        
        left_status = f" Tokens {start_token:,}-{end_token:,} of {self.buffer.total_tokens:,}"
        right_status = f"{percentage:.1f}% | ↑↓ scroll | q quit "
        
        # Draw status bar
        status_line = left_status + " " * (width - len(left_status) - len(right_status)) + right_status
        
        try:
            stdscr.addstr(height - 1, 0, status_line[:width], status_attr)
        except curses.error:
            pass
    
    def draw_row(self, stdscr, row: int, screen_row: int, width: int):
        """Draw a single row of tokens and their decoded text."""
        start_idx = row * self.tokens_per_row
        tokens = self.buffer.get_tokens(start_idx, self.tokens_per_row)
        
        if not tokens:
            return
            
        y = screen_row + 3  # Account for header
        
        # Draw line number
        line_num = f"{start_idx:>{self.line_num_width - 2}} "
        stdscr.addstr(y, 0, line_num, 
                     curses.color_pair(Colors.LINE_NUM) | curses.A_DIM)
        
        # Draw tokens
        x = self.line_num_width
        token_width = self.buffer.token_width + self.config.token_padding
        
        decoded_parts = []
        
        for i, token in enumerate(tokens):
            token_str = str(token).rjust(self.buffer.token_width)
            color = self.get_token_color(token)
            
            try:
                stdscr.addstr(y, x, token_str, color)
            except curses.error:
                pass
                
            x += token_width
            
            # Get decoded text for this token
            decoded = self.tokenizer.decode(token)
            decoded_parts.append((token, decoded))
        
        # Draw divider
        try:
            stdscr.addstr(y, self.divider_col, f" {self.config.divider_char} ",
                         curses.color_pair(Colors.DIVIDER))
        except curses.error:
            pass
        
        # Draw decoded text
        text_x = self.divider_col + 3
        available_width = width - text_x - 1
        
        current_x = text_x
        for token_id, decoded in decoded_parts:
            escaped, escape_positions = escape_text(decoded)
            color = self.get_token_color(token_id)
            
            # Check if this contains escape sequences
            has_escapes = len(escape_positions) > 0
            
            for j, char in enumerate(escaped):
                if current_x >= width - 1:
                    break
                    
                # Check if this character is part of an escape sequence
                in_escape = any(start <= j < end for start, end in escape_positions)
                
                char_color = color
                if in_escape:
                    char_color = curses.color_pair(Colors.NEWLINE) | curses.A_DIM
                
                try:
                    stdscr.addch(y, current_x, char, char_color)
                except curses.error:
                    pass
                current_x += 1
                
    def run(self, stdscr):
        """Main run loop."""
        # Setup
        curses.curs_set(0)  # Hide cursor
        stdscr.timeout(100)  # For responsive key handling
        self.setup_colors()
        
        # Initial load
        self.buffer.ensure_loaded(0)
        
        while True:
            stdscr.clear()
            height, width = stdscr.getmaxyx()
            
            self.calculate_layout(width, height)
            
            # Ensure tokens are loaded for current view
            center_token = (self.scroll_offset + self.visible_rows // 2) * self.tokens_per_row
            self.buffer.ensure_loaded(center_token)
            
            # Draw UI
            self.draw_header(stdscr, width)
            
            # Draw visible rows
            max_rows = (self.buffer.total_tokens + self.tokens_per_row - 1) // self.tokens_per_row
            
            for i in range(self.visible_rows):
                row = self.scroll_offset + i
                if row >= max_rows:
                    break
                self.draw_row(stdscr, row, i, width)
            
            self.draw_status(stdscr, height, width)
            
            stdscr.refresh()
            
            # Handle input
            try:
                key = stdscr.getch()
            except KeyboardInterrupt:
                break
                
            if key == ord('q') or key == ord('Q'):
                break
            elif key == curses.KEY_DOWN or key == ord('j'):
                max_scroll = max(0, max_rows - self.visible_rows)
                self.scroll_offset = min(self.scroll_offset + 1, max_scroll)
            elif key == curses.KEY_UP or key == ord('k'):
                self.scroll_offset = max(0, self.scroll_offset - 1)
            elif key == curses.KEY_NPAGE or key == ord(' '):  # Page Down
                max_scroll = max(0, max_rows - self.visible_rows)
                self.scroll_offset = min(self.scroll_offset + self.visible_rows, max_scroll)
            elif key == curses.KEY_PPAGE:  # Page Up
                self.scroll_offset = max(0, self.scroll_offset - self.visible_rows)
            elif key == curses.KEY_HOME or key == ord('g'):
                self.scroll_offset = 0
            elif key == curses.KEY_END or key == ord('G'):
                max_scroll = max(0, max_rows - self.visible_rows)
                self.scroll_offset = max_scroll


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Beautiful terminal-based token viewer",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s tokens.bin --tokenizer gpt2
  %(prog)s data.bin --tokenizer meta-llama/Llama-2-7b-hf --dtype int32
  %(prog)s corpus.bin --tokenizer EleutherAI/gpt-neox-20b --dtype uint16
  %(prog)s sample.bin --tokenizer demo          # Use built-in demo tokenizer

Navigation:
  ↑/k         Scroll up one line
  ↓/j         Scroll down one line
  PgUp        Scroll up one page
  PgDn/Space  Scroll down one page
  g/Home      Go to beginning
  G/End       Go to end
  q           Quit

Colors:
  Red         End of sequence (EOS) token
  Green       Beginning of sequence (BOS) token
  Dim         Padding token
  Yellow      Unknown token
  Magenta     Other special tokens
  Cyan        Escaped characters (\\n, \\t, etc.)
        """
    )
    
    parser.add_argument("file", help="Path to the token file (raw bytes)")
    parser.add_argument("--tokenizer", "-t", required=True,
                       help="HuggingFace tokenizer name or path")
    parser.add_argument("--dtype", "-d", default="uint16",
                       help="NumPy dtype for tokens (default: uint16)")
    parser.add_argument("--buffer-size", "-b", type=int, default=10000,
                       help="Number of tokens to buffer (default: 10000)")
    
    args = parser.parse_args()
    
    # Validate dtype
    try:
        np.dtype(args.dtype)
    except TypeError:
        print(f"Error: Invalid dtype '{args.dtype}'", file=sys.stderr)
        sys.exit(1)
    
    # Load components
    print(f"Loading tokenizer: {args.tokenizer}...")
    try:
        tokenizer = TokenizerWrapper(args.tokenizer)
    except Exception as e:
        print(f"Error loading tokenizer: {e}", file=sys.stderr)
        sys.exit(1)
    
    print(f"Opening token file: {args.file}...")
    try:
        token_buffer = TokenBuffer(args.file, args.dtype, args.buffer_size)
    except Exception as e:
        print(f"Error opening file: {e}", file=sys.stderr)
        sys.exit(1)
    
    print(f"Found {token_buffer.total_tokens:,} tokens")
    print("Starting viewer...")
    
    # Setup signal handler for clean exit
    def signal_handler(sig, frame):
        token_buffer.close()
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    
    # Create and run viewer
    config = ViewerConfig(buffer_size=args.buffer_size)
    viewer = TokenViewer(token_buffer, tokenizer, config)
    
    try:
        curses.wrapper(viewer.run)
    finally:
        token_buffer.close()
    
    print("Goodbye!")


if __name__ == "__main__":
    main()