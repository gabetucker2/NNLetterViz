import os
import traceback
from collections import defaultdict
from typing import List, Tuple
from colorama import Fore, Style
import colorama

colorama.init(autoreset=True)

class LogFormatter:
    def __init__(self, max_repeats: int = 3):
        self.message_counts = defaultdict(int)
        self.max_repeats = max_repeats

    def format(self, raw_message: str, tag: str, indent_level: int = 0) -> str:
        prefix = "  |" * indent_level
        return f"{prefix}[{tag}] {raw_message}"

    def track(self, base_msg: str) -> int:
        self.message_counts[base_msg] += 1
        count = self.message_counts[base_msg]
        if count == self.max_repeats:
            return 1
        elif count > self.max_repeats:
            return 2
        return 0

class Log:
    indent_level = 0

    def __init__(self):
        self.is_internal = False
        self.formatter = LogFormatter()
        self.queued_logs: List[Tuple[str, str, int]] = []
        self.default_stack_depth = 1

    def _get_caller_info(self, n_back: int) -> str:
        try:
            stack = traceback.extract_stack()
            index = max(0, len(stack) - n_back - 3)
            frame = stack[index]
            return f"{os.path.basename(frame.filename)}::{frame.name} @ Line {frame.lineno}"
        except Exception:
            return "UnknownSource"

    def _print(self, msg: str, suppression_state: int):
        if suppression_state == 0:
            print(msg)
        elif suppression_state == 1:
            print(f"{Fore.MAGENTA}[SUPPRESSED AFTER {self.formatter.max_repeats} REPEATS]{Style.RESET_ALL} {msg}")
        # suppression_state == 2: do not print

    def _log(self, message: str, tag: str, color: str = "", show_caller: bool = False, stack_depth: int = 1):
        if self.is_internal:
            return
        self.is_internal = True

        try:
            indent = Log.indent_level
            base_msg = self.formatter.format(message, tag, indent)
            suppression = self.formatter.track(base_msg)

            if show_caller:
                base_msg += f" | {self._get_caller_info(stack_depth)}"

            color_code = getattr(Fore, color.upper(), "")
            formatted_msg = f"{color_code}{base_msg}{Style.RESET_ALL}"

            self._print(formatted_msg, suppression)
        finally:
            self.is_internal = False

    def flush(self) -> List[Tuple[str, str, int]]:
        logs = self.queued_logs[:]
        self.queued_logs.clear()
        return logs

    # Logging entry points
    def warning(self, msg: str): self._log(msg, "WARNING", "yellow")
    def error(self, msg: str): self._log(msg, "ERROR", "red")

    def procedure(self, msg: str): self._log(msg, "PROCEDURE", "white")
    def epoch(self, msg: str): self._log(msg, "EPOCH", "cyan")
    def training(self, msg: str): self._log(msg, "TRAINING", "magenta")
    def testing(self, msg: str): self._log(msg, "TESTING", "white")
    def analysis(self, msg: str): self._log(msg, "ANALYSIS", "blue")

    def fwdProp(self, msg: str): self._log(msg, "FWDPROP", "lightmagenta_ex")
    def backProp(self, msg: str): self._log(msg, "BACKPROP", "lightblue_ex")

    def axons(self, msg: str): self._log(msg, "AXONS", "lightblack_ex")

log = Log()
