import os
import traceback
from collections import defaultdict
from typing import List, Tuple

class LogFormatter:
    def __init__(self, max_repeats: int = 3):
        self.message_counts = defaultdict(int)
        self.max_repeats = max_repeats

    def format(self, raw_message: str, tag: str, indent_level: int = 0) -> str:
        prefix = "  |" * indent_level
        return f"{prefix}[{tag}] {raw_message}"

    def track(self, msg: str) -> int:
        self.message_counts[msg] += 1
        count = self.message_counts[msg]
        if count == self.max_repeats:
            return 1
        elif count > self.max_repeats:
            return 2
        return 0

class Log:
    indent_level = 0  # Global/static logging indent level

    def __init__(self):
        self.is_internal = False
        self.formatter = LogFormatter()
        self.queued_logs: List[Tuple[str, str, int]] = []
        self.default_stack_depth = 1

    def _get_caller_info(self, n_back: int) -> str:
        try:
            stack = traceback.extract_stack()
            index = max(0, len(stack) - n_back - 2)
            frame = stack[index]
            return f"{os.path.basename(frame.filename)}::{frame.name} @ Line {frame.lineno}"
        except Exception:
            return "UnknownSource"

    def _print(self, msg: str, suppression_state: int):
        if suppression_state == 0:
            print(msg)
        elif suppression_state == 1:
            print(f"\033[95m[SUPPRESSED AFTER {self.formatter.max_repeats} REPEATS]\033[0m {msg}")
        elif suppression_state == 2:
            pass
        else:
            print("[ERROR] Unknown suppression state.")

    def _log(self, message: str, tag: str, show_caller: bool = False, stack_depth: int = 1):
        if self.is_internal:
            return
        self.is_internal = True

        try:
            indent = Log.indent_level
            formatted = self.formatter.format(message, tag, indent)
            suppression = self.formatter.track(formatted)
            full_msg = f"{formatted} | {self._get_caller_info(stack_depth)}" if show_caller else formatted

            self._print(full_msg, suppression)
        finally:
            self.is_internal = False

    def flush(self) -> List[Tuple[str, str, int]]:
        logs = self.queued_logs[:]
        self.queued_logs.clear()
        return logs

    # Public log functions without needing a `level` argument
    def warning(self, msg: str): self._log(msg, "WARNING", "yellow")
    def error(self, msg: str): self._log(msg, "ERROR", "red")

    def procedure(self, msg: str): self._log(msg, "PROCEDURE", "white")
    def epoch(self, msg: str): self._log(msg, "EPOCH", "cyan")
    def training(self, msg: str): self._log(msg, "TRAINING", "pink")
    def testing(self, msg: str): self._log(msg, "TESTING", "white")

    def fwdProp(self, msg: str): self._log(msg, "FWDPROP", "orange")
    def backProp(self, msg: str): self._log(msg, "BACKPROP", "blue")

    def axons(self, msg: str): self._log(msg, "AXONS", "grey")
