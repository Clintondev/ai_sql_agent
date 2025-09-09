from typing import Optional

import pandas as pd

# Impressão rica opcional
try:
    from rich.console import Console
    from rich.table import Table
    from rich.box import SIMPLE_HEAVY
except Exception:
    Console = None
    Table = None
    SIMPLE_HEAVY = None


class UIPrinter:
    def __init__(self, mode: str = "auto", max_colwidth: int = 100, no_color: bool = False):
        self.no_color = no_color
        self.max_colwidth = max_colwidth
        if mode == "auto":
            self.mode = "pretty" if Console is not None and not no_color else "plain"
        else:
            self.mode = mode
        self.console = Console(no_color=True) if (Console and no_color) else (Console() if Console else None)

    def set_mode(self, mode: str):
        if mode == "auto":
            self.mode = "pretty" if Console is not None and not self.no_color else "plain"
        else:
            self.mode = mode

    def _clip(self, s: str) -> str:
        if s is None:
            return ""
        s = str(s)
        if len(s) > self.max_colwidth:
            return s[: self.max_colwidth - 1] + "…"
        return s

    def _format_scalar(self, v):
        if isinstance(v, (int,)):
            return f"{v:,}".replace(",", ".")
        return str(v)

    def print_df(self, df: pd.DataFrame, title: Optional[str] = None):
        if df is None:
            print("(Sem dados)")
            return
        if df.empty:
            print("(Nenhum resultado)")
            return

        if df.shape == (1, 1):
            val = df.iloc[0, 0]
            label = df.columns[0]
            print(f"{label}: {self._format_scalar(val)}")
            return

        if self.mode == "pretty" and self.console and Table:
            table = Table(title=title, box=SIMPLE_HEAVY, show_lines=False, header_style="bold")
            for col in df.columns:
                table.add_column(str(col))
            for _, row in df.iterrows():
                table.add_row(*[self._clip(x) for x in row.tolist()])
            self.console.print(table)
        elif self.mode == "csv":
            print(df.to_csv(index=False))
        elif self.mode == "json":
            print(df.to_json(orient="records", force_ascii=False))
        else:
            with pd.option_context("display.max_colwidth", self.max_colwidth, "display.max_rows", 200):
                print(df.to_string(index=False))

    def print_info(self, msg: str):
        if self.console and self.mode == "pretty" and not self.no_color:
            self.console.print(f"[bold cyan]{msg}[/bold cyan]")
        else:
            print(msg)

    def print_warn(self, msg: str):
        if self.console and self.mode == "pretty" and not self.no_color:
            self.console.print(f"[bold yellow]{msg}[/bold yellow]")
        else:
            print(msg)

    def print_err(self, msg: str):
        if self.console and self.mode == "pretty" and not self.no_color:
            self.console.print(f"[bold red]{msg}[/bold red]")
        else:
            print(msg)

