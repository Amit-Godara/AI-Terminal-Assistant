"""Terminal UI module using Rich for formatting and interactive display.

Provides styled banners, help screens, input prompts, status badges,
informational preview cards, validation cards, confirmation prompts,
execution results, and command history tables.
"""

from datetime import datetime
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from rich.syntax import Syntax
from rich import box

from core.command_generator import (
    CommandOperation,
    OperationType,
    RiskLevel,
    GenerationStatus,
)
from core.command_validator import (
    ValidationResult,
    ValidationStatus,
)
from core.executor import (
    ExecutionResult,
    ExecutionStatus,
)


class TerminalUI:
    """Provides styled, readable terminal interfaces using Rich."""

    def __init__(self, console: Optional[Console] = None) -> None:
        self.console = console or Console()

    # --------------------------------------------------------------------------
    # General Messaging & Banners
    # --------------------------------------------------------------------------

    def show_banner(self) -> None:
        """Displays application header and status banner."""
        banner_text = Text()
        banner_text.append("AI TERMINAL ASSISTANT\n", style="bold cyan")
        banner_text.append(
            "Intelligent Windows PowerShell Command Assistant (NLP-Powered)\n",
            style="bold white",
        )
        banner_text.append(
            "Hybrid Intent Classification • Safe Filesystem Handlers • Independent Validation",
            style="dim italic cyan",
        )
        self.console.print(
            Panel(
                banner_text,
                border_style="cyan",
                box=box.DOUBLE,
                padding=(1, 2),
            )
        )

    def show_welcome(
        self,
        workspace_path: Path,
        model_info: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Displays welcome screen with active environment and workspace details."""
        welcome_table = Table.grid(padding=(0, 2))
        welcome_table.add_column(style="bold yellow", justify="right")
        welcome_table.add_column(style="white")

        welcome_table.add_row("Platform:", "Windows (PowerShell compatible)")
        welcome_table.add_row("Sandbox Workspace:", f"[bold green]{workspace_path}[/bold green]")
        welcome_table.add_row("Safety Mode:", "[bold green]ACTIVE (Strict Workspace Containment)[/bold green]")
        welcome_table.add_row(
            "Commands:",
            "[cyan]$ask <query>[/cyan] • [cyan]$help[/cyan] • [cyan]$history[/cyan] • [cyan]$clear[/cyan] • [cyan]$exit[/cyan]",
        )

        if model_info:
            for k, v in model_info.items():
                welcome_table.add_row(f"{k}:", str(v))

        self.console.print(
            Panel(
                welcome_table,
                title="[bold green]Ready for Instructions[/bold green]",
                border_style="green",
                box=box.ROUNDED,
                padding=(0, 1),
            )
        )

    def show_help(self, workspace_path: Optional[Path] = None) -> None:
        """Displays comprehensive interactive help screen and safety rules."""
        help_table = Table(
            title="Supported Interactive Commands",
            box=box.ROUNDED,
            header_style="bold cyan",
            border_style="dim cyan",
        )
        help_table.add_column("Command", style="bold yellow", width=22)
        help_table.add_column("Description", style="white")
        help_table.add_column("Example", style="dim italic")

        help_table.add_row(
            "$ask <request>",
            "Process a natural-language terminal instruction",
            '$ask "list all pdf files in workspace"',
        )
        help_table.add_row(
            "<natural language>",
            "Direct query without prefix (shorthand for $ask)",
            "show current directory",
        )
        help_table.add_row(
            "$history",
            "View recent command execution history and audit log",
            "$history",
        )
        help_table.add_row(
            "$clear",
            "Clear the terminal screen",
            "$clear",
        )
        help_table.add_row(
            "$help",
            "Display this help screen and security policies",
            "$help",
        )
        help_table.add_row(
            "$exit / quit",
            "Exit the AI Terminal Assistant cleanly",
            "$exit",
        )

        self.console.print(help_table)

        # Safety policies panel
        safety_text = Text()
        safety_text.append("Safety Architecture & Risk Levels:\n", style="bold underline yellow")
        safety_text.append("  • ", style="dim")
        safety_text.append("LOW (Read-Only): ", style="bold green")
        safety_text.append(
            "Listing files, searching, version check, network & system info. Validated and executed automatically.\n"
        )
        safety_text.append("  • ", style="dim")
        safety_text.append("MEDIUM (State Modification): ", style="bold yellow")
        safety_text.append(
            "Creating files/directories, copying, moving. Requires explicit user approval [y/N]. Target overwrite blocked.\n"
        )
        safety_text.append("  • ", style="dim")
        safety_text.append("HIGH (Destructive): ", style="bold red")
        safety_text.append(
            "Deleting files. Requires user approval AND explicit confirmation token 'DELETE'. Directory deletion strictly forbidden.\n"
        )
        safety_text.append("  • ", style="dim")
        safety_text.append("Workspace Sandbox: ", style="bold cyan")
        ws_str = str(workspace_path) if workspace_path else "config.WORKSPACE_DIR"
        safety_text.append(f"Operations are strictly confined to '{ws_str}'. Traversal (..) and system directories blocked.\n")
        safety_text.append("  • ", style="dim")
        safety_text.append("Informational Preview: ", style="bold magenta")
        safety_text.append(
            "PowerShell previews are for user reference only. Execution strictly uses Python standard library handlers."
        )

        self.console.print(
            Panel(
                safety_text,
                title="[bold yellow]Security Policies & Safeguards[/bold yellow]",
                border_style="yellow",
                box=box.ROUNDED,
            )
        )

    def print_message(self, message: str, style: str = "white") -> None:
        """Prints a styled message."""
        self.console.print(f"[{style}]{message}[/{style}]")

    def show_error(self, message: str, error_type: Optional[str] = None) -> None:
        """Prints a styled error message."""
        err_title = f"Error ({error_type})" if error_type else "Error"
        self.console.print(
            Panel(
                f"[bold red]{message}[/bold red]",
                title=f"[bold red]{err_title}[/bold red]",
                border_style="red",
                box=box.ROUNDED,
            )
        )

    def show_warning(self, message: str) -> None:
        """Prints a styled warning message."""
        self.console.print(f"[bold yellow]Warning:[/bold yellow] [yellow]{message}[/yellow]")

    def show_exit(self) -> None:
        """Displays exit message."""
        self.console.print("\n[bold cyan]Exiting AI Terminal Assistant. Goodbye![/bold cyan]")

    def clear_screen(self) -> None:
        """Cross-platform terminal screen clear."""
        os.system("cls" if os.name == "nt" else "clear")
        self.console.clear()

    # --------------------------------------------------------------------------
    # Interactive User Prompts
    # --------------------------------------------------------------------------

    def prompt_user(self, prompt_text: str = "assistant> ") -> str:
        """Prompts user for input via console, handling EOF and Interrupt gracefully."""
        try:
            self.console.print(f"[bold cyan]{prompt_text}[/bold cyan]", end="")
            user_input = input()
            return user_input.strip()
        except (KeyboardInterrupt, EOFError):
            return "$exit"

    def prompt_confirmation(
        self,
        operation_type: str,
        risk_level: RiskLevel,
        details: str = "",
    ) -> bool:
        """Prompts user for [y/N] confirmation. Defaults to NO.

        Only explicit 'y' or 'yes' approves. Blank, EOF, or other inputs reject.
        """
        risk_style = "bold red" if risk_level == RiskLevel.HIGH else "bold yellow"
        self.console.print()
        self.console.print(
            Panel(
                f"[bold white]Operation:[/bold white] [cyan]{operation_type}[/cyan]\n"
                f"[bold white]Risk Level:[/bold white] [{risk_style}]{risk_level.value}[/{risk_style}]\n"
                f"{details}\n\n"
                f"[{risk_style}]Execute this operation? [y/N][/{risk_style}]",
                title="[bold yellow]Authorization Required[/bold yellow]",
                border_style="yellow",
                box=box.ROUNDED,
            )
        )
        try:
            self.console.print("[bold yellow]Confirmation [y/N]: [/bold yellow]", end="")
            choice = input().strip().lower()
            return choice in ("y", "yes")
        except (KeyboardInterrupt, EOFError):
            self.console.print("\n[yellow]Operation cancelled by user.[/yellow]")
            return False

    def prompt_deletion_token(self, target_name: str) -> str:
        """Prompts user for exact 'DELETE' confirmation token."""
        self.console.print()
        self.console.print(
            Panel(
                f"[bold red]WARNING: Destructive Operation![/bold red]\n"
                f"Target file: [bold white]{target_name}[/bold white]\n"
                f"This file will be permanently removed. This action cannot be undone.\n\n"
                f"Type [bold red]DELETE[/bold red] to permanently delete this file:",
                title="[bold red]Double Confirmation Gating[/bold red]",
                border_style="red",
                box=box.DOUBLE,
            )
        )
        try:
            self.console.print("[bold red]Token: [/bold red]", end="")
            token = input().strip()
            return token
        except (KeyboardInterrupt, EOFError):
            self.console.print("\n[yellow]Deletion cancelled by user.[/yellow]")
            return ""

    # --------------------------------------------------------------------------
    # NLP Pipeline Display Cards
    # --------------------------------------------------------------------------

    def show_query_header(self, query: str) -> None:
        """Displays user query header."""
        self.console.print(f"\n[dim]─[/dim]" * 60)
        self.console.print(f"[bold cyan]Input Request:[/bold cyan] [white]{query}[/white]")

    def show_intent(
        self,
        intent: str,
        confidence: float,
        source: Optional[str] = None,
    ) -> None:
        """Displays predicted intent, confidence score, and inference source."""
        conf_style = "bold green" if confidence >= 0.70 else ("bold yellow" if confidence >= 0.40 else "bold red")
        source_str = f" [dim]({source})[/dim]" if source else ""
        self.console.print(
            f"[bold cyan]Predicted Intent:[/bold cyan] [bold white]{intent}[/bold white]  "
            f"[bold cyan]Confidence:[/bold cyan] [{conf_style}]{confidence:.1%}[/{conf_style}]{source_str}"
        )

    def show_inference_metadata(self, metadata: Dict[str, Any]) -> None:
        """Displays semantic inference details if available."""
        if not metadata:
            return
        clf_intent = metadata.get("classifier_intent")
        clf_conf = metadata.get("classifier_confidence")
        sem_intent = metadata.get("semantic_intent")
        sem_score = metadata.get("semantic_score")
        source = metadata.get("source")

        details = []
        if source:
            details.append(f"Inference Mode: [bold cyan]{source}[/bold cyan]")
        if clf_intent and clf_conf is not None:
            details.append(f"Classifier: {clf_intent} ({clf_conf:.1%})")
        if sem_intent and sem_score is not None:
            details.append(f"Semantic Engine: {sem_intent} (similarity {sem_score:.2f})")

        if details:
            self.console.print(f"[dim]  └─ {' • '.join(details)}[/dim]")

    def show_entities(self, entities: Dict[str, Any]) -> None:
        """Displays extracted entities in a compact, styled representation."""
        if not entities:
            self.console.print("[bold cyan]Extracted Entities:[/bold cyan] [dim](none)[/dim]")
            return

        ent_parts = []
        for k, v in entities.items():
            if v is not None and v != "":
                ent_parts.append(f"[bold white]{k}:[/bold white] [green]{v}[/green]")

        if ent_parts:
            self.console.print(f"[bold cyan]Extracted Entities:[/bold cyan] {', '.join(ent_parts)}")
        else:
            self.console.print("[bold cyan]Extracted Entities:[/bold cyan] [dim](none)[/dim]")

    def show_command_preview(
        self,
        powershell_preview: Optional[str],
        explanation: str,
        risk_level: RiskLevel,
    ) -> None:
        """Displays the informational PowerShell command preview and explanation."""
        risk_badge = self._get_risk_badge(risk_level)

        preview_content = Text()
        if powershell_preview:
            preview_content.append("PowerShell Preview (Informational Only):\n", style="bold dim white")
            # Render command in distinctive style
            self.console.print(
                Panel(
                    Syntax(powershell_preview, "powershell", theme="monokai", line_numbers=False),
                    title=f"[bold cyan]Preview & Proposed Action[/bold cyan] — Risk: {risk_badge}",
                    subtitle="[dim italic red]⚠ Informational preview. Never executed directly as arbitrary shell script.[/dim italic red]",
                    border_style="cyan",
                    box=box.ROUNDED,
                )
            )
        else:
            self.console.print(
                Panel(
                    "[dim italic]No PowerShell command preview generated for this operation.[/dim italic]",
                    title=f"[bold yellow]Notice[/bold yellow] — Risk: {risk_badge}",
                    border_style="yellow",
                    box=box.ROUNDED,
                )
            )

        if explanation:
            self.console.print(f"[bold cyan]Explanation:[/bold cyan] [white]{explanation}[/white]")

    def show_clarification_prompt(self, prompt_text: str, missing_params: List[str]) -> None:
        """Displays clarification request when arguments are missing."""
        missing_str = ", ".join(f"[bold yellow]{p}[/bold yellow]" for p in missing_params)
        self.console.print(
            Panel(
                f"[bold white]{prompt_text}[/bold white]\n\n"
                f"Missing Required Arguments: {missing_str}",
                title="[bold yellow]Clarification Needed[/bold yellow]",
                border_style="yellow",
                box=box.ROUNDED,
            )
        )

    # --------------------------------------------------------------------------
    # Validation & Execution Display Cards
    # --------------------------------------------------------------------------

    def show_validation_result(self, val_result: ValidationResult) -> None:
        """Displays the outcome of independent security validation."""
        if val_result.is_valid:
            status_text = "[bold green]PASSED (VALID)[/bold green]"
            border_style = "green"
            details = f"[green]{val_result.reason or 'Operation verified safe within workspace boundary.'}[/green]"
        else:
            status_text = f"[bold red]BLOCKED ({val_result.validation_status.value})[/bold red]"
            border_style = "red"
            error_lines = "\n".join(f"  • {e}" for e in val_result.validation_errors) or val_result.reason
            details = f"[red]{error_lines}[/red]"

        val_grid = Table.grid(padding=(0, 2))
        val_grid.add_column(style="bold white", justify="right")
        val_grid.add_column()
        val_grid.add_row("Validation Status:", status_text)
        val_grid.add_row("Policy Detail:", details)

        if val_result.target_path:
            val_grid.add_row("Target Resolved:", f"[dim]{val_result.target_path}[/dim]")
        if val_result.destination_path:
            val_grid.add_row("Destination Resolved:", f"[dim]{val_result.destination_path}[/dim]")

        self.console.print(
            Panel(
                val_grid,
                title="[bold white]Independent Safety Validation[/bold white]",
                border_style=border_style,
                box=box.ROUNDED,
            )
        )

    def show_execution_result(self, result: ExecutionResult) -> None:
        """Displays actual execution status, output, errors, and timing."""
        if result.success:
            status_badge = "[bold green]SUCCESS[/bold green]"
            border_style = "green"
        elif result.was_cancelled:
            status_badge = "[bold yellow]CANCELLED BY USER[/bold yellow]"
            border_style = "yellow"
        elif result.execution_status == ExecutionStatus.TIMEOUT:
            status_badge = "[bold red]TIMED OUT[/bold red]"
            border_style = "red"
        elif result.execution_status == ExecutionStatus.TARGET_ALREADY_EXISTS:
            status_badge = "[bold yellow]TARGET ALREADY EXISTS[/bold yellow]"
            border_style = "yellow"
        else:
            status_badge = f"[bold red]{result.execution_status.value}[/bold red]"
            border_style = "red"

        exec_grid = Table.grid(padding=(0, 2))
        exec_grid.add_column(style="bold white", justify="right")
        exec_grid.add_column()

        exec_grid.add_row("Outcome:", status_badge)
        exec_grid.add_row("Duration:", f"{result.execution_time_ms:.1f} ms")
        exec_grid.add_row("Return Code:", str(result.return_code))

        # Distinguish execution handler type
        handler_info = self._get_handler_info(result.operation_type)
        exec_grid.add_row("Execution Handler:", f"[dim]{handler_info}[/dim]")

        if result.stdout:
            stdout_panel = Panel(
                Text(result.stdout.strip(), style="white"),
                title="[bold green]Standard Output[/bold green]",
                border_style="green",
                box=box.HORIZONTALS,
            )
        else:
            stdout_panel = None

        if result.stderr:
            stderr_panel = Panel(
                Text(result.stderr.strip(), style="red"),
                title="[bold red]Error / Cancellation Notice[/bold red]",
                border_style="red",
                box=box.HORIZONTALS,
            )
        else:
            stderr_panel = None

        self.console.print(
            Panel(
                exec_grid,
                title="[bold white]Execution Outcome[/bold white]",
                border_style=border_style,
                box=box.ROUNDED,
            )
        )

        if stdout_panel:
            self.console.print(stdout_panel)
        if stderr_panel:
            self.console.print(stderr_panel)

    # --------------------------------------------------------------------------
    # History Display
    # --------------------------------------------------------------------------

    def show_history(self, records: List[Dict[str, Any]], limit: int = 20) -> None:
        """Displays recent execution history in a Rich table."""
        if not records:
            self.console.print(
                Panel(
                    "[dim italic]No execution history records found in log.[/dim italic]",
                    title="[bold cyan]Command History[/bold cyan]",
                    border_style="cyan",
                    box=box.ROUNDED,
                )
            )
            return

        table = Table(
            title=f"Command Execution History (Showing last {min(len(records), limit)})",
            box=box.ROUNDED,
            header_style="bold cyan",
            border_style="dim cyan",
        )
        table.add_column("#", justify="right", style="dim", width=4)
        table.add_column("Timestamp", style="dim white", width=19)
        table.add_column("Intent", style="bold white", width=18)
        table.add_column("Operation", style="cyan", width=14)
        table.add_column("Risk", width=8)
        table.add_column("Status", width=16)
        table.add_column("Duration", justify="right", width=10)

        # Show most recent records
        slice_records = records[-limit:]
        for idx, rec in enumerate(slice_records, 1):
            ts = rec.get("timestamp", "")
            if "T" in ts:
                ts = ts.split(".")[0].replace("T", " ")
            intent = rec.get("intent", "UNKNOWN")
            op_type = rec.get("operation_type", "-")
            risk = rec.get("risk_level", "LOW")
            status = rec.get("status", "UNKNOWN")
            dur = rec.get("execution_time_ms", 0.0)

            # Styling
            status_style = (
                "[green]SUCCESS[/green]"
                if status == "SUCCESS"
                else (
                    "[yellow]CANCELLED[/yellow]"
                    if "CANCEL" in status
                    else f"[red]{status}[/red]"
                )
            )
            risk_style = (
                "[red]HIGH[/red]"
                if risk == "HIGH"
                else ("[yellow]MED[/yellow]" if risk == "MEDIUM" else "[green]LOW[/green]")
            )

            table.add_row(
                str(idx),
                ts,
                intent,
                op_type,
                risk_style,
                status_style,
                f"{dur:.1f}ms",
            )

        self.console.print(table)

    # --------------------------------------------------------------------------
    # Helper Methods
    # --------------------------------------------------------------------------

    def _get_risk_badge(self, risk_level: RiskLevel) -> str:
        """Returns colored badge for risk level."""
        if risk_level == RiskLevel.HIGH:
            return "[bold red]HIGH (Destructive)[/bold red]"
        elif risk_level == RiskLevel.MEDIUM:
            return "[bold yellow]MEDIUM (State-Modifying)[/bold yellow]"
        return "[bold green]LOW (Read-Only)[/bold green]"

    def _get_handler_info(self, op_type: OperationType) -> str:
        """Returns clear note distinguishing controlled Python handler vs subprocess."""
        if op_type in (
            OperationType.LIST,
            OperationType.CREATE_DIR,
            OperationType.DELETE,
            OperationType.SEARCH,
            OperationType.MOVE,
            OperationType.COPY,
            OperationType.CREATE_FILE,
            OperationType.GET_LOCATION,
        ):
            return "Controlled Python Standard Library (pathlib/shutil - No arbitrary shell execution)"
        elif op_type in (
            OperationType.CHECK_VERSION,
            OperationType.SYSTEM_INFO,
            OperationType.NETWORK_INFO,
        ):
            return "Allowlisted Subprocess (subprocess.run, shell=False, timeout protected)"
        return "Not Executed"
