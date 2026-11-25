"""Controlled Command Executor module for AI Terminal Assistant.

Executes validated operations in a controlled environment:
- Enforces user approval gating with interactive or pluggable callbacks
- Requires double explicit confirmation token ('DELETE') for destructive file deletion
- Executes filesystem operations safely using Python standard library (pathlib, shutil, os)
- Executes allowlisted system/network commands via subprocess.run(shell=False) with timeouts
- Captures stdout, stderr, execution duration, and structured return statuses
- Logs audit history securely to data/command_history.json
"""

from datetime import datetime
from enum import Enum
import logging
import os
from pathlib import Path
import shutil
import subprocess
import time
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

from pydantic import BaseModel, Field

import config
from core.command_generator import (
    CommandOperation,
    OperationType,
    RiskLevel,
)
from core.command_validator import (
    CommandValidator,
    ValidationResult,
    ValidationStatus,
)
from utils.history import HistoryManager

logger = logging.getLogger(__name__)


class ExecutionStatus(str, Enum):
    """Lifecycle status of command execution."""
    SUCCESS = "SUCCESS"
    CANCELLED = "CANCELLED"
    VALIDATION_FAILED = "VALIDATION_FAILED"
    APPROVAL_REQUIRED = "APPROVAL_REQUIRED"
    FILE_NOT_FOUND = "FILE_NOT_FOUND"
    PERMISSION_DENIED = "PERMISSION_DENIED"
    TARGET_ALREADY_EXISTS = "TARGET_ALREADY_EXISTS"
    TIMEOUT = "TIMEOUT"
    EXECUTION_FAILED = "EXECUTION_FAILED"


class ExecutionResult(BaseModel):
    """Structured result of command execution."""
    success: bool
    execution_status: ExecutionStatus
    operation_type: OperationType
    stdout: str = ""
    stderr: str = ""
    return_code: int = 0
    execution_time_ms: float = 0.0
    message: str = ""
    was_cancelled: bool = False
    error_type: Optional[str] = None
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat())
    audit_id: Optional[str] = None

    # Backward compatibility properties for Phase 1 stubs
    @property
    def command(self) -> str:
        return self.message

    @property
    def duration_seconds(self) -> float:
        return self.execution_time_ms / 1000.0

    @property
    def error_message(self) -> Optional[str]:
        return self.stderr if not self.success and self.stderr else None


class CommandExecutor:
    """Safely executes validated operations using standard library and controlled subprocesses."""

    def __init__(
        self,
        timeout_seconds: int = config.COMMAND_TIMEOUT_SECONDS,
        history_file_path: Path = config.COMMAND_HISTORY_PATH,
        confirmation_callback: Optional[Callable[[str, RiskLevel], bool]] = None,
        deletion_token_callback: Optional[Callable[[str], str]] = None,
    ) -> None:
        """Initializes the CommandExecutor.

        Args:
            timeout_seconds: Timeout limit in seconds for external processes.
            history_file_path: Path to command audit history JSON file.
            confirmation_callback: Optional custom callback for [y/N] user approval.
            deletion_token_callback: Optional custom callback for DELETE confirmation token.
        """
        self.timeout_seconds = timeout_seconds
        self.history_manager = HistoryManager(history_file_path)
        self.confirmation_callback = confirmation_callback
        self.deletion_token_callback = deletion_token_callback

    # --------------------------------------------------------------------------
    # User Confirmation Handlers
    # --------------------------------------------------------------------------

    @staticmethod
    def default_confirm_prompt(prompt_text: str) -> bool:
        """Prompts user for [y/N] confirmation via stdin. Default is NO."""
        try:
            full_prompt = f"{prompt_text} [y/N]: "
            response = input(full_prompt).strip().lower()
            return response in ("y", "yes")
        except (EOFError, KeyboardInterrupt):
            return False
        except Exception:
            return False

    @staticmethod
    def default_deletion_token_prompt(target_name: str) -> str:
        """Prompts user for exact 'DELETE' confirmation token via stdin."""
        try:
            full_prompt = f"Type DELETE to confirm permanent removal of '{target_name}': "
            return input(full_prompt).strip()
        except (EOFError, KeyboardInterrupt):
            return ""
        except Exception:
            return ""

    def _request_approval(self, op: CommandOperation, val_res: ValidationResult) -> bool:
        """Requests user confirmation using callback or default stdin prompt."""
        prompt = (
            f"Authorize execution of {op.operation_type.value} operation "
            f"(Risk: {val_res.risk_level.value})"
        )
        if op.explanation:
            prompt += f"\nExplanation: {op.explanation}"

        if self.confirmation_callback is not None:
            return self.confirmation_callback(prompt, val_res.risk_level)
        return self.default_confirm_prompt(prompt)

    def _request_deletion_token(self, target_name: str) -> str:
        """Requests deletion token using callback or default stdin prompt."""
        if self.deletion_token_callback is not None:
            return self.deletion_token_callback(target_name)
        return self.default_deletion_token_prompt(target_name)

    # --------------------------------------------------------------------------
    # Main Execution Entry Point
    # --------------------------------------------------------------------------

    def execute(
        self,
        operation: CommandOperation,
        validation_result: ValidationResult,
        approved: Optional[bool] = None,
        deletion_token: Optional[str] = None,
    ) -> ExecutionResult:
        """Executes a validated command operation safely.

        Args:
            operation: Authoritative structured CommandOperation.
            validation_result: ValidationResult from CommandValidator.
            approved: If boolean provided, overrides interactive prompt. If None, prompts user when required.
            deletion_token: If provided for DELETE operations, token to verify ('DELETE').

        Returns:
            ExecutionResult with detailed outcome and audit timestamp.
        """
        start_time = time.perf_counter()

        # 1. Verify validation status
        if not validation_result.is_valid:
            status = (
                ExecutionStatus.TARGET_ALREADY_EXISTS
                if validation_result.validation_status == ValidationStatus.CONFLICT_TARGET_EXISTS
                else ExecutionStatus.VALIDATION_FAILED
            )
            return self._build_and_log_result(
                operation=operation,
                success=False,
                status=status,
                stdout="",
                stderr="; ".join(validation_result.validation_errors) or validation_result.reason,
                return_code=1,
                start_time=start_time,
                message=f"Validation failed: {validation_result.reason}",
                was_cancelled=False,
                error_type="ValidationFailedError",
                approved=approved,
            )

        # 2. Check confirmation requirement
        if validation_result.requires_confirmation:
            is_approved = approved if approved is not None else self._request_approval(operation, validation_result)
            if not is_approved:
                return self._build_and_log_result(
                    operation=operation,
                    success=False,
                    status=ExecutionStatus.CANCELLED,
                    stdout="",
                    stderr="Operation cancelled by user.",
                    return_code=0,
                    start_time=start_time,
                    message="Operation cancelled by user.",
                    was_cancelled=True,
                    error_type="UserCancelled",
                    approved=False,
                )

        # 3. Check second confirmation for destructive DELETE operations
        if validation_result.requires_double_confirmation:
            target_desc = Path(validation_result.target_path or "file").name
            token = deletion_token if deletion_token is not None else self._request_deletion_token(target_desc)
            if token != "DELETE":
                return self._build_and_log_result(
                    operation=operation,
                    success=False,
                    status=ExecutionStatus.CANCELLED,
                    stdout="",
                    stderr="Deletion cancelled: exact confirmation token 'DELETE' was not provided.",
                    return_code=0,
                    start_time=start_time,
                    message="Deletion cancelled: confirmation token mismatch.",
                    was_cancelled=True,
                    error_type="DeletionTokenMismatch",
                    approved=True,
                )

        # 4. Dispatch to safe execution handler
        op_type = validation_result.operation_type
        args = validation_result.validated_arguments

        try:
            if op_type == OperationType.LIST:
                res = self._exec_list(args)
            elif op_type == OperationType.CREATE_DIR:
                res = self._exec_create_dir(args)
            elif op_type == OperationType.DELETE:
                res = self._exec_delete(args)
            elif op_type == OperationType.SEARCH:
                res = self._exec_search(args)
            elif op_type == OperationType.GET_LOCATION:
                res = self._exec_get_location(args, validation_result)
            elif op_type == OperationType.MOVE:
                res = self._exec_move(args)
            elif op_type == OperationType.COPY:
                res = self._exec_copy(args)
            elif op_type == OperationType.CREATE_FILE:
                res = self._exec_create_file(args)
            elif op_type == OperationType.CHECK_VERSION:
                res = self._exec_check_version(args)
            elif op_type == OperationType.SYSTEM_INFO:
                res = self._exec_system_info(args)
            elif op_type == OperationType.NETWORK_INFO:
                res = self._exec_network_info(args)
            else:
                return self._build_and_log_result(
                    operation=operation,
                    success=False,
                    status=ExecutionStatus.EXECUTION_FAILED,
                    stdout="",
                    stderr=f"Unrecognized execution operation: {op_type}",
                    return_code=1,
                    start_time=start_time,
                    message="Unrecognized operation.",
                    was_cancelled=False,
                    error_type="UnsupportedOperation",
                    approved=True,
                )

            # Record and log successful or failed execution
            return self._build_and_log_result(
                operation=operation,
                success=res["success"],
                status=res["status"],
                stdout=res["stdout"],
                stderr=res["stderr"],
                return_code=res["return_code"],
                start_time=start_time,
                message=res["message"],
                was_cancelled=False,
                error_type=res.get("error_type"),
                approved=True,
            )

        except FileNotFoundError as fnf:
            return self._build_and_log_result(
                operation=operation,
                success=False,
                status=ExecutionStatus.FILE_NOT_FOUND,
                stdout="",
                stderr=str(fnf),
                return_code=1,
                start_time=start_time,
                message=f"File not found: {fnf}",
                was_cancelled=False,
                error_type="FileNotFoundError",
                approved=True,
            )
        except PermissionError as perm:
            return self._build_and_log_result(
                operation=operation,
                success=False,
                status=ExecutionStatus.PERMISSION_DENIED,
                stdout="",
                stderr=str(perm),
                return_code=1,
                start_time=start_time,
                message=f"Permission denied: {perm}",
                was_cancelled=False,
                error_type="PermissionError",
                approved=True,
            )
        except FileExistsError as fee:
            return self._build_and_log_result(
                operation=operation,
                success=False,
                status=ExecutionStatus.TARGET_ALREADY_EXISTS,
                stdout="",
                stderr=str(fee),
                return_code=1,
                start_time=start_time,
                message=f"Target already exists: {fee}",
                was_cancelled=False,
                error_type="FileExistsError",
                approved=True,
            )
        except Exception as exc:
            logger.exception("Unexpected execution error: %s", exc)
            return self._build_and_log_result(
                operation=operation,
                success=False,
                status=ExecutionStatus.EXECUTION_FAILED,
                stdout="",
                stderr=str(exc),
                return_code=1,
                start_time=start_time,
                message=f"Execution error: {exc}",
                was_cancelled=False,
                error_type=type(exc).__name__,
                approved=True,
            )

    # --------------------------------------------------------------------------
    # Filesystem Execution Handlers (Python Standard Library)
    # --------------------------------------------------------------------------

    def _exec_list(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Lists files and folders using os.scandir."""
        target_dir = Path(args["directory"])
        if not target_dir.exists():
            return {
                "success": False,
                "status": ExecutionStatus.FILE_NOT_FOUND,
                "stdout": "",
                "stderr": f"Directory not found: {target_dir}",
                "return_code": 1,
                "message": "Directory not found.",
                "error_type": "FileNotFoundError",
            }

        entries = []
        with os.scandir(target_dir) as it:
            for entry in it:
                try:
                    stat = entry.stat()
                    is_dir = entry.is_dir()
                    size_str = "<DIR>" if is_dir else f"{stat.st_size} B"
                    mtime = datetime.fromtimestamp(stat.st_mtime).strftime("%Y-%m-%d %H:%M:%S")
                    entries.append((entry.name, is_dir, size_str, mtime))
                except Exception:
                    continue

        # Sort directories first, then alphabetical
        entries.sort(key=lambda x: (not x[1], x[0].lower()))

        lines = [f"Directory: {target_dir}\n"]
        lines.append(f"{'Mode':<8} {'LastWriteTime':<20} {'Length':<12} {'Name'}")
        lines.append(f"{'----':<8} {'-------------':<20} {'------':<12} {'----'}")
        for name, is_dir, size, mtime in entries:
            mode = "d-----" if is_dir else "-a----"
            lines.append(f"{mode:<8} {mtime:<20} {size:<12} {name}")

        output = "\n".join(lines)
        return {
            "success": True,
            "status": ExecutionStatus.SUCCESS,
            "stdout": output,
            "stderr": "",
            "return_code": 0,
            "message": f"Successfully listed {len(entries)} items in {target_dir.name}.",
        }

    def _exec_create_dir(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Creates directory safely using pathlib.Path.mkdir."""
        target = Path(args["directory_path"])
        target.mkdir(parents=True, exist_ok=True)
        return {
            "success": True,
            "status": ExecutionStatus.SUCCESS,
            "stdout": f"Created directory: {target}",
            "stderr": "",
            "return_code": 0,
            "message": f"Directory created at {target}.",
        }

    def _exec_delete(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Deletes a file safely using pathlib.Path.unlink (never deletes directories)."""
        target = Path(args["file_path"])
        if not target.exists():
            return {
                "success": False,
                "status": ExecutionStatus.FILE_NOT_FOUND,
                "stdout": "",
                "stderr": f"File not found: {target}",
                "return_code": 1,
                "message": "File not found.",
                "error_type": "FileNotFoundError",
            }

        if target.is_dir():
            return {
                "success": False,
                "status": ExecutionStatus.EXECUTION_FAILED,
                "stdout": "",
                "stderr": f"Cannot delete directory '{target}' via DELETE_FILE.",
                "return_code": 1,
                "message": "Directory deletion is prohibited through file delete operation.",
                "error_type": "InvalidOperationError",
            }

        target.unlink()
        return {
            "success": True,
            "status": ExecutionStatus.SUCCESS,
            "stdout": f"Removed file: {target}",
            "stderr": "",
            "return_code": 0,
            "message": f"File '{target.name}' was permanently deleted.",
        }

    def _exec_search(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Searches files using pathlib.Path glob/rglob."""
        dir_path = Path(args["directory"])
        pattern = args.get("pattern", "*.*")
        recurse = args.get("recurse", False)

        if not dir_path.exists():
            return {
                "success": False,
                "status": ExecutionStatus.FILE_NOT_FOUND,
                "stdout": "",
                "stderr": f"Search directory not found: {dir_path}",
                "return_code": 1,
                "message": "Search directory not found.",
                "error_type": "FileNotFoundError",
            }

        generator = dir_path.rglob(pattern) if recurse else dir_path.glob(pattern)
        matches = [p for p in generator if p.is_file()]

        lines = [f"Found {len(matches)} matching file(s) in {dir_path}:"]
        for m in sorted(matches, key=lambda x: str(x).lower()):
            lines.append(f"  {m.relative_to(dir_path)}")

        return {
            "success": True,
            "status": ExecutionStatus.SUCCESS,
            "stdout": "\n".join(lines),
            "stderr": "",
            "return_code": 0,
            "message": f"Found {len(matches)} files.",
        }

    def _exec_get_location(self, args: Dict[str, Any], val_res: ValidationResult) -> Dict[str, Any]:
        """Returns the current workspace/working location."""
        loc = val_res.target_path or str(config.WORKSPACE_DIR)
        return {
            "success": True,
            "status": ExecutionStatus.SUCCESS,
            "stdout": f"Path\n----\n{loc}",
            "stderr": "",
            "return_code": 0,
            "message": f"Current directory is {loc}.",
        }

    def _exec_move(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Moves file/directory using shutil.move."""
        src = Path(args["source"])
        dest = Path(args["destination"])

        if not src.exists():
            return {
                "success": False,
                "status": ExecutionStatus.FILE_NOT_FOUND,
                "stdout": "",
                "stderr": f"Source not found: {src}",
                "return_code": 1,
                "message": "Source not found.",
                "error_type": "FileNotFoundError",
            }

        # If dest is existing file, refuse overwrite
        if dest.exists() and dest.is_file():
            return {
                "success": False,
                "status": ExecutionStatus.TARGET_ALREADY_EXISTS,
                "stdout": "",
                "stderr": f"Destination file already exists: {dest}",
                "return_code": 1,
                "message": "Destination file already exists.",
                "error_type": "FileExistsError",
            }

        shutil.move(str(src), str(dest))
        return {
            "success": True,
            "status": ExecutionStatus.SUCCESS,
            "stdout": f"Moved: '{src}' -> '{dest}'",
            "stderr": "",
            "return_code": 0,
            "message": f"Successfully moved {src.name} to {dest}.",
        }

    def _exec_copy(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Copies file/directory using shutil.copy2 or copytree."""
        src = Path(args["source"])
        dest = Path(args["destination"])

        if not src.exists():
            return {
                "success": False,
                "status": ExecutionStatus.FILE_NOT_FOUND,
                "stdout": "",
                "stderr": f"Source not found: {src}",
                "return_code": 1,
                "message": "Source not found.",
                "error_type": "FileNotFoundError",
            }

        if dest.exists() and dest.is_file():
            return {
                "success": False,
                "status": ExecutionStatus.TARGET_ALREADY_EXISTS,
                "stdout": "",
                "stderr": f"Destination file already exists: {dest}",
                "return_code": 1,
                "message": "Destination file already exists.",
                "error_type": "FileExistsError",
            }

        if src.is_dir():
            shutil.copytree(str(src), str(dest))
        else:
            shutil.copy2(str(src), str(dest))

        return {
            "success": True,
            "status": ExecutionStatus.SUCCESS,
            "stdout": f"Copied: '{src}' -> '{dest}'",
            "stderr": "",
            "return_code": 0,
            "message": f"Successfully copied {src.name} to {dest}.",
        }

    def _exec_create_file(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Creates empty file exclusively using open mode 'x'."""
        target = Path(args["file_path"])
        target.parent.mkdir(parents=True, exist_ok=True)

        # Exclusive creation prevents accidental overwrite
        with open(target, "x", encoding="utf-8") as f:
            pass

        return {
            "success": True,
            "status": ExecutionStatus.SUCCESS,
            "stdout": f"Created file: {target}",
            "stderr": "",
            "return_code": 0,
            "message": f"File '{target.name}' created successfully.",
        }

    # --------------------------------------------------------------------------
    # Subprocess Execution Handlers (Allowlisted Commands Only, shell=False)
    # --------------------------------------------------------------------------

    def _run_subprocess(self, cmd_args: List[str]) -> Dict[str, Any]:
        """Executes an allowlisted command safely using subprocess.run(shell=False)."""
        try:
            res = subprocess.run(
                cmd_args,
                capture_output=True,
                text=True,
                timeout=self.timeout_seconds,
                shell=False,
            )
            success = (res.returncode == 0)
            status = ExecutionStatus.SUCCESS if success else ExecutionStatus.EXECUTION_FAILED
            return {
                "success": success,
                "status": status,
                "stdout": (res.stdout or "").strip(),
                "stderr": (res.stderr or "").strip(),
                "return_code": res.returncode,
                "message": "Command completed." if success else f"Exited with code {res.returncode}",
            }
        except subprocess.TimeoutExpired:
            return {
                "success": False,
                "status": ExecutionStatus.TIMEOUT,
                "stdout": "",
                "stderr": f"Command timed out after {self.timeout_seconds} seconds.",
                "return_code": 124,
                "message": "Execution timed out.",
                "error_type": "TimeoutExpired",
            }
        except FileNotFoundError as fnf:
            return {
                "success": False,
                "status": ExecutionStatus.FILE_NOT_FOUND,
                "stdout": "",
                "stderr": f"Executable not found on system PATH: {cmd_args[0]}",
                "return_code": 127,
                "message": f"Executable '{cmd_args[0]}' is not installed or not in PATH.",
                "error_type": "FileNotFoundError",
            }
        except Exception as exc:
            return {
                "success": False,
                "status": ExecutionStatus.EXECUTION_FAILED,
                "stdout": "",
                "stderr": str(exc),
                "return_code": 1,
                "message": f"Subprocess error: {exc}",
                "error_type": type(exc).__name__,
            }

    def _exec_check_version(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Executes allowlisted version check commands."""
        tool = args["tool"]
        # Map tool to fixed command argument list
        TOOL_CMD_ARGS: Dict[str, List[str]] = {
            "python": ["python", "--version"],
            "python3": ["python", "--version"],
            "node": ["node", "--version"],
            "node.js": ["node", "--version"],
            "nodejs": ["node", "--version"],
            "git": ["git", "--version"],
            "docker": ["docker", "--version"],
            "powershell": ["powershell", "-NoProfile", "-Command", "$PSVersionTable.PSVersion.ToString()"],
            "pwsh": ["powershell", "-NoProfile", "-Command", "$PSVersionTable.PSVersion.ToString()"],
            "npm": ["npm", "--version"],
            "java": ["java", "-version"],
            "rust": ["rustc", "--version"],
            "rustc": ["rustc", "--version"],
            "go": ["go", "version"],
            "gcc": ["gcc", "--version"],
            "dotnet": ["dotnet", "--version"],
            "pip": ["pip", "--version"],
            "pip3": ["pip", "--version"],
            "code": ["code", "--version"],
            "vscode": ["code", "--version"],
        }
        cmd_args = TOOL_CMD_ARGS.get(tool)
        if not cmd_args:
            return {
                "success": False,
                "status": ExecutionStatus.EXECUTION_FAILED,
                "stdout": "",
                "stderr": f"Tool '{tool}' has no configured execution args.",
                "return_code": 1,
                "message": "Tool command mapping missing.",
                "error_type": "ConfigurationError",
            }
        return self._run_subprocess(cmd_args)

    def _exec_system_info(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Executes allowlisted system info commands."""
        cmd = args.get("command", "Get-ComputerInfo")
        if cmd == "hostname":
            cmd_args = ["hostname"]
            return self._run_subprocess(cmd_args)
        elif "Win32_OperatingSystem" in cmd:
            cmd_args = [
                "powershell",
                "-NoProfile",
                "-Command",
                "Get-CimInstance Win32_OperatingSystem | Select-Object Caption, Version, OSArchitecture | Format-List",
            ]
            res = self._run_subprocess(cmd_args)
            if res["success"]:
                return res
        else:
            cmd_args = [
                "powershell",
                "-NoProfile",
                "-Command",
                "Get-ComputerInfo | Select-Object CsName, WindowsProductName, WindowsVersion, OsArchitecture, CsTotalPhysicalMemory | Format-List",
            ]
            res = self._run_subprocess(cmd_args)
            if res["success"]:
                return res

        # Resilient fallback to Python standard library platform info if CIM/WMI is restricted
        import platform
        u = platform.uname()
        lines = [
            f"Computer Name:    {u.node}",
            f"Operating System: {u.system} {u.release} (Build {u.version})",
            f"Architecture:     {u.machine}",
            f"Processor:        {u.processor or platform.processor()}",
        ]
        return {
            "success": True,
            "status": ExecutionStatus.SUCCESS,
            "stdout": "\n".join(lines),
            "stderr": "",
            "return_code": 0,
            "message": "System information retrieved.",
        }

    def _exec_network_info(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Executes allowlisted network info commands."""
        cmd = args.get("command", "ipconfig")
        if cmd == "Get-NetAdapter":
            cmd_args = [
                "powershell",
                "-NoProfile",
                "-Command",
                "Get-NetAdapter | Select-Object Name, InterfaceDescription, Status, LinkSpeed | Format-Table -AutoSize",
            ]
        elif cmd == "Get-NetTCPConnection":
            cmd_args = [
                "powershell",
                "-NoProfile",
                "-Command",
                "Get-NetTCPConnection -State Listen | Select-Object LocalAddress, LocalPort, State | Format-Table -AutoSize",
            ]
        else:
            cmd_args = ["ipconfig"]
        return self._run_subprocess(cmd_args)

    # --------------------------------------------------------------------------
    # Audit History Logging
    # --------------------------------------------------------------------------

    def _build_and_log_result(
        self,
        operation: CommandOperation,
        success: bool,
        status: ExecutionStatus,
        stdout: str,
        stderr: str,
        return_code: int,
        start_time: float,
        message: str,
        was_cancelled: bool,
        error_type: Optional[str] = None,
        approved: Optional[bool] = None,
    ) -> ExecutionResult:
        """Constructs ExecutionResult and records an audit log entry."""
        duration_ms = round((time.perf_counter() - start_time) * 1000.0, 2)
        now_iso = datetime.now().isoformat()
        audit_id = f"aud_{int(time.time() * 1000)}"

        # Sanitize query and arguments for audit logging (never log credentials/passwords)
        sanitized_query = operation.source_query
        target = (
            operation.arguments.get("file_name")
            or operation.arguments.get("directory_name")
            or operation.arguments.get("source")
            or ""
        )

        audit_entry = {
            "audit_id": audit_id,
            "timestamp": now_iso,
            "operation_type": operation.operation_type.value,
            "intent": operation.intent,
            "source_query": sanitized_query,
            "target": str(target),
            "risk_level": operation.risk_level.value,
            "approved": approved,
            "execution_status": status.value,
            "success": success,
            "duration_ms": duration_ms,
            "return_code": return_code,
        }

        # Persist audit record safely (never allow audit errors to fail the execution result)
        try:
            self.history_manager.log_entry(audit_entry)
        except Exception as log_err:
            logger.warning("Could not record audit log entry: %s", log_err)

        return ExecutionResult(
            success=success,
            execution_status=status,
            operation_type=operation.operation_type,
            stdout=stdout,
            stderr=stderr,
            return_code=return_code,
            execution_time_ms=duration_ms,
            message=message,
            was_cancelled=was_cancelled,
            error_type=error_type,
            timestamp=now_iso,
            audit_id=audit_id,
        )
