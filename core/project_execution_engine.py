"""Controlled Project Command Execution Engine (Phase 13).

Consumes Phase 12 ProjectCommandRecommendation objects, validates boundaries,
parses commands into structured argument arrays, generates an explicit preview,
requires bound one-time user approval, executes via subprocess.run(shell=False),
and records every attempt in the audit log.

STRICT SECURITY POLICY:
- No execution without explicit, bound, unexpired, one-time-use approval.
- Never uses shell=True, cmd.exe, PowerShell, or arbitrary shell strings.
- HIGH-risk operations are blocked by default.
- shell=False is not an OS-level sandbox; approved project scripts execute
  project-controlled code and are explicitly flagged with warnings.
"""

import ctypes
from ctypes import wintypes
from datetime import datetime, timedelta
from enum import Enum
import hashlib
import hmac
import json
import logging
import os
from pathlib import Path
import re
import secrets
import shlex
import shutil
import subprocess
import sys
import threading
import time
import uuid
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union

from pydantic import BaseModel, Field

import config
from core.project_command_engine import (
    ProjectCommandRecommendation,
    RecommendedCommand,
    EvidenceCategory,
    CommandRiskLevel,
)
from utils.history import HistoryManager

logger = logging.getLogger(__name__)

DEFAULT_PREVIEW_TTL_SECONDS = 300
DEFAULT_MAX_OUTPUT_CHARS = 65536


class ExecutionStatus(str, Enum):
    """Lifecycle status of a project command execution."""
    PREVIEW_CREATED = "PREVIEW_CREATED"
    APPROVAL_REQUIRED = "APPROVAL_REQUIRED"
    APPROVED = "APPROVED"
    RUNNING = "RUNNING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    TIMED_OUT = "TIMED_OUT"
    REJECTED = "REJECTED"
    BLOCKED = "BLOCKED"


class ExecutionRiskLevel(str, Enum):
    """Risk classification for controlled project execution."""
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"


class ProjectExecutionRequest(BaseModel):
    """Input request for preparing a controlled project command execution."""
    recommendation: ProjectCommandRecommendation
    authorized_project_root: str
    selected_command_index: int = 0
    timeout_seconds: int = Field(default=config.COMMAND_TIMEOUT_SECONDS, ge=1, le=600)


class ExecutionPreview(BaseModel):
    """Immutable-bound preview presented to the user before approval."""
    preview_id: str
    preview_hash: str
    created_at: str
    expires_at: str
    user_request: str
    project_name: Optional[str] = None
    authorized_project_root: str
    command: str
    executable: str
    resolved_executable: Optional[str] = None
    arguments: List[str] = Field(default_factory=list)
    working_directory: str
    evidence_source: str
    evidence_category: EvidenceCategory
    risk_level: ExecutionRiskLevel
    expected_operation: str
    executes_project_code: bool = False
    warnings: List[str] = Field(default_factory=list)
    timeout_seconds: int = config.COMMAND_TIMEOUT_SECONDS
    approval_required: bool = True
    status: ExecutionStatus = ExecutionStatus.PREVIEW_CREATED
    is_blocked: bool = False
    block_reason: Optional[str] = None


class ExecutionApproval(BaseModel):
    """Explicit, operation-specific, one-time-use approval bound to an ExecutionPreview."""
    preview_id: str
    preview_hash: str
    approved: bool = False
    approved_executable: str
    approved_arguments: List[str] = Field(default_factory=list)
    approved_working_directory: str
    approved_project_root: str
    approved_risk_level: ExecutionRiskLevel
    approved_at: str = Field(default_factory=lambda: datetime.now().isoformat())
    approval_token: str = ""
    auth_method: str = "NONE"
    is_generic: bool = False


class ProjectExecutionResult(BaseModel):
    """Structured result of a controlled project execution attempt."""
    execution_id: str
    status: ExecutionStatus
    project_root: str
    command: str
    working_directory: str
    exit_code: Optional[int] = None
    stdout: str = ""
    stderr: str = ""
    duration_seconds: float = 0.0
    timed_out: bool = False
    output_truncated: bool = False
    stdout_truncated: bool = False
    stderr_truncated: bool = False
    approval_verified: bool = False
    risk_level: ExecutionRiskLevel = ExecutionRiskLevel.LOW
    warnings: List[str] = Field(default_factory=list)
    error_message: Optional[str] = None


# Allowlisted executable base names (lowercase, without .exe)
DEFAULT_ALLOWED_EXECUTABLES: Set[str] = {
    "python", "python3", "py",
    "node", "npm", "pnpm", "yarn", "bun",
    "git",
    "mvn", "gradle", "java",
    "cmake", "make", "meson",
    "flutter", "dart",
    "dotnet",
    "docker",
    "pytest", "pip", "pipenv", "poetry", "uvicorn",
}

# Shells and interpreters that must NEVER be invoked directly
BLOCKED_SHELL_EXECUTABLES: Set[str] = {
    "cmd", "cmd.exe", "powershell", "powershell.exe", "pwsh", "pwsh.exe",
    "bash", "sh", "zsh", "fish", "wsl", "wsl.exe", "wscript", "cscript",
}

# Forbidden shell metacharacters / operators
_FORBIDDEN_SHELL_PATTERNS: List[Tuple[re.Pattern, str]] = [
    (re.compile(r"&&"), "Compound operator '&&' is not permitted."),
    (re.compile(r"\|\|"), "Compound operator '||' is not permitted."),
    (re.compile(r"\|"), "Pipe operator '|' is not permitted."),
    (re.compile(r";"), "Command separator ';' is not permitted."),
    (re.compile(r">>|>|<"), "I/O redirection ('>', '>>', '<') is not permitted."),
    (re.compile(r"\$\("), "Command substitution '$()' is not permitted."),
    (re.compile(r"`"), "Backtick command substitution is not permitted."),
    (re.compile(r"\$\{[^}]*\}"), "Variable expansion '${...}' is not permitted."),
    (re.compile(r"[\r\n]"), "Multiline shell expressions are not permitted."),
]

_HIGH_RISK_KEYWORDS: Set[str] = {
    "migrate", "migration", "reset", "drop", "destroy", "delete",
    "clean", "purge", "prune", "deploy", "publish", "release", "wipe",
}

_SAFE_ENV_KEYS: Set[str] = {
    "PATH", "SYSTEMROOT", "WINDIR", "TEMP", "TMP", "USERPROFILE", "HOME",
    "HOMEDRIVE", "HOMEPATH", "PATHEXT", "COMSPEC", "APPDATA", "LOCALAPPDATA",
    "VIRTUAL_ENV", "PYTHONPATH", "LANG", "LC_ALL",
}

_SECRET_REDACTION_PATTERNS: List[Tuple[re.Pattern, str]] = [
    (re.compile(r"(?i)(api[_-]?key|secret|token|password|passwd|credential)\s*[:=]\s*([^\s'\"&;]+)"), r"\1=[REDACTED]"),
    (re.compile(r"(?i)(Bearer\s+)[A-Za-z0-9\-._~+/]+=*"), r"\1[REDACTED]"),
    (re.compile(r"\b(sk-[A-Za-z0-9]{12,}|ghp_[A-Za-z0-9]{12,})\b"), "[REDACTED_TOKEN]"),
]


class _WindowsJobObject:
    """Manages a Windows Job Object to terminate child process trees on timeout."""

    JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE = 0x00002000
    JobObjectExtendedLimitInformation = 9

    def __init__(self) -> None:
        self.handle: Optional[int] = None
        self.error: Optional[str] = None
        if os.name != "nt":
            return
        try:
            kernel32 = ctypes.windll.kernel32
            self.handle = kernel32.CreateJobObjectW(None, None)
            if not self.handle:
                self.error = "CreateJobObjectW returned NULL."
                return

            class IO_COUNTERS(ctypes.Structure):
                _fields_ = [
                    ("ReadOperationCount", ctypes.c_uint64),
                    ("WriteOperationCount", ctypes.c_uint64),
                    ("OtherOperationCount", ctypes.c_uint64),
                    ("ReadTransferCount", ctypes.c_uint64),
                    ("WriteTransferCount", ctypes.c_uint64),
                    ("OtherTransferCount", ctypes.c_uint64),
                ]

            class JOBOBJECT_BASIC_LIMIT_INFORMATION(ctypes.Structure):
                _fields_ = [
                    ("PerProcessUserTimeLimit", ctypes.c_int64),
                    ("PerJobUserTimeLimit", ctypes.c_int64),
                    ("LimitFlags", wintypes.DWORD),
                    ("MinimumWorkingSetSize", ctypes.c_size_t),
                    ("MaximumWorkingSetSize", ctypes.c_size_t),
                    ("ActiveProcessLimit", wintypes.DWORD),
                    ("Affinity", ctypes.c_size_t),
                    ("PriorityClass", wintypes.DWORD),
                    ("SchedulingClass", wintypes.DWORD),
                ]

            class JOBOBJECT_EXTENDED_LIMIT_INFORMATION(ctypes.Structure):
                _fields_ = [
                    ("BasicLimitInformation", JOBOBJECT_BASIC_LIMIT_INFORMATION),
                    ("IoInfo", IO_COUNTERS),
                    ("ProcessMemoryLimit", ctypes.c_size_t),
                    ("JobMemoryLimit", ctypes.c_size_t),
                    ("PeakProcessMemoryUsed", ctypes.c_size_t),
                    ("PeakJobMemoryUsed", ctypes.c_size_t),
                ]

            info = JOBOBJECT_EXTENDED_LIMIT_INFORMATION()
            info.BasicLimitInformation.LimitFlags = self.JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
            ok = kernel32.SetInformationJobObject(
                self.handle,
                self.JobObjectExtendedLimitInformation,
                ctypes.byref(info),
                ctypes.sizeof(info),
            )
            if not ok:
                self.error = "SetInformationJobObject failed."
        except Exception as exc:
            self.error = f"Job Object initialization error: {exc}"
            self.handle = None

    def assign(self, proc: subprocess.Popen) -> bool:
        if os.name != "nt" or not self.handle:
            return False
        try:
            proc_handle = getattr(proc, "_handle", None)
            if not proc_handle:
                self.error = "Process handle unavailable for Job Object assignment."
                return False
            ok = ctypes.windll.kernel32.AssignProcessToJobObject(self.handle, int(proc_handle))
            if not ok:
                self.error = "AssignProcessToJobObject failed (nested job restriction or access denied)."
                return False
            return True
        except Exception as exc:
            self.error = f"AssignProcessToJobObject error: {exc}"
            return False

    def terminate_tree(self) -> Optional[str]:
        """Terminates all processes in the job object. Returns an error string if termination fails."""
        if os.name != "nt" or not self.handle:
            return self.error
        try:
            ok = ctypes.windll.kernel32.TerminateJobObject(self.handle, 1)
            if not ok:
                return "TerminateJobObject failed."
            return None
        except Exception as exc:
            return f"TerminateJobObject error: {exc}"

    def close(self) -> None:
        if os.name == "nt" and self.handle:
            try:
                ctypes.windll.kernel32.CloseHandle(self.handle)
            except Exception:
                pass
            self.handle = None


class ProjectExecutionEngine:
    """Controlled, approval-gated execution engine for Phase 12 recommendations."""

    def __init__(
        self,
        timeout_seconds: int = config.COMMAND_TIMEOUT_SECONDS,
        max_output_chars: int = DEFAULT_MAX_OUTPUT_CHARS,
        preview_ttl_seconds: int = DEFAULT_PREVIEW_TTL_SECONDS,
        history_file_path: Path = config.COMMAND_HISTORY_PATH,
        allowed_executables: Optional[Set[str]] = None,
        user_approval_callback: Optional[Callable[[ExecutionPreview], bool]] = None,
        require_trusted_callback: bool = False,
    ) -> None:
        self.timeout_seconds = timeout_seconds
        self.max_output_chars = max_output_chars
        self.preview_ttl_seconds = preview_ttl_seconds
        self.history_manager = HistoryManager(history_file_path)
        self.allowed_executables = (
            {e.lower() for e in allowed_executables}
            if allowed_executables is not None
            else set(DEFAULT_ALLOWED_EXECUTABLES)
        )
        self.user_approval_callback = user_approval_callback
        self.require_trusted_callback = require_trusted_callback
        self._auth_secret: bytes = secrets.token_bytes(32)
        self._protected_dirs = [
            p.resolve() for p in getattr(config, "PROTECTED_SYSTEM_DIRS", [])
        ] + [
            p.resolve() for p in getattr(config, "PROTECTED_ROOT_DIRS", [])
        ]
        self._issued_previews: Dict[str, ExecutionPreview] = {}
        self._consumed_preview_ids: Set[str] = set()

    # --------------------------------------------------------------------------
    # Step 1–6: Prepare Execution & Generate Bound Preview (Zero Execution)
    # --------------------------------------------------------------------------

    def prepare_execution(
        self,
        recommendation: ProjectCommandRecommendation,
        authorized_project_root: Union[str, Path],
        selected_command_index: int = 0,
        timeout_seconds: Optional[int] = None,
    ) -> ExecutionPreview:
        """Validates a Phase 12 recommendation and builds a bound ExecutionPreview."""
        eff_timeout = timeout_seconds if timeout_seconds is not None else self.timeout_seconds
        now = datetime.now()
        expires = now + timedelta(seconds=self.preview_ttl_seconds)
        preview_id = str(uuid.uuid4())
        warnings: List[str] = list(recommendation.warnings)

        # Validate recommendation has commands
        if not recommendation.recommended_commands or selected_command_index >= len(recommendation.recommended_commands) or selected_command_index < 0:
            return self._blocked_preview(
                preview_id, now, expires, recommendation, str(authorized_project_root),
                command="", executable="", arguments=[], working_directory=str(authorized_project_root),
                evidence_source="", evidence_category=EvidenceCategory.INFERRED,
                risk_level=ExecutionRiskLevel.HIGH, executes_project_code=False,
                warnings=warnings, timeout_seconds=eff_timeout,
                reason="No valid recommended command is available for execution.",
            )

        rec_cmd: RecommendedCommand = recommendation.recommended_commands[selected_command_index]
        if not rec_cmd.evidence_source:
            return self._blocked_preview(
                preview_id, now, expires, recommendation, str(authorized_project_root),
                command=rec_cmd.command, executable="", arguments=[], working_directory=rec_cmd.working_directory,
                evidence_source="", evidence_category=rec_cmd.evidence_category,
                risk_level=ExecutionRiskLevel.HIGH, executes_project_code=False,
                warnings=warnings, timeout_seconds=eff_timeout,
                reason="Missing evidence source in recommendation.",
            )

        # Validate authorized project root & working directory boundaries
        root_ok, resolved_root, root_err = self._validate_directory_boundary(
            Path(authorized_project_root), parent_root=None
        )
        if not root_ok or resolved_root is None:
            return self._blocked_preview(
                preview_id, now, expires, recommendation, str(authorized_project_root),
                command=rec_cmd.command, executable="", arguments=[], working_directory=rec_cmd.working_directory,
                evidence_source=rec_cmd.evidence_source, evidence_category=rec_cmd.evidence_category,
                risk_level=ExecutionRiskLevel.HIGH, executes_project_code=False,
                warnings=warnings, timeout_seconds=eff_timeout,
                reason=root_err or "Invalid authorized project root.",
            )

        # Ensure recommendation's project_root matches authorized_project_root
        rec_root_ok, resolved_rec_root, _ = self._validate_directory_boundary(
            Path(recommendation.project_root), parent_root=resolved_root
        )
        if not rec_root_ok or resolved_rec_root is None:
            return self._blocked_preview(
                preview_id, now, expires, recommendation, str(resolved_root),
                command=rec_cmd.command, executable="", arguments=[], working_directory=rec_cmd.working_directory,
                evidence_source=rec_cmd.evidence_source, evidence_category=rec_cmd.evidence_category,
                risk_level=ExecutionRiskLevel.HIGH, executes_project_code=False,
                warnings=warnings, timeout_seconds=eff_timeout,
                reason="Recommendation project_root is outside the authorized project root.",
            )

        wd_ok, resolved_wd, wd_err = self._validate_directory_boundary(
            Path(rec_cmd.working_directory), parent_root=resolved_root
        )
        if not wd_ok or resolved_wd is None:
            return self._blocked_preview(
                preview_id, now, expires, recommendation, str(resolved_root),
                command=rec_cmd.command, executable="", arguments=[], working_directory=rec_cmd.working_directory,
                evidence_source=rec_cmd.evidence_source, evidence_category=rec_cmd.evidence_category,
                risk_level=ExecutionRiskLevel.HIGH, executes_project_code=False,
                warnings=warnings, timeout_seconds=eff_timeout,
                reason=wd_err or "Working directory is outside the authorized project boundary.",
            )

        # Parse & validate command into structured executable + args + resolved path
        parse_ok, exe, resolved_exe, args, parse_err = self._parse_and_validate_command(
            rec_cmd.command, resolved_root, resolved_wd
        )
        if not parse_ok:
            return self._blocked_preview(
                preview_id, now, expires, recommendation, str(resolved_root),
                command=rec_cmd.command, executable=exe, arguments=args, working_directory=str(resolved_wd),
                evidence_source=rec_cmd.evidence_source, evidence_category=rec_cmd.evidence_category,
                risk_level=ExecutionRiskLevel.HIGH, executes_project_code=False,
                warnings=warnings, timeout_seconds=eff_timeout,
                reason=parse_err or "Command rejected by parser or executable policy.",
            )

        # Assess risk & project-defined code execution
        risk_level, executes_project_code = self._assess_risk(rec_cmd, exe, args)
        if executes_project_code:
            code_warn = (
                "Project-defined code warning: this operation executes local project code "
                "which is not sandboxed by the OS."
            )
            if code_warn not in warnings:
                warnings.append(code_warn)

        if risk_level == ExecutionRiskLevel.HIGH:
            return self._blocked_preview(
                preview_id, now, expires, recommendation, str(resolved_root),
                command=rec_cmd.command, executable=exe, arguments=args, working_directory=str(resolved_wd),
                evidence_source=rec_cmd.evidence_source, evidence_category=rec_cmd.evidence_category,
                risk_level=ExecutionRiskLevel.HIGH, executes_project_code=executes_project_code,
                warnings=warnings, timeout_seconds=eff_timeout,
                reason="HIGH-risk operations (migrations, resets, cleanup, deletion, deployment) are blocked by default.",
            )

        preview_hash = self._compute_preview_hash(
            preview_id=preview_id,
            executable=exe,
            arguments=args,
            working_directory=str(resolved_wd),
            authorized_project_root=str(resolved_root),
            risk_level=risk_level,
        )

        preview = ExecutionPreview(
            preview_id=preview_id,
            preview_hash=preview_hash,
            created_at=now.isoformat(),
            expires_at=expires.isoformat(),
            user_request=recommendation.user_request,
            project_name=recommendation.project_name,
            authorized_project_root=str(resolved_root),
            command=rec_cmd.command,
            executable=exe,
            resolved_executable=resolved_exe,
            arguments=args,
            working_directory=str(resolved_wd),
            evidence_source=rec_cmd.evidence_source,
            evidence_category=rec_cmd.evidence_category,
            risk_level=risk_level,
            expected_operation=rec_cmd.explanation,
            executes_project_code=executes_project_code,
            warnings=warnings,
            timeout_seconds=eff_timeout,
            approval_required=True,
            status=ExecutionStatus.PREVIEW_CREATED,
            is_blocked=False,
        )
        self._issued_previews[preview_id] = preview
        return preview

    def _compute_approval_hmac(self, preview: ExecutionPreview) -> str:
        """Computes an instance-private HMAC-SHA256 authentication token for a preview."""
        msg = json.dumps(
            {
                "preview_id": preview.preview_id,
                "preview_hash": preview.preview_hash,
                "executable": preview.executable,
                "arguments": preview.arguments,
                "working_directory": preview.working_directory,
                "authorized_project_root": preview.authorized_project_root,
                "risk_level": preview.risk_level.value,
            },
            sort_keys=True,
        ).encode("utf-8")
        return hmac.new(self._auth_secret, msg, hashlib.sha256).hexdigest()

    def authorize_preview(
        self,
        preview: ExecutionPreview,
        user_confirm_callback: Optional[Callable[[ExecutionPreview], bool]] = None,
    ) -> ExecutionApproval:
        """Trusted application-controlled user-interaction boundary for approval.

        Requires a user confirmation callback returning True before minting
        the HMAC-authenticated ExecutionApproval.
        """
        cb = user_confirm_callback or self.user_approval_callback
        if cb is None:
            return ExecutionApproval(
                preview_id=preview.preview_id,
                preview_hash=preview.preview_hash,
                approved=False,
                approved_executable=preview.executable,
                approved_arguments=list(preview.arguments),
                approved_working_directory=preview.working_directory,
                approved_project_root=preview.authorized_project_root,
                approved_risk_level=preview.risk_level,
                approval_token="",
                auth_method="NO_CALLBACK",
                is_generic=False,
            )

        confirmed = False
        try:
            confirmed = bool(cb(preview))
        except Exception:
            confirmed = False

        token = self._compute_approval_hmac(preview) if confirmed else ""
        return ExecutionApproval(
            preview_id=preview.preview_id,
            preview_hash=preview.preview_hash,
            approved=confirmed,
            approved_executable=preview.executable,
            approved_arguments=list(preview.arguments),
            approved_working_directory=preview.working_directory,
            approved_project_root=preview.authorized_project_root,
            approved_risk_level=preview.risk_level,
            approval_token=token,
            auth_method="TRUSTED_CALLBACK" if confirmed else "DENIED_BY_USER",
            is_generic=False,
        )

    def create_approval(self, preview: ExecutionPreview, approved: bool = True) -> ExecutionApproval:
        """Constructs an HMAC-authenticated ExecutionApproval bound to a specific ExecutionPreview.

        If require_trusted_callback=True on the engine, delegates to authorize_preview()
        and refuses to issue approval without a trusted user callback.
        """
        if self.require_trusted_callback:
            return self.authorize_preview(preview)

        token = self._compute_approval_hmac(preview) if approved else ""
        return ExecutionApproval(
            preview_id=preview.preview_id,
            preview_hash=preview.preview_hash,
            approved=approved,
            approved_executable=preview.executable,
            approved_arguments=list(preview.arguments),
            approved_working_directory=preview.working_directory,
            approved_project_root=preview.authorized_project_root,
            approved_risk_level=preview.risk_level,
            approval_token=token,
            auth_method="ENGINE_ISSUED" if approved else "DENIED",
            is_generic=False,
        )

    # --------------------------------------------------------------------------
    # Step 7–12: Verify Approval, Revalidate Boundaries, Execute & Audit Log
    # --------------------------------------------------------------------------

    def execute(
        self,
        preview: ExecutionPreview,
        approval: Optional[ExecutionApproval],
    ) -> ProjectExecutionResult:
        """Executes an approved preview after verifying binding and revalidating boundaries."""
        execution_id = str(uuid.uuid4())
        warnings = list(preview.warnings)

        # 1. Blocked previews cannot be executed
        if preview.is_blocked or preview.risk_level == ExecutionRiskLevel.HIGH:
            reason = preview.block_reason or "HIGH-risk or policy-blocked operation cannot be executed."
            res = ProjectExecutionResult(
                execution_id=execution_id,
                status=ExecutionStatus.BLOCKED,
                project_root=preview.authorized_project_root,
                command=preview.command,
                working_directory=preview.working_directory,
                risk_level=preview.risk_level,
                approval_verified=False,
                warnings=warnings,
                error_message=reason,
            )
            self._record_audit(preview, res, approval_status="BLOCKED")
            return res

        # 2. Verify explicit, bound, unexpired, one-time-use approval
        ok, err_msg = self._verify_approval(preview, approval)
        if not ok:
            res = ProjectExecutionResult(
                execution_id=execution_id,
                status=ExecutionStatus.REJECTED,
                project_root=preview.authorized_project_root,
                command=preview.command,
                working_directory=preview.working_directory,
                risk_level=preview.risk_level,
                approval_verified=False,
                warnings=warnings,
                error_message=err_msg,
            )
            self._record_audit(preview, res, approval_status="REJECTED")
            return res

        # Consume preview ID immediately so it cannot be reused
        self._consumed_preview_ids.add(preview.preview_id)

        # 3. Revalidate all security boundaries immediately before execution
        root_ok, resolved_root, root_err = self._validate_directory_boundary(
            Path(preview.authorized_project_root), parent_root=None
        )
        wd_ok, resolved_wd, wd_err = self._validate_directory_boundary(
            Path(preview.working_directory), parent_root=resolved_root
        ) if root_ok and resolved_root else (False, None, root_err)

        if not root_ok or not wd_ok or resolved_wd is None:
            reason = root_err or wd_err or "Security boundary revalidation failed."
            res = ProjectExecutionResult(
                execution_id=execution_id,
                status=ExecutionStatus.BLOCKED,
                project_root=preview.authorized_project_root,
                command=preview.command,
                working_directory=preview.working_directory,
                risk_level=preview.risk_level,
                approval_verified=True,
                warnings=warnings,
                error_message=reason,
            )
            self._record_audit(preview, res, approval_status="APPROVED_BUT_BLOCKED")
            return res

        # Re-validate command array & resolved executable
        parse_ok, _, resolved_exe, _, parse_err = self._parse_and_validate_command(
            preview.command, resolved_root, resolved_wd
        )
        if not parse_ok:
            res = ProjectExecutionResult(
                execution_id=execution_id,
                status=ExecutionStatus.BLOCKED,
                project_root=preview.authorized_project_root,
                command=preview.command,
                working_directory=preview.working_directory,
                risk_level=preview.risk_level,
                approval_verified=True,
                warnings=warnings,
                error_message=parse_err or "Executable policy revalidation failed.",
            )
            self._record_audit(preview, res, approval_status="APPROVED_BUT_BLOCKED")
            return res

        exec_binary = resolved_exe or preview.resolved_executable or preview.executable
        cmd_args = [exec_binary] + list(preview.arguments)

        # 4. Execute through bounded streaming Popen(shell=False) + Job Object cleanup
        safe_env = self._build_safe_env()
        start_time = time.perf_counter()

        try:
            rc, raw_out, raw_err, out_trunc, err_trunc, timed_out, cleanup_warn = self._run_bounded_subprocess(
                cmd_args=cmd_args,
                cwd=str(resolved_wd),
                timeout_seconds=preview.timeout_seconds,
                env=safe_env,
            )
            duration = round(time.perf_counter() - start_time, 4)
            if cleanup_warn:
                warnings.append(cleanup_warn)

            stdout_clean, redact_out_trunc = self._sanitize_and_truncate(raw_out)
            stderr_clean, redact_err_trunc = self._sanitize_and_truncate(raw_err)
            final_out_trunc = out_trunc or redact_out_trunc
            final_err_trunc = err_trunc or redact_err_trunc
            truncated = final_out_trunc or final_err_trunc

            if timed_out:
                msg = f"Process timed out after {preview.timeout_seconds}s."
                if cleanup_warn:
                    msg = f"{msg} ({cleanup_warn})"
                res = ProjectExecutionResult(
                    execution_id=execution_id,
                    status=ExecutionStatus.TIMED_OUT,
                    project_root=str(resolved_root),
                    command=preview.command,
                    working_directory=str(resolved_wd),
                    exit_code=None,
                    stdout=stdout_clean,
                    stderr=stderr_clean,
                    duration_seconds=duration,
                    timed_out=True,
                    output_truncated=truncated,
                    stdout_truncated=final_out_trunc,
                    stderr_truncated=final_err_trunc,
                    approval_verified=True,
                    risk_level=preview.risk_level,
                    warnings=warnings,
                    error_message=msg,
                )
                self._record_audit(preview, res, approval_status="APPROVED")
                return res

            status = ExecutionStatus.COMPLETED if rc == 0 else ExecutionStatus.FAILED
            err_summary = stderr_clean[:300] if status == ExecutionStatus.FAILED and stderr_clean else None

            res = ProjectExecutionResult(
                execution_id=execution_id,
                status=status,
                project_root=str(resolved_root),
                command=preview.command,
                working_directory=str(resolved_wd),
                exit_code=rc,
                stdout=stdout_clean,
                stderr=stderr_clean,
                duration_seconds=duration,
                timed_out=False,
                output_truncated=truncated,
                stdout_truncated=final_out_trunc,
                stderr_truncated=final_err_trunc,
                approval_verified=True,
                risk_level=preview.risk_level,
                warnings=warnings,
                error_message=err_summary,
            )
            self._record_audit(preview, res, approval_status="APPROVED")
            return res

        except Exception as exc:
            duration = round(time.perf_counter() - start_time, 4)
            err_clean, _ = self._sanitize_and_truncate(str(exc))
            res = ProjectExecutionResult(
                execution_id=execution_id,
                status=ExecutionStatus.FAILED,
                project_root=str(resolved_root),
                command=preview.command,
                working_directory=str(resolved_wd),
                exit_code=None,
                stdout="",
                stderr="",
                duration_seconds=duration,
                timed_out=False,
                output_truncated=False,
                approval_verified=True,
                risk_level=preview.risk_level,
                warnings=warnings,
                error_message=f"Execution error: {err_clean}",
            )
            self._record_audit(preview, res, approval_status="APPROVED")
            return res

    def _run_bounded_subprocess(
        self,
        cmd_args: List[str],
        cwd: str,
        timeout_seconds: int,
        env: Dict[str, str],
    ) -> Tuple[Optional[int], str, str, bool, bool, bool, Optional[str]]:
        """Executes a command with streaming memory-bounded stdout/stderr capture and process-tree cleanup."""
        job = _WindowsJobObject()
        cleanup_warning: Optional[str] = None

        proc = subprocess.Popen(
            cmd_args,
            cwd=cwd,
            shell=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=env,
        )

        job_assigned = job.assign(proc)
        if os.name == "nt" and not job_assigned and job.error:
            cleanup_warning = f"Process-tree Job Object warning: {job.error}"

        out_chunks: List[bytes] = []
        err_chunks: List[bytes] = []
        out_trunc = [False]
        err_trunc = [False]

        def _drain_stream(stream: Any, chunks: List[bytes], trunc_flag: List[bool]) -> None:
            kept = 0
            try:
                while True:
                    chunk = stream.read(4096)
                    if not chunk:
                        break
                    if kept < self.max_output_chars:
                        remaining = self.max_output_chars - kept
                        if len(chunk) > remaining:
                            chunks.append(chunk[:remaining])
                            kept += remaining
                            trunc_flag[0] = True
                        else:
                            chunks.append(chunk)
                            kept += len(chunk)
                    else:
                        trunc_flag[0] = True
            except Exception:
                pass
            finally:
                try:
                    stream.close()
                except Exception:
                    pass

        t_out = threading.Thread(target=_drain_stream, args=(proc.stdout, out_chunks, out_trunc), daemon=True)
        t_err = threading.Thread(target=_drain_stream, args=(proc.stderr, err_chunks, err_trunc), daemon=True)
        t_out.start()
        t_err.start()

        timed_out = False
        rc: Optional[int] = None
        try:
            rc = proc.wait(timeout=timeout_seconds)
        except subprocess.TimeoutExpired:
            timed_out = True
            if job_assigned:
                term_err = job.terminate_tree()
                if term_err:
                    cleanup_warning = f"Job termination warning: {term_err}"
            try:
                proc.kill()
            except Exception:
                pass
            try:
                proc.wait(timeout=2)
            except Exception:
                pass
        finally:
            job.close()

        t_out.join(timeout=2.0)
        t_err.join(timeout=2.0)

        stdout_str = b"".join(out_chunks).decode("utf-8", errors="replace")
        stderr_str = b"".join(err_chunks).decode("utf-8", errors="replace")
        if out_trunc[0]:
            stdout_str += "\n... [OUTPUT TRUNCATED]"
        if err_trunc[0]:
            stderr_str += "\n... [OUTPUT TRUNCATED]"

        return rc, stdout_str, stderr_str, out_trunc[0], err_trunc[0], timed_out, cleanup_warning

    # --------------------------------------------------------------------------
    # Command Parsing & Executable Policy
    # --------------------------------------------------------------------------

    def _parse_and_validate_command(
        self,
        command_str: str,
        resolved_root: Path,
        resolved_wd: Optional[Path] = None,
    ) -> Tuple[bool, str, Optional[str], List[str], Optional[str]]:
        """Parses a command into [executable, *args], resolves path safely, and enforces policy."""
        raw = (command_str or "").strip()
        if not raw:
            return False, "", None, [], "Empty command cannot be executed."

        # 1. Reject forbidden shell syntax (never silently strip)
        for pattern, msg in _FORBIDDEN_SHELL_PATTERNS:
            if pattern.search(raw):
                return False, "", None, [], f"Security: {msg}"

        # 2. Tokenize without shell interpolation
        try:
            tokens = shlex.split(raw, posix=False)
            cleaned_tokens = [self._strip_matching_quotes(t) for t in tokens if t.strip()]
        except ValueError as ve:
            return False, "", None, [], f"Command tokenization failed: {ve}"

        if not cleaned_tokens:
            return False, "", None, [], "No executable token found."

        exe = cleaned_tokens[0]
        args = cleaned_tokens[1:]
        exe_lower = exe.lower()
        exe_name = Path(exe).name.lower()
        exe_stem = Path(exe).stem.lower()

        # 3. Reject shells and Windows batch launchers (.bat / .cmd)
        if exe_name in BLOCKED_SHELL_EXECUTABLES or exe_stem in BLOCKED_SHELL_EXECUTABLES:
            return False, exe, None, args, f"Security: Shell interpreter '{exe}' is blocked."

        if exe_lower.endswith((".bat", ".cmd")):
            return (
                False, exe, None, args,
                f"Security: Batch launcher '{exe}' (.bat/.cmd) implicitly invokes cmd.exe on Windows "
                "and is rejected under the shell=False execution policy."
            )

        # 4. Enforce executable allowlist policy
        if exe_stem not in self.allowed_executables and exe_name not in self.allowed_executables:
            return False, exe, None, args, f"Executable '{exe}' is not in the allowed executable policy."

        # 5. Anti-hijacking & executable path resolution
        wd_check = resolved_wd or resolved_root
        resolved_exe: Optional[str] = None

        if "/" in exe or "\\" in exe:
            exe_path = Path(exe)
            is_active_python = False
            try:
                is_active_python = exe_path.resolve() == Path(sys.executable).resolve()
            except Exception:
                pass
            if is_active_python:
                resolved_exe = str(exe_path.resolve())
            else:
                if not exe_path.is_absolute():
                    exe_path = resolved_root / exe_path
                ok, _, err = self._validate_directory_boundary(exe_path.parent, parent_root=resolved_root, check_is_dir=False)
                if not ok:
                    return False, exe, None, args, err or f"Executable path '{exe}' is outside the authorized project root."
                # Even inside project root, block shadowed binaries matching system tool names
                return False, exe, None, args, (
                    f"Security: Blocked local executable '{exe}' inside project directory to prevent binary shadowing."
                )
        else:
            # Check for CWD / project-root shadowing (Windows CreateProcess searches CWD before PATH!)
            shadow_candidates = [exe, f"{exe}.exe", f"{exe}.bat", f"{exe}.cmd"]
            for check_dir in {wd_check, resolved_root}:
                for cand in shadow_candidates:
                    cand_path = check_dir / cand
                    if cand_path.is_file():
                        return False, exe, None, args, (
                            f"Security: Blocked potential PATH hijacking / executable shadowing "
                            f"('{cand}' found in project directory '{check_dir}')."
                        )

            # Resolve executable strictly from trusted PATH directories (excluding '.', '', project dirs)
            raw_path_entries = os.environ.get("PATH", "").split(os.pathsep)
            safe_path_entries: List[str] = []
            for entry in raw_path_entries:
                cleaned_entry = entry.strip()
                if not cleaned_entry or cleaned_entry in (".", ".."):
                    continue
                try:
                    p_entry = Path(cleaned_entry).resolve()
                    if p_entry == resolved_root or resolved_root in p_entry.parents:
                        continue
                    if p_entry == wd_check or wd_check in p_entry.parents:
                        continue
                    safe_path_entries.append(str(p_entry))
                except Exception:
                    continue

            trusted_path = os.pathsep.join(safe_path_entries)
            found = shutil.which(exe, path=trusted_path) if trusted_path else None
            if found:
                resolved_found = Path(found).resolve()
                # Reject if resolved binary is inside project root/wd
                if resolved_found == resolved_root or resolved_root in resolved_found.parents:
                    return False, exe, None, args, (
                        f"Security: Resolved executable '{resolved_found}' is inside project directory."
                    )
                resolved_exe = str(resolved_found)

        return True, exe, resolved_exe, args, None

    @staticmethod
    def _strip_matching_quotes(token: str) -> str:
        if len(token) >= 2 and ((token[0] == '"' and token[-1] == '"') or (token[0] == "'" and token[-1] == "'")):
            return token[1:-1]
        return token

    # --------------------------------------------------------------------------
    # Security Boundaries, Risk, Approval Binding & Audit
    # --------------------------------------------------------------------------

    def _validate_directory_boundary(
        self,
        target: Path,
        parent_root: Optional[Path],
        check_is_dir: bool = True,
    ) -> Tuple[bool, Optional[Path], Optional[str]]:
        """Resolves canonical paths, blocks traversal/symlinks escaping root, and blocks protected system dirs."""
        try:
            if ".." in target.parts:
                return False, None, f"Security: Path traversal ('..') rejected in '{target}'."
            resolved = target.resolve()
        except Exception as e:
            return False, None, f"Failed to resolve path '{target}': {e}"

        if check_is_dir and (not resolved.exists() or not resolved.is_dir()):
            return False, None, f"Directory does not exist: '{resolved}'."

        # Protected system & root directories check
        for prot in self._protected_dirs:
            if resolved == prot or (prot != resolved and prot in getattr(config, "PROTECTED_SYSTEM_DIRS", []) and prot in resolved.parents):
                return False, None, f"Security: Access to protected system path '{resolved}' is forbidden."

        # Check confinement inside parent_root if supplied
        if parent_root is not None:
            if resolved != parent_root and parent_root not in resolved.parents:
                return False, None, f"Security: Path '{resolved}' escapes authorized project root '{parent_root}'."

        return True, resolved, None

    def _assess_risk(
        self,
        rec_cmd: RecommendedCommand,
        exe: str,
        args: List[str],
    ) -> Tuple[ExecutionRiskLevel, bool]:
        """Determines ExecutionRiskLevel and whether project-defined code is executed."""
        combined = f"{exe} {' '.join(args)}".lower()
        exe_stem = Path(exe).stem.lower()

        executes_project_code = (
            rec_cmd.evidence_category == EvidenceCategory.PROJECT_DEFINED
            or (exe_stem in {"npm", "pnpm", "yarn", "bun"} and any(a in args for a in ("run", "dev", "start", "test", "build")))
            or (exe_stem in {"python", "python3", "py"} and any(a.endswith(".py") or a in {"-m", "pytest"} for a in args))
            or exe_stem in {"poetry", "pipenv", "make", "cmake", "meson", "gradle", "mvn", "dotnet", "flutter", "dart"}
        )

        # HIGH risk check (never lowered even if manifest declares it)
        if rec_cmd.risk_level == CommandRiskLevel.HIGH or any(
            re.search(rf"\b{kw}\b", combined) for kw in _HIGH_RISK_KEYWORDS
        ):
            return ExecutionRiskLevel.HIGH, executes_project_code

        if rec_cmd.risk_level == CommandRiskLevel.LOW and not executes_project_code:
            return ExecutionRiskLevel.LOW, False

        return ExecutionRiskLevel.MEDIUM, executes_project_code

    def _compute_preview_hash(
        self,
        preview_id: str,
        executable: str,
        arguments: List[str],
        working_directory: str,
        authorized_project_root: str,
        risk_level: ExecutionRiskLevel,
    ) -> str:
        payload = json.dumps(
            {
                "preview_id": preview_id,
                "executable": executable,
                "arguments": arguments,
                "working_directory": working_directory,
                "authorized_project_root": authorized_project_root,
                "risk_level": risk_level.value,
            },
            sort_keys=True,
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def _verify_approval(
        self,
        preview: ExecutionPreview,
        approval: Optional[ExecutionApproval],
    ) -> Tuple[bool, Optional[str]]:
        """Strictly validates one-time-use, unexpired, exact-bound ExecutionApproval."""
        if approval is None:
            return False, "Execution rejected: explicit user approval is required."
        if not approval.approved or approval.is_generic:
            return False, "Execution rejected: approval must be explicit and operation-specific."
        if preview.preview_id in self._consumed_preview_ids:
            return False, "Execution rejected: approval/preview has already been used (one-time-use)."
        if preview.preview_id not in self._issued_previews:
            return False, "Execution rejected: unknown or unissued preview."

        # Check expiration
        try:
            if datetime.now() > datetime.fromisoformat(preview.expires_at):
                return False, "Execution rejected: preview approval has expired."
        except Exception:
            return False, "Execution rejected: invalid preview expiration timestamp."

        # Verify preview integrity against recomputed hash
        expected_hash = self._compute_preview_hash(
            preview_id=preview.preview_id,
            executable=preview.executable,
            arguments=preview.arguments,
            working_directory=preview.working_directory,
            authorized_project_root=preview.authorized_project_root,
            risk_level=preview.risk_level,
        )
        if preview.preview_hash != expected_hash:
            return False, "Execution rejected: preview integrity hash mismatch (tampered preview)."

        # Verify approval binds to exact preview fields
        if approval.preview_id != preview.preview_id or approval.preview_hash != preview.preview_hash:
            return False, "Execution rejected: approval does not match preview identity/hash."
        if approval.approved_executable != preview.executable:
            return False, "Execution rejected: executable changed since approval."
        if approval.approved_arguments != preview.arguments:
            return False, "Execution rejected: arguments changed since approval."
        if approval.approved_working_directory != preview.working_directory:
            return False, "Execution rejected: working directory changed since approval."
        if approval.approved_project_root != preview.authorized_project_root:
            return False, "Execution rejected: authorized project root changed since approval."
        if approval.approved_risk_level != preview.risk_level:
            return False, "Execution rejected: risk level changed since approval."

        # Verify cryptographic HMAC authentication token (distinguishes user auth from preview hash)
        expected_hmac = self._compute_approval_hmac(preview)
        if not approval.approval_token or not hmac.compare_digest(approval.approval_token, expected_hmac):
            return False, "Execution rejected: invalid or forged approval authentication token."

        return True, None

    def _blocked_preview(
        self,
        preview_id: str,
        now: datetime,
        expires: datetime,
        recommendation: ProjectCommandRecommendation,
        auth_root: str,
        command: str,
        executable: str,
        arguments: List[str],
        working_directory: str,
        evidence_source: str,
        evidence_category: EvidenceCategory,
        risk_level: ExecutionRiskLevel,
        executes_project_code: bool,
        warnings: List[str],
        timeout_seconds: int,
        reason: str,
    ) -> ExecutionPreview:
        p_hash = self._compute_preview_hash(
            preview_id, executable, arguments, working_directory, auth_root, risk_level
        )
        return ExecutionPreview(
            preview_id=preview_id,
            preview_hash=p_hash,
            created_at=now.isoformat(),
            expires_at=expires.isoformat(),
            user_request=recommendation.user_request,
            project_name=recommendation.project_name,
            authorized_project_root=auth_root,
            command=command,
            executable=executable,
            arguments=arguments,
            working_directory=working_directory,
            evidence_source=evidence_source,
            evidence_category=evidence_category,
            risk_level=risk_level,
            expected_operation=reason,
            executes_project_code=executes_project_code,
            warnings=warnings + [reason],
            timeout_seconds=timeout_seconds,
            approval_required=True,
            status=ExecutionStatus.BLOCKED,
            is_blocked=True,
            block_reason=reason,
        )

    def _build_safe_env(self) -> Dict[str, str]:
        """Constructs a minimal, secret-free subprocess environment."""
        safe_env: Dict[str, str] = {}
        for k, v in os.environ.items():
            k_upper = k.upper()
            if any(s in k_upper for s in ("KEY", "SECRET", "TOKEN", "PASSWORD", "CREDENTIAL", "AUTH", "PRIVATE")):
                continue
            if k_upper in _SAFE_ENV_KEYS:
                safe_env[k] = v
        if os.name == "nt":
            # Prevent Windows CreateProcess from searching CWD for executables
            safe_env["NoDefaultCurrentDirectoryInExePath"] = "1"
        return safe_env

    def _sanitize_and_truncate(self, text: str) -> Tuple[str, bool]:
        """Redacts secret patterns and enforces output length ceiling."""
        cleaned = text
        for pat, repl in _SECRET_REDACTION_PATTERNS:
            cleaned = pat.sub(repl, cleaned)

        truncated = False
        if len(cleaned) > self.max_output_chars:
            cleaned = cleaned[: self.max_output_chars] + "\n... [OUTPUT TRUNCATED]"
            truncated = True
        return cleaned, truncated

    def _record_audit(
        self,
        preview: ExecutionPreview,
        result: ProjectExecutionResult,
        approval_status: str,
    ) -> None:
        """Records execution attempt in HistoryManager without leaking secrets."""
        err_summary = None
        if result.error_message:
            err_summary, _ = self._sanitize_and_truncate(result.error_message)

        entry: Dict[str, Any] = {
            "execution_id": result.execution_id,
            "user_request": preview.user_request,
            "project_root": preview.authorized_project_root,
            "project_name": preview.project_name,
            "command": preview.command,
            "executable": preview.executable,
            "arguments": preview.arguments,
            "working_directory": preview.working_directory,
            "risk_level": result.risk_level.value,
            "approval_status": approval_status,
            "execution_status": result.status.value,
            "exit_code": result.exit_code,
            "duration_seconds": result.duration_seconds,
            "output_truncated": result.output_truncated,
            "error_summary": err_summary,
            # Backward-compatible summary keys for existing HistoryManager readers
            "intent": "PROJECT_COMMAND_EXECUTION",
            "status": result.status.value,
        }
        self.history_manager.log_entry(entry)
