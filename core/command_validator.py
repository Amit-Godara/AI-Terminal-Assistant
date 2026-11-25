"""Independent Command Safety Validator module for AI Terminal Assistant.

Validates structured command operations independently of the command generator:
- Enforces strict operation allowlists
- Verifies argument existence and expected types
- Validates Windows path safety and detects directory traversal (.. escapes)
- Restricts filesystem operations to an approved workspace boundary
- Blocks access to protected Windows system directories
- Detects existing target conflicts (prevents silent overwrites)
- Enforces risk levels and mandatory confirmation requirements
"""

from enum import Enum
import logging
import os
from pathlib import Path
import re
from typing import Any, Dict, List, Optional, Set, Tuple, Union

from pydantic import BaseModel, Field

import config
from core.command_generator import (
    CommandOperation,
    OperationType,
    RiskLevel,
    GenerationStatus,
)

logger = logging.getLogger(__name__)


class ValidationStatus(str, Enum):
    """Status outcomes of safety validation."""
    VALID = "VALID"
    INVALID_OPERATION = "INVALID_OPERATION"
    INVALID_ARGUMENT = "INVALID_ARGUMENT"
    UNSAFE_PATH = "UNSAFE_PATH"
    BLOCKED_OPERATION = "BLOCKED_OPERATION"
    MISSING_PARAMETERS = "MISSING_PARAMETERS"
    LOW_CONFIDENCE = "LOW_CONFIDENCE"
    AMBIGUOUS_OPERATION = "AMBIGUOUS_OPERATION"
    CONFLICT_TARGET_EXISTS = "CONFLICT_TARGET_EXISTS"


class ValidationResult(BaseModel):
    """Structured result of independent safety validation."""
    is_valid: bool
    validation_status: ValidationStatus
    operation_type: OperationType
    risk_level: RiskLevel = RiskLevel.LOW
    requires_confirmation: bool = False
    requires_double_confirmation: bool = False
    validation_errors: List[str] = Field(default_factory=list)
    security_warnings: List[str] = Field(default_factory=list)
    validated_arguments: Dict[str, Any] = Field(default_factory=dict)
    target_path: Optional[str] = None
    destination_path: Optional[str] = None
    reason: str = ""
    operation: Optional[CommandOperation] = None


class CommandValidator:
    """Independent Safety Validator enforcing strict security boundaries."""

    # Disallowed characters in Windows file/directory names (excluding path separators and drive colon)
    INVALID_CHARS_PATTERN = re.compile(r'[<>:"|?*]')

    # Allowlisted system commands
    ALLOWED_SYSTEM_COMMANDS: Set[str] = {
        "Get-ComputerInfo",
        "hostname",
        "Get-CimInstance Win32_OperatingSystem | Select-Object Caption, Version, OSArchitecture",
    }

    # Allowlisted network commands
    ALLOWED_NETWORK_COMMANDS: Set[str] = {
        "ipconfig",
        "Get-NetAdapter",
        "Get-NetTCPConnection",
    }

    def __init__(
        self,
        workspace_dir: Optional[Path] = None,
        safe_mode: bool = True,
        protected_paths: Optional[List[Path]] = None,
    ) -> None:
        """Initializes the CommandValidator.

        Args:
            workspace_dir: Configured root directory for filesystem operations.
            safe_mode: Enforces strict sandboxing and confirmation requirements.
            protected_paths: Explicit list of protected Windows system paths.
        """
        self.workspace_dir = (workspace_dir or config.WORKSPACE_DIR).resolve()
        self.safe_mode = safe_mode
        self.protected_system_dirs = [
            p.resolve() for p in getattr(config, "PROTECTED_SYSTEM_DIRS", [])
        ]
        self.protected_root_dirs = [
            p.resolve() for p in (protected_paths or getattr(config, "PROTECTED_ROOT_DIRS", []))
        ]

    def validate_path(
        self,
        path_str: str,
        must_exist: bool = False,
        allow_directory: bool = True,
        allow_file: bool = True,
    ) -> Tuple[bool, Optional[Path], List[str]]:
        """Validates and resolves a path string against the workspace sandbox boundary.

        Checks:
        1. Non-empty string
        2. No null bytes or invalid control characters
        3. Path traversal detection (.. escaping workspace)
        4. Target resolves strictly within workspace_dir
        5. Not matching any protected system locations
        6. Symlinks/junctions do not escape the workspace
        7. Existence checks when requested

        Returns:
            Tuple of (is_safe, resolved_path, list_of_errors).
        """
        errors: List[str] = []
        if not path_str or not str(path_str).strip():
            return False, None, ["Path string is empty or whitespace."]

        raw_str = str(path_str).strip().strip('"').strip("'")
        if "\0" in raw_str:
            return False, None, ["Null bytes are strictly prohibited in paths."]

        try:
            candidate_path = Path(raw_str)
        except Exception as e:
            return False, None, [f"Invalid path syntax: {e}"]

        # Resolve path against workspace
        try:
            if candidate_path.is_absolute():
                # On Windows, candidate must match workspace drive
                resolved = candidate_path.resolve()
            else:
                resolved = (self.workspace_dir / candidate_path).resolve()
        except Exception as e:
            return False, None, [f"Path resolution error: {e}"]

        # 1. Workspace Boundary Containment Check
        # Resolved path must either be workspace_dir itself or a child of workspace_dir
        try:
            is_inside = (resolved == self.workspace_dir) or (self.workspace_dir in resolved.parents)
        except Exception:
            is_inside = False

        if not is_inside:
            errors.append(
                f"Path '{raw_str}' escapes the permitted workspace boundary ({self.workspace_dir})."
            )
            return False, None, errors

        # 2. Protected System Location Check
        for root_dir in self.protected_root_dirs:
            try:
                if resolved == root_dir:
                    errors.append(f"Direct operation on protected root location '{raw_str}' ({root_dir}) is blocked.")
                    return False, None, errors
            except Exception:
                continue

        for sys_dir in self.protected_system_dirs:
            try:
                if resolved == sys_dir or sys_dir in resolved.parents:
                    errors.append(f"Path '{raw_str}' resolves inside a protected system directory ({sys_dir}).")
                    return False, None, errors
            except Exception:
                continue

        # 3. Symlink / Junction Escape Check
        if resolved.is_symlink():
            try:
                real_target = resolved.resolve()
                if not ((real_target == self.workspace_dir) or (self.workspace_dir in real_target.parents)):
                    errors.append(f"Symlink '{raw_str}' points outside the permitted workspace.")
                    return False, None, errors
            except Exception as e:
                errors.append(f"Cannot reliably resolve symlink target: {e}")
                return False, None, errors

        # 4. Existence and type constraints
        if must_exist and not resolved.exists():
            errors.append(f"Path '{raw_str}' does not exist.")
            return False, resolved, errors

        if resolved.exists():
            if not allow_directory and resolved.is_dir():
                errors.append(f"Expected a file, but '{raw_str}' is a directory.")
                return False, resolved, errors
            if not allow_file and resolved.is_file():
                errors.append(f"Expected a directory, but '{raw_str}' is a file.")
                return False, resolved, errors

        return True, resolved, errors

    def validate(self, operation: Union[CommandOperation, str]) -> ValidationResult:
        """Validates a command operation for security, syntax, and execution safety.

        Args:
            operation: CommandOperation object or legacy command string.

        Returns:
            ValidationResult with validation status and security determinations.
        """
        # Backward compatibility check for plain string commands
        if isinstance(operation, str):
            return self._validate_legacy_string(operation)

        if not isinstance(operation, CommandOperation):
            return ValidationResult(
                is_valid=False,
                validation_status=ValidationStatus.INVALID_OPERATION,
                operation_type=OperationType.UNKNOWN,
                validation_errors=["Operation must be an instance of CommandOperation."],
                reason="Invalid operation input type.",
            )

        # 1. Verify generation status
        if operation.generation_status == GenerationStatus.NEEDS_CLARIFICATION:
            return ValidationResult(
                is_valid=False,
                validation_status=ValidationStatus.MISSING_PARAMETERS,
                operation_type=operation.operation_type,
                risk_level=operation.risk_level,
                validation_errors=[
                    operation.clarification_prompt
                    or f"Missing required parameters: {', '.join(operation.missing_parameters)}"
                ],
                reason="Operation cannot execute because required parameters are missing.",
                operation=operation,
            )

        if operation.generation_status == GenerationStatus.UNSUPPORTED_INTENT:
            return ValidationResult(
                is_valid=False,
                validation_status=ValidationStatus.BLOCKED_OPERATION,
                operation_type=operation.operation_type,
                risk_level=operation.risk_level,
                validation_errors=["The requested operation is unsupported or blocked by safety policy."],
                reason="Unsupported intent.",
                operation=operation,
            )

        if operation.generation_status == GenerationStatus.LOW_CONFIDENCE:
            return ValidationResult(
                is_valid=False,
                validation_status=ValidationStatus.LOW_CONFIDENCE,
                operation_type=operation.operation_type,
                risk_level=operation.risk_level,
                validation_errors=["Operation rejected due to low confidence in intent inference."],
                reason="Low confidence classification.",
                operation=operation,
            )

        if operation.generation_status == GenerationStatus.AMBIGUOUS_REQUEST:
            return ValidationResult(
                is_valid=False,
                validation_status=ValidationStatus.AMBIGUOUS_OPERATION,
                operation_type=operation.operation_type,
                risk_level=operation.risk_level,
                validation_errors=["Operation is ambiguous and requires user clarification."],
                reason="Ambiguous request.",
                operation=operation,
            )

        if operation.operation_type == OperationType.UNKNOWN:
            return ValidationResult(
                is_valid=False,
                validation_status=ValidationStatus.INVALID_OPERATION,
                operation_type=OperationType.UNKNOWN,
                validation_errors=["Unknown operation type."],
                reason="Operation type is UNKNOWN.",
                operation=operation,
            )

        # 2. Dispatch to operation-specific safety validators
        op_type = operation.operation_type

        if op_type == OperationType.LIST:
            return self._validate_list(operation)
        elif op_type == OperationType.CREATE_DIR:
            return self._validate_create_dir(operation)
        elif op_type == OperationType.DELETE:
            return self._validate_delete(operation)
        elif op_type == OperationType.SEARCH:
            return self._validate_search(operation)
        elif op_type == OperationType.CHECK_VERSION:
            return self._validate_check_version(operation)
        elif op_type == OperationType.GET_LOCATION:
            return self._validate_get_location(operation)
        elif op_type == OperationType.MOVE:
            return self._validate_move(operation)
        elif op_type == OperationType.COPY:
            return self._validate_copy(operation)
        elif op_type == OperationType.SYSTEM_INFO:
            return self._validate_system_info(operation)
        elif op_type == OperationType.NETWORK_INFO:
            return self._validate_network_info(operation)
        elif op_type == OperationType.CREATE_FILE:
            return self._validate_create_file(operation)
        else:
            return ValidationResult(
                is_valid=False,
                validation_status=ValidationStatus.INVALID_OPERATION,
                operation_type=op_type,
                validation_errors=[f"Unsupported operation type: {op_type}"],
                reason="Operation not supported in validator.",
                operation=operation,
            )

    # --------------------------------------------------------------------------
    # Specific Operation Safety Validators
    # --------------------------------------------------------------------------

    def _validate_list(self, op: CommandOperation) -> ValidationResult:
        """Validates LIST_FILES operation."""
        target_dir = op.arguments.get("directory_name") or op.arguments.get("source")
        if target_dir:
            is_safe, resolved, errors = self.validate_path(
                target_dir, must_exist=True, allow_directory=True, allow_file=False
            )
            if not is_safe:
                return ValidationResult(
                    is_valid=False,
                    validation_status=ValidationStatus.UNSAFE_PATH,
                    operation_type=op.operation_type,
                    risk_level=op.risk_level,
                    validation_errors=errors,
                    reason="Directory for listing is unsafe or does not exist.",
                    operation=op,
                )
            target_path_str = str(resolved)
        else:
            target_path_str = str(self.workspace_dir)

        return ValidationResult(
            is_valid=True,
            validation_status=ValidationStatus.VALID,
            operation_type=op.operation_type,
            risk_level=RiskLevel.LOW,
            requires_confirmation=False,
            validated_arguments={"directory": target_path_str},
            target_path=target_path_str,
            reason="Read-only directory listing within permitted workspace.",
            operation=op,
        )

    def _validate_create_dir(self, op: CommandOperation) -> ValidationResult:
        """Validates CREATE_DIRECTORY operation."""
        dir_name = op.arguments.get("directory_name") or op.arguments.get("destination") or op.arguments.get("source")
        if not dir_name:
            return ValidationResult(
                is_valid=False,
                validation_status=ValidationStatus.MISSING_PARAMETERS,
                operation_type=op.operation_type,
                risk_level=op.risk_level,
                validation_errors=["Directory name is required for CREATE_DIRECTORY."],
                reason="Missing directory name.",
                operation=op,
            )

        is_safe, resolved, errors = self.validate_path(dir_name, must_exist=False)
        if not is_safe or resolved is None:
            return ValidationResult(
                is_valid=False,
                validation_status=ValidationStatus.UNSAFE_PATH,
                operation_type=op.operation_type,
                risk_level=op.risk_level,
                validation_errors=errors,
                reason="Target directory path is unsafe or violates workspace boundaries.",
                operation=op,
            )

        warnings = []
        if resolved.exists():
            if resolved.is_file():
                return ValidationResult(
                    is_valid=False,
                    validation_status=ValidationStatus.CONFLICT_TARGET_EXISTS,
                    operation_type=op.operation_type,
                    risk_level=op.risk_level,
                    validation_errors=[f"A file already exists at '{resolved}'."],
                    reason="Conflict: Path already exists as a file.",
                    operation=op,
                )
            warnings.append(f"Directory '{resolved}' already exists.")

        return ValidationResult(
            is_valid=True,
            validation_status=ValidationStatus.VALID,
            operation_type=op.operation_type,
            risk_level=RiskLevel.MEDIUM,
            requires_confirmation=True,
            security_warnings=warnings,
            validated_arguments={"directory_path": str(resolved)},
            target_path=str(resolved),
            reason="Directory creation validated within workspace boundary.",
            operation=op,
        )

    def _validate_delete(self, op: CommandOperation) -> ValidationResult:
        """Validates DELETE_FILE operation with high-risk safeguards."""
        target = op.arguments.get("file_name") or op.arguments.get("source")
        if not target:
            return ValidationResult(
                is_valid=False,
                validation_status=ValidationStatus.MISSING_PARAMETERS,
                operation_type=op.operation_type,
                risk_level=RiskLevel.HIGH,
                validation_errors=["File name is required for DELETE_FILE."],
                reason="Missing file name.",
                operation=op,
            )

        # File must exist, must be inside workspace, and CANNOT be a directory
        is_safe, resolved, errors = self.validate_path(
            target, must_exist=True, allow_directory=False, allow_file=True
        )
        if not is_safe or resolved is None:
            return ValidationResult(
                is_valid=False,
                validation_status=ValidationStatus.UNSAFE_PATH,
                operation_type=op.operation_type,
                risk_level=RiskLevel.HIGH,
                validation_errors=errors,
                reason="Target file for deletion is unsafe, not found, or is a directory.",
                operation=op,
            )

        return ValidationResult(
            is_valid=True,
            validation_status=ValidationStatus.VALID,
            operation_type=op.operation_type,
            risk_level=RiskLevel.HIGH,
            requires_confirmation=True,
            requires_double_confirmation=True,
            security_warnings=[
                f"Permanent deletion of '{resolved.name}'. This action cannot be undone."
            ],
            validated_arguments={"file_path": str(resolved)},
            target_path=str(resolved),
            reason="File deletion validated within workspace. Double confirmation is mandatory.",
            operation=op,
        )

    def _validate_search(self, op: CommandOperation) -> ValidationResult:
        """Validates SEARCH_FILES operation."""
        directory = op.arguments.get("directory") or op.arguments.get("directory_name") or "."
        is_safe, resolved, errors = self.validate_path(
            directory, must_exist=True, allow_directory=True, allow_file=False
        )
        if not is_safe or resolved is None:
            return ValidationResult(
                is_valid=False,
                validation_status=ValidationStatus.UNSAFE_PATH,
                operation_type=op.operation_type,
                risk_level=op.risk_level,
                validation_errors=errors,
                reason="Search directory is unsafe or does not exist.",
                operation=op,
            )

        pattern = op.arguments.get("pattern", "*.*")
        # Validate pattern against dangerous wildcard combinations
        if ";" in pattern or "|" in pattern or "&" in pattern:
            return ValidationResult(
                is_valid=False,
                validation_status=ValidationStatus.INVALID_ARGUMENT,
                operation_type=op.operation_type,
                validation_errors=["Search pattern contains invalid characters."],
                reason="Invalid pattern characters.",
                operation=op,
            )

        recurse = bool(op.arguments.get("recurse", False))

        return ValidationResult(
            is_valid=True,
            validation_status=ValidationStatus.VALID,
            operation_type=op.operation_type,
            risk_level=RiskLevel.LOW,
            requires_confirmation=False,
            validated_arguments={
                "directory": str(resolved),
                "pattern": pattern,
                "recurse": recurse,
            },
            target_path=str(resolved),
            reason="Read-only file search within permitted workspace.",
            operation=op,
        )

    def _validate_check_version(self, op: CommandOperation) -> ValidationResult:
        """Validates CHECK_VERSION operation against software allowlist."""
        from core.command_generator import CommandGenerator

        tool = op.arguments.get("version_name")
        if not tool:
            return ValidationResult(
                is_valid=False,
                validation_status=ValidationStatus.MISSING_PARAMETERS,
                operation_type=op.operation_type,
                risk_level=op.risk_level,
                validation_errors=["Software tool name is required for version check."],
                reason="Missing version tool name.",
                operation=op,
            )

        norm_tool = tool.lower().strip()
        cmd = CommandGenerator.ALLOWLISTED_VERSION_COMMANDS.get(norm_tool)
        if not cmd:
            return ValidationResult(
                is_valid=False,
                validation_status=ValidationStatus.BLOCKED_OPERATION,
                operation_type=op.operation_type,
                risk_level=op.risk_level,
                validation_errors=[f"Software '{tool}' is not in the approved version allowlist."],
                reason="Tool not allowlisted.",
                operation=op,
            )

        return ValidationResult(
            is_valid=True,
            validation_status=ValidationStatus.VALID,
            operation_type=op.operation_type,
            risk_level=RiskLevel.LOW,
            requires_confirmation=False,
            validated_arguments={"tool": norm_tool, "command": cmd},
            reason=f"Approved version check for allowlisted tool '{norm_tool}'.",
            operation=op,
        )

    def _validate_get_location(self, op: CommandOperation) -> ValidationResult:
        """Validates CURRENT_DIRECTORY (Get-Location) operation."""
        return ValidationResult(
            is_valid=True,
            validation_status=ValidationStatus.VALID,
            operation_type=op.operation_type,
            risk_level=RiskLevel.LOW,
            requires_confirmation=False,
            validated_arguments={},
            target_path=str(self.workspace_dir),
            reason="Safe read-only current directory query.",
            operation=op,
        )

    def _validate_move(self, op: CommandOperation) -> ValidationResult:
        """Validates MOVE_FILE operation."""
        src = op.arguments.get("source")
        dest = op.arguments.get("destination")

        if not src or not dest:
            return ValidationResult(
                is_valid=False,
                validation_status=ValidationStatus.MISSING_PARAMETERS,
                operation_type=op.operation_type,
                risk_level=op.risk_level,
                validation_errors=["Both source and destination are required for MOVE_FILE."],
                reason="Missing move arguments.",
                operation=op,
            )

        # Validate source (must exist within workspace)
        is_safe_src, res_src, err_src = self.validate_path(src, must_exist=True)
        if not is_safe_src or res_src is None:
            return ValidationResult(
                is_valid=False,
                validation_status=ValidationStatus.UNSAFE_PATH,
                operation_type=op.operation_type,
                risk_level=op.risk_level,
                validation_errors=[f"Source path error: {'; '.join(err_src)}"],
                reason="Source path is invalid or does not exist.",
                operation=op,
            )

        # Validate destination (must resolve within workspace)
        is_safe_dest, res_dest, err_dest = self.validate_path(dest, must_exist=False)
        if not is_safe_dest or res_dest is None:
            return ValidationResult(
                is_valid=False,
                validation_status=ValidationStatus.UNSAFE_PATH,
                operation_type=op.operation_type,
                risk_level=op.risk_level,
                validation_errors=[f"Destination path error: {'; '.join(err_dest)}"],
                reason="Destination path is invalid or escapes workspace.",
                operation=op,
            )

        # Conflict check: Destination cannot overwrite an existing file automatically
        if res_dest.exists() and res_dest.is_file():
            return ValidationResult(
                is_valid=False,
                validation_status=ValidationStatus.CONFLICT_TARGET_EXISTS,
                operation_type=op.operation_type,
                risk_level=op.risk_level,
                validation_errors=[f"Target file already exists at '{res_dest}'. Overwriting is blocked."],
                reason="Target file conflict.",
                operation=op,
            )

        return ValidationResult(
            is_valid=True,
            validation_status=ValidationStatus.VALID,
            operation_type=op.operation_type,
            risk_level=RiskLevel.MEDIUM,
            requires_confirmation=True,
            validated_arguments={"source": str(res_src), "destination": str(res_dest)},
            target_path=str(res_src),
            destination_path=str(res_dest),
            reason="Move operation validated within workspace boundary.",
            operation=op,
        )

    def _validate_copy(self, op: CommandOperation) -> ValidationResult:
        """Validates COPY_FILE operation."""
        src = op.arguments.get("source")
        dest = op.arguments.get("destination")

        if not src or not dest:
            return ValidationResult(
                is_valid=False,
                validation_status=ValidationStatus.MISSING_PARAMETERS,
                operation_type=op.operation_type,
                risk_level=op.risk_level,
                validation_errors=["Both source and destination are required for COPY_FILE."],
                reason="Missing copy arguments.",
                operation=op,
            )

        # Validate source (must exist within workspace)
        is_safe_src, res_src, err_src = self.validate_path(src, must_exist=True)
        if not is_safe_src or res_src is None:
            return ValidationResult(
                is_valid=False,
                validation_status=ValidationStatus.UNSAFE_PATH,
                operation_type=op.operation_type,
                risk_level=op.risk_level,
                validation_errors=[f"Source path error: {'; '.join(err_src)}"],
                reason="Source path is invalid or does not exist.",
                operation=op,
            )

        # Validate destination (must resolve within workspace)
        is_safe_dest, res_dest, err_dest = self.validate_path(dest, must_exist=False)
        if not is_safe_dest or res_dest is None:
            return ValidationResult(
                is_valid=False,
                validation_status=ValidationStatus.UNSAFE_PATH,
                operation_type=op.operation_type,
                risk_level=op.risk_level,
                validation_errors=[f"Destination path error: {'; '.join(err_dest)}"],
                reason="Destination path is invalid or escapes workspace.",
                operation=op,
            )

        # Conflict check: Destination cannot overwrite an existing file automatically
        if res_dest.exists() and res_dest.is_file():
            return ValidationResult(
                is_valid=False,
                validation_status=ValidationStatus.CONFLICT_TARGET_EXISTS,
                operation_type=op.operation_type,
                risk_level=op.risk_level,
                validation_errors=[f"Target file already exists at '{res_dest}'. Overwriting is blocked."],
                reason="Target file conflict.",
                operation=op,
            )

        return ValidationResult(
            is_valid=True,
            validation_status=ValidationStatus.VALID,
            operation_type=op.operation_type,
            risk_level=RiskLevel.MEDIUM,
            requires_confirmation=True,
            validated_arguments={"source": str(res_src), "destination": str(res_dest)},
            target_path=str(res_src),
            destination_path=str(res_dest),
            reason="Copy operation validated within workspace boundary.",
            operation=op,
        )

    def _validate_system_info(self, op: CommandOperation) -> ValidationResult:
        """Validates SYSTEM_INFO operation."""
        preview = op.powershell_preview or ""
        matched_cmd = None
        for allowed in self.ALLOWED_SYSTEM_COMMANDS:
            if allowed in preview:
                matched_cmd = allowed
                break

        if not matched_cmd:
            matched_cmd = "Get-ComputerInfo"

        return ValidationResult(
            is_valid=True,
            validation_status=ValidationStatus.VALID,
            operation_type=op.operation_type,
            risk_level=RiskLevel.LOW,
            requires_confirmation=False,
            validated_arguments={"command": matched_cmd},
            reason="Safe read-only system info query.",
            operation=op,
        )

    def _validate_network_info(self, op: CommandOperation) -> ValidationResult:
        """Validates NETWORK_INFO operation."""
        preview = op.powershell_preview or ""
        matched_cmd = None
        for allowed in self.ALLOWED_NETWORK_COMMANDS:
            if allowed in preview:
                matched_cmd = allowed
                break

        if not matched_cmd:
            matched_cmd = "ipconfig"

        return ValidationResult(
            is_valid=True,
            validation_status=ValidationStatus.VALID,
            operation_type=op.operation_type,
            risk_level=RiskLevel.LOW,
            requires_confirmation=False,
            validated_arguments={"command": matched_cmd},
            reason="Safe read-only network configuration query.",
            operation=op,
        )

    def _validate_create_file(self, op: CommandOperation) -> ValidationResult:
        """Validates CREATE_FILE operation."""
        file_name = op.arguments.get("file_name") or op.arguments.get("source")
        if not file_name:
            return ValidationResult(
                is_valid=False,
                validation_status=ValidationStatus.MISSING_PARAMETERS,
                operation_type=op.operation_type,
                risk_level=op.risk_level,
                validation_errors=["File name is required for CREATE_FILE."],
                reason="Missing file name.",
                operation=op,
            )

        is_safe, resolved, errors = self.validate_path(file_name, must_exist=False)
        if not is_safe or resolved is None:
            return ValidationResult(
                is_valid=False,
                validation_status=ValidationStatus.UNSAFE_PATH,
                operation_type=op.operation_type,
                risk_level=op.risk_level,
                validation_errors=errors,
                reason="Target file path is unsafe or violates workspace boundaries.",
                operation=op,
            )

        if resolved.exists():
            return ValidationResult(
                is_valid=False,
                validation_status=ValidationStatus.CONFLICT_TARGET_EXISTS,
                operation_type=op.operation_type,
                risk_level=op.risk_level,
                validation_errors=[f"File '{resolved}' already exists. Overwrite requires explicit decision."],
                reason="Target file already exists.",
                operation=op,
            )

        return ValidationResult(
            is_valid=True,
            validation_status=ValidationStatus.VALID,
            operation_type=op.operation_type,
            risk_level=RiskLevel.MEDIUM,
            requires_confirmation=True,
            validated_arguments={"file_path": str(resolved)},
            target_path=str(resolved),
            reason="File creation validated within workspace boundary.",
            operation=op,
        )

    def _validate_legacy_string(self, cmd_str: str) -> ValidationResult:
        """Validates legacy string input for Phase 1 compatibility."""
        clean = (cmd_str or "").strip()
        if not clean:
            return ValidationResult(
                is_valid=False,
                validation_status=ValidationStatus.INVALID_ARGUMENT,
                operation_type=OperationType.UNKNOWN,
                validation_errors=["Command string is empty."],
                reason="Empty command.",
            )

        # Basic dangerous pattern check on legacy strings
        dangerous_tokens = ["rm -rf", "format", "del /f", "drop table", "shutdown", "reg delete"]
        for d in dangerous_tokens:
            if d in clean.lower():
                return ValidationResult(
                    is_valid=False,
                    validation_status=ValidationStatus.BLOCKED_OPERATION,
                    operation_type=OperationType.UNKNOWN,
                    validation_errors=[f"Disallowed dangerous token detected: '{d}'"],
                    reason="Dangerous token blocked.",
                )

        return ValidationResult(
            is_valid=True,
            validation_status=ValidationStatus.VALID,
            operation_type=OperationType.UNKNOWN,
            validated_arguments={"raw_command": clean},
            reason="Legacy string command passed basic checks.",
        )
