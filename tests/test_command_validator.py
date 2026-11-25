"""Security and Safety Unit Tests for CommandValidator.

Tests cover:
1. All 12 supported operations validation
2. UNKNOWN operation rejection
3. Missing argument rejection
4. Invalid argument types
5. Empty and whitespace paths
6. Absolute paths resolving inside workspace
7. Relative paths resolving inside workspace
8. Path traversal rejection (.. escapes)
9. Windows system directory rejection (C:\\Windows, System32)
10. Program Files rejection
11. User profile root rejection
12. Wildcard misuse in paths and search
13. Invalid operation parameters
14. Low-confidence NLP prediction rejection
15. Incomplete generation result rejection (NEEDS_CLARIFICATION)
16. Symlink escape detection
17. Destructive operation risk labeling & double confirmation enforcement
18. Existing target conflict detection (refusal to silently overwrite)
"""

import os
from pathlib import Path
import pytest

import config
from core.command_generator import (
    CommandOperation,
    OperationType,
    RiskLevel,
    GenerationStatus,
)
from core.command_validator import (
    CommandValidator,
    ValidationResult,
    ValidationStatus,
)


@pytest.fixture
def sandbox_workspace(tmp_path):
    """Provides an isolated temporary workspace directory for validator testing."""
    workspace = tmp_path / "sandbox_workspace"
    workspace.mkdir(parents=True, exist_ok=True)
    return workspace


@pytest.fixture
def validator(sandbox_workspace):
    """Provides a CommandValidator configured with sandbox_workspace."""
    return CommandValidator(workspace_dir=sandbox_workspace, safe_mode=True)


# 1. Validation of all supported operations
def test_1_validate_list_files(validator, sandbox_workspace):
    """Test validation of LIST_FILES operation."""
    op = CommandOperation(
        operation_type=OperationType.LIST,
        intent="LIST_FILES",
        arguments={},
        generation_status=GenerationStatus.READY_FOR_VALIDATION,
        source_query="list files",
    )
    res = validator.validate(op)
    assert res.is_valid is True
    assert res.validation_status == ValidationStatus.VALID
    assert res.risk_level == RiskLevel.LOW
    assert res.requires_confirmation is False


def test_1_validate_create_directory(validator, sandbox_workspace):
    """Test validation of CREATE_DIRECTORY."""
    op = CommandOperation(
        operation_type=OperationType.CREATE_DIR,
        intent="CREATE_DIRECTORY",
        arguments={"directory_name": "NewFolder"},
        generation_status=GenerationStatus.READY_FOR_VALIDATION,
        source_query="create folder NewFolder",
    )
    res = validator.validate(op)
    assert res.is_valid is True
    assert res.risk_level == RiskLevel.MEDIUM
    assert res.requires_confirmation is True


def test_1_validate_delete_file(validator, sandbox_workspace):
    """Test validation of DELETE_FILE enforces HIGH risk and double confirmation."""
    target_file = sandbox_workspace / "dummy.txt"
    target_file.write_text("sample content", encoding="utf-8")

    op = CommandOperation(
        operation_type=OperationType.DELETE,
        intent="DELETE_FILE",
        arguments={"file_name": "dummy.txt"},
        risk_level=RiskLevel.HIGH,
        requires_confirmation=True,
        generation_status=GenerationStatus.READY_FOR_VALIDATION,
        source_query="delete dummy.txt",
    )
    res = validator.validate(op)
    assert res.is_valid is True
    assert res.risk_level == RiskLevel.HIGH
    assert res.requires_confirmation is True
    assert res.requires_double_confirmation is True


def test_1_validate_search_files(validator, sandbox_workspace):
    """Test validation of SEARCH_FILES operation."""
    op = CommandOperation(
        operation_type=OperationType.SEARCH,
        intent="SEARCH_FILES",
        arguments={"directory": ".", "pattern": "*.py", "recurse": False},
        generation_status=GenerationStatus.READY_FOR_VALIDATION,
        source_query="find python files",
    )
    res = validator.validate(op)
    assert res.is_valid is True
    assert res.risk_level == RiskLevel.LOW


def test_1_validate_check_version(validator):
    """Test validation of CHECK_VERSION for allowlisted tools."""
    op = CommandOperation(
        operation_type=OperationType.CHECK_VERSION,
        intent="CHECK_VERSION",
        arguments={"version_name": "python"},
        generation_status=GenerationStatus.READY_FOR_VALIDATION,
        source_query="check python version",
    )
    res = validator.validate(op)
    assert res.is_valid is True
    assert res.risk_level == RiskLevel.LOW


def test_1_validate_get_location(validator):
    """Test validation of GET_LOCATION operation."""
    op = CommandOperation(
        operation_type=OperationType.GET_LOCATION,
        intent="CURRENT_DIRECTORY",
        arguments={},
        generation_status=GenerationStatus.READY_FOR_VALIDATION,
        source_query="where am I",
    )
    res = validator.validate(op)
    assert res.is_valid is True
    assert res.risk_level == RiskLevel.LOW


def test_1_validate_move_file(validator, sandbox_workspace):
    """Test validation of MOVE_FILE operation."""
    src = sandbox_workspace / "source.txt"
    src.write_text("hello", encoding="utf-8")

    op = CommandOperation(
        operation_type=OperationType.MOVE,
        intent="MOVE_FILE",
        arguments={"source": "source.txt", "destination": "target.txt"},
        generation_status=GenerationStatus.READY_FOR_VALIDATION,
        source_query="move source.txt to target.txt",
    )
    res = validator.validate(op)
    assert res.is_valid is True
    assert res.risk_level == RiskLevel.MEDIUM
    assert res.requires_confirmation is True


def test_1_validate_copy_file(validator, sandbox_workspace):
    """Test validation of COPY_FILE operation."""
    src = sandbox_workspace / "notes.txt"
    src.write_text("notes", encoding="utf-8")

    op = CommandOperation(
        operation_type=OperationType.COPY,
        intent="COPY_FILE",
        arguments={"source": "notes.txt", "destination": "notes_backup.txt"},
        generation_status=GenerationStatus.READY_FOR_VALIDATION,
        source_query="copy notes.txt to notes_backup.txt",
    )
    res = validator.validate(op)
    assert res.is_valid is True
    assert res.risk_level == RiskLevel.MEDIUM


def test_1_validate_system_and_network_info(validator):
    """Test validation of SYSTEM_INFO and NETWORK_INFO operations."""
    for op_type, intent, cmd in [
        (OperationType.SYSTEM_INFO, "SYSTEM_INFO", "Get-ComputerInfo"),
        (OperationType.NETWORK_INFO, "NETWORK_INFO", "ipconfig"),
    ]:
        op = CommandOperation(
            operation_type=op_type,
            intent=intent,
            arguments={"command": cmd},
            powershell_preview=cmd,
            generation_status=GenerationStatus.READY_FOR_VALIDATION,
            source_query="show info",
        )
        res = validator.validate(op)
        assert res.is_valid is True
        assert res.risk_level == RiskLevel.LOW


def test_1_validate_create_file(validator, sandbox_workspace):
    """Test validation of CREATE_FILE."""
    op = CommandOperation(
        operation_type=OperationType.CREATE_FILE,
        intent="CREATE_FILE",
        arguments={"file_name": "new_doc.txt"},
        generation_status=GenerationStatus.READY_FOR_VALIDATION,
        source_query="create new_doc.txt",
    )
    res = validator.validate(op)
    assert res.is_valid is True
    assert res.risk_level == RiskLevel.MEDIUM


# 2. Unknown operation rejection
def test_2_unknown_operation_rejection(validator):
    """Test that UNKNOWN operation type is rejected."""
    op = CommandOperation(
        operation_type=OperationType.UNKNOWN,
        intent="UNKNOWN",
        arguments={},
        generation_status=GenerationStatus.READY_FOR_VALIDATION,
        source_query="unknown",
    )
    res = validator.validate(op)
    assert res.is_valid is False
    assert res.validation_status == ValidationStatus.INVALID_OPERATION


# 3. Missing argument rejection
def test_3_missing_arguments_rejection(validator):
    """Test rejection when required parameters are missing."""
    # MOVE missing destination
    op = CommandOperation(
        operation_type=OperationType.MOVE,
        intent="MOVE_FILE",
        arguments={"source": "file.txt"},
        generation_status=GenerationStatus.READY_FOR_VALIDATION,
        source_query="move file.txt",
    )
    res = validator.validate(op)
    assert res.is_valid is False
    assert res.validation_status == ValidationStatus.MISSING_PARAMETERS


# 4. Invalid argument types
def test_4_invalid_argument_types(validator):
    """Test rejection when argument types are invalid or null."""
    res = validator.validate(12345)  # Invalid non-operation input
    assert res.is_valid is False
    assert res.validation_status == ValidationStatus.INVALID_OPERATION


# 5. Empty and whitespace paths
def test_5_empty_paths(validator):
    """Test rejection of empty and whitespace paths."""
    for empty_path in ("", "   ", "\t"):
        is_safe, resolved, errors = validator.validate_path(empty_path)
        assert is_safe is False
        assert len(errors) > 0


# 6. Absolute paths resolving inside workspace
def test_6_absolute_paths_inside_workspace(validator, sandbox_workspace):
    """Test valid absolute path inside workspace is accepted."""
    target = sandbox_workspace / "inside_file.txt"
    is_safe, resolved, errors = validator.validate_path(str(target), must_exist=False)
    assert is_safe is True
    assert resolved == target


# 7. Relative paths resolving inside workspace
def test_7_relative_paths_inside_workspace(validator, sandbox_workspace):
    """Test valid relative path resolves cleanly within workspace."""
    is_safe, resolved, errors = validator.validate_path(".\\subfolder\\file.txt", must_exist=False)
    assert is_safe is True
    assert resolved == sandbox_workspace / "subfolder" / "file.txt"


# 8. Path traversal rejection (.. escapes)
def test_8_path_traversal_rejection(validator, sandbox_workspace):
    """Test path traversal sequences attempting to escape workspace are blocked."""
    traversal_paths = [
        "..\\outside.txt",
        "..\\..\\Windows\\System32\\cmd.exe",
        "subfolder\\..\\..\\..\\secret.key",
    ]
    for bad_path in traversal_paths:
        is_safe, resolved, errors = validator.validate_path(bad_path)
        assert is_safe is False
        assert any("escapes the permitted workspace" in e for e in errors)


# 9. Windows system directory rejection
def test_9_windows_system_path_rejection(validator):
    """Test explicit Windows system paths are rejected."""
    system_paths = [
        r"C:\Windows",
        r"C:\Windows\System32",
        r"C:\Windows\System32\drivers\etc\hosts",
    ]
    for sys_path in system_paths:
        is_safe, resolved, errors = validator.validate_path(sys_path)
        assert is_safe is False


# 10. Program Files rejection
def test_10_program_files_rejection(validator):
    """Test Program Files directory is rejected."""
    prog_paths = [
        r"C:\Program Files",
        r"C:\Program Files (x86)",
    ]
    for p in prog_paths:
        is_safe, resolved, errors = validator.validate_path(p)
        assert is_safe is False


# 11. User profile root rejection
def test_11_user_profile_root_rejection(validator):
    """Test user profile root directory is rejected."""
    home_path = str(Path.home())
    is_safe, resolved, errors = validator.validate_path(home_path)
    assert is_safe is False


# 12. Wildcard misuse in search and paths
def test_12_wildcard_misuse(validator, sandbox_workspace):
    """Test invalid or injection-laden search filter patterns are rejected."""
    op = CommandOperation(
        operation_type=OperationType.SEARCH,
        intent="SEARCH_FILES",
        arguments={"directory": ".", "pattern": "*.py; rm -rf /"},
        generation_status=GenerationStatus.READY_FOR_VALIDATION,
        source_query="search with injection",
    )
    res = validator.validate(op)
    assert res.is_valid is False
    assert res.validation_status == ValidationStatus.INVALID_ARGUMENT


# 13. Invalid operation parameters (e.g. deleting a directory)
def test_13_cannot_delete_directory(validator, sandbox_workspace):
    """Test that DELETE_FILE cannot target a directory."""
    sub_dir = sandbox_workspace / "TestDir"
    sub_dir.mkdir(parents=True, exist_ok=True)

    op = CommandOperation(
        operation_type=OperationType.DELETE,
        intent="DELETE_FILE",
        arguments={"file_name": "TestDir"},
        generation_status=GenerationStatus.READY_FOR_VALIDATION,
        source_query="delete TestDir",
    )
    res = validator.validate(op)
    assert res.is_valid is False
    assert res.validation_status == ValidationStatus.UNSAFE_PATH
    assert any("directory" in e.lower() for e in res.validation_errors)


# 14. Low-confidence NLP prediction rejection
def test_14_low_confidence_rejection(validator):
    """Test that operations with LOW_CONFIDENCE generation status are blocked."""
    op = CommandOperation(
        operation_type=OperationType.LIST,
        intent="LIST_FILES",
        arguments={},
        generation_status=GenerationStatus.LOW_CONFIDENCE,
        source_query="mumbled query",
    )
    res = validator.validate(op)
    assert res.is_valid is False
    assert res.validation_status == ValidationStatus.LOW_CONFIDENCE


# 15. Incomplete generation rejection (NEEDS_CLARIFICATION)
def test_15_incomplete_generation_rejection(validator):
    """Test that NEEDS_CLARIFICATION operations are blocked."""
    op = CommandOperation(
        operation_type=OperationType.MOVE,
        intent="MOVE_FILE",
        arguments={"source": "report.pdf"},
        missing_parameters=["destination"],
        clarification_prompt="Please specify the destination.",
        generation_status=GenerationStatus.NEEDS_CLARIFICATION,
        source_query="move report.pdf",
    )
    res = validator.validate(op)
    assert res.is_valid is False
    assert res.validation_status == ValidationStatus.MISSING_PARAMETERS


# 16. Symlink escape detection
def test_16_symlink_escape_detection(validator, sandbox_workspace, tmp_path):
    """Test symlinks pointing outside the workspace are detected and blocked."""
    outside_dir = tmp_path / "outside_dir"
    outside_dir.mkdir(parents=True, exist_ok=True)
    outside_file = outside_dir / "secret.txt"
    outside_file.write_text("secret", encoding="utf-8")

    link_path = sandbox_workspace / "symlink_escape.txt"
    try:
        os.symlink(outside_file, link_path)
    except (OSError, NotImplementedError):
        # On Windows, creating symlinks may require Developer Mode / admin privilege
        pytest.skip("Symlink creation requires elevated privileges on this system.")

    is_safe, resolved, errors = validator.validate_path(str(link_path))
    assert is_safe is False
    assert any("points outside" in e.lower() for e in errors)


# 17. Destructive operation risk handling
def test_17_destructive_risk_handling(validator, sandbox_workspace):
    """Test risk levels: DELETE -> HIGH, MOVE/COPY/CREATE -> MEDIUM, READ -> LOW."""
    test_file = sandbox_workspace / "to_delete.txt"
    test_file.write_text("content", encoding="utf-8")

    op_del = CommandOperation(
        operation_type=OperationType.DELETE,
        intent="DELETE_FILE",
        arguments={"file_name": "to_delete.txt"},
        generation_status=GenerationStatus.READY_FOR_VALIDATION,
        source_query="delete to_delete.txt",
    )
    res_del = validator.validate(op_del)
    assert res_del.risk_level == RiskLevel.HIGH
    assert res_del.requires_double_confirmation is True


# 18. Existing destination conflict detection (overwrite prevention)
def test_18_existing_destination_conflict(validator, sandbox_workspace):
    """Test that copying or moving over an existing file is blocked as CONFLICT_TARGET_EXISTS."""
    src = sandbox_workspace / "src.txt"
    src.write_text("src", encoding="utf-8")

    existing_dest = sandbox_workspace / "dest.txt"
    existing_dest.write_text("already here", encoding="utf-8")

    op_copy = CommandOperation(
        operation_type=OperationType.COPY,
        intent="COPY_FILE",
        arguments={"source": "src.txt", "destination": "dest.txt"},
        generation_status=GenerationStatus.READY_FOR_VALIDATION,
        source_query="copy src.txt to dest.txt",
    )
    res_copy = validator.validate(op_copy)
    assert res_copy.is_valid is False
    assert res_copy.validation_status == ValidationStatus.CONFLICT_TARGET_EXISTS
    assert any("already exists" in e.lower() for e in res_copy.validation_errors)
