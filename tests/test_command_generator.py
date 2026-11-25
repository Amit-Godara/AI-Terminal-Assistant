"""Unit tests for Phase 6 Intelligent PowerShell Command Generator.

Tests cover:
1. LIST_FILES generation (current directory, specific folder, hidden flag)
2. CREATE_DIRECTORY generation (valid name, path with spaces)
3. DELETE_FILE generation (high risk labeling, confirmation required, no destructive flags)
4. SEARCH_FILES generation (filter by extension, keyword, recursion flag)
5. CHECK_VERSION mappings (Python, Node.js, Git, Docker, PowerShell, etc.)
6. CURRENT_DIRECTORY generation (Get-Location)
7. MOVE_FILE generation (source and destination quoting, LiteralPath)
8. COPY_FILE generation (source and destination quoting, LiteralPath)
9. SYSTEM_INFO generation (Get-ComputerInfo, OS-specific, hostname)
10. NETWORK_INFO generation (ipconfig, Get-NetAdapter, Get-NetTCPConnection)
11. CREATE_FILE generation (New-Item -ItemType File)
12. UNKNOWN intent rejection (no command generated, UNSUPPORTED_INTENT)
13. Missing parameters (move without dest, copy without dest, delete without target, create dir without name, etc.)
14. Spaces in paths (safe quoting and preservation)
15. Windows absolute paths (C:\\... preservation)
16. Relative paths (.\\subfolder, ..\\backup)
17. Quoted filenames preservation
18. Special characters handling (brackets, hyphens, underscores)
19. Unsupported software name rejection
20. Ambiguous requests handling
21. Low-confidence NLP prediction rejection
22. Hybrid inference metadata verification
23. Destructive-operation risk labeling (DELETE -> HIGH, MOVE/COPY/CREATE -> MEDIUM, READ -> LOW)
24. No command generation for incomplete inputs (powershell_preview is None)
25. Verification that NO command execution occurs during generation
"""

import pytest
from unittest.mock import MagicMock

import config
from core.command_generator import (
    CommandGenerator,
    CommandOperation,
    OperationType,
    RiskLevel,
    GenerationStatus,
)


@pytest.fixture(scope="module")
def command_generator():
    """CommandGenerator instance using shared NLP components."""
    return CommandGenerator()


# 1. LIST_FILES
def test_1_list_files_current_directory(command_generator):
    """Test LIST_FILES without arguments generates Get-ChildItem."""
    op = command_generator.generate("list all files", intent="LIST_FILES", entities={})
    assert op.operation_type == OperationType.LIST
    assert op.powershell_preview == "Get-ChildItem"
    assert op.risk_level == RiskLevel.LOW
    assert op.requires_confirmation is False
    assert op.generation_status == GenerationStatus.READY_FOR_VALIDATION


def test_1_list_files_specific_directory(command_generator):
    """Test LIST_FILES with directory argument generates Get-ChildItem -LiteralPath."""
    op = command_generator.generate(
        "list files in C:\\Projects",
        intent="LIST_FILES",
        entities={"directory_name": "C:\\Projects"},
    )
    assert op.operation_type == OperationType.LIST
    assert op.powershell_preview == 'Get-ChildItem -LiteralPath "C:\\Projects"'
    assert op.generation_status == GenerationStatus.READY_FOR_VALIDATION


def test_1_list_files_hidden_files(command_generator):
    """Test LIST_FILES requesting hidden files includes -Force."""
    op = command_generator.generate(
        "show all hidden files",
        intent="LIST_FILES",
        entities={},
    )
    assert op.operation_type == OperationType.LIST
    assert op.powershell_preview == "Get-ChildItem -Force"


# 2. CREATE_DIRECTORY
def test_2_create_directory(command_generator):
    """Test CREATE_DIRECTORY generates New-Item -ItemType Directory."""
    op = command_generator.generate(
        "create a new directory named Assets",
        intent="CREATE_DIRECTORY",
        entities={"directory_name": "Assets"},
    )
    assert op.operation_type == OperationType.CREATE_DIR
    assert op.powershell_preview == 'New-Item -ItemType Directory -Path "Assets"'
    assert op.risk_level == RiskLevel.MEDIUM
    assert op.generation_status == GenerationStatus.READY_FOR_VALIDATION


# 3. DELETE_FILE
def test_3_delete_file_high_risk(command_generator):
    """Test DELETE_FILE is marked HIGH risk, requires confirmation, and uses LiteralPath."""
    op = command_generator.generate(
        "delete test.log",
        intent="DELETE_FILE",
        entities={"file_name": "test.log"},
    )
    assert op.operation_type == OperationType.DELETE
    assert op.powershell_preview == 'Remove-Item -LiteralPath "test.log"'
    assert op.risk_level == RiskLevel.HIGH
    assert op.requires_confirmation is True
    assert "-Recurse" not in op.powershell_preview
    assert "-Force" not in op.powershell_preview
    assert op.generation_status == GenerationStatus.READY_FOR_VALIDATION


# 4. SEARCH_FILES
def test_4_search_files_extension(command_generator):
    """Test SEARCH_FILES by file extension."""
    op = command_generator.generate(
        "find all python files",
        intent="SEARCH_FILES",
        entities={"file_extension": ".py"},
    )
    assert op.operation_type == OperationType.SEARCH
    assert op.powershell_preview == 'Get-ChildItem -Path "." -Filter "*.py" -File'
    assert op.risk_level == RiskLevel.LOW
    assert op.generation_status == GenerationStatus.READY_FOR_VALIDATION


def test_4_search_files_keyword_and_directory(command_generator):
    """Test SEARCH_FILES with keyword and directory."""
    op = command_generator.generate(
        "search for files containing error in logs",
        intent="SEARCH_FILES",
        entities={"keyword": "error", "directory_name": "logs"},
    )
    assert op.operation_type == OperationType.SEARCH
    assert op.powershell_preview == 'Get-ChildItem -Path "logs" -Filter "*error*" -File'


def test_4_search_files_recursive(command_generator):
    """Test SEARCH_FILES with explicit recursion flag."""
    op = command_generator.generate(
        "recursively search for config files in C:\\App",
        intent="SEARCH_FILES",
        entities={"keyword": "config", "directory_name": "C:\\App"},
    )
    assert "-Recurse" in op.powershell_preview


# 5. CHECK_VERSION
def test_5_check_version_allowlisted_tools(command_generator):
    """Test CHECK_VERSION for all allowlisted tools."""
    tools_expected = {
        "python": "python --version",
        "node": "node --version",
        "git": "git --version",
        "docker": "docker --version",
        "powershell": "$PSVersionTable.PSVersion",
        "npm": "npm --version",
        "java": "java --version",
        "rust": "rustc --version",
        "go": "go version",
        "gcc": "gcc --version",
        "dotnet": "dotnet --version",
    }
    for tool, expected_cmd in tools_expected.items():
        op = command_generator.generate(
            f"check {tool} version",
            intent="CHECK_VERSION",
            entities={"version_name": tool},
        )
        assert op.operation_type == OperationType.CHECK_VERSION
        assert op.powershell_preview == expected_cmd
        assert op.risk_level == RiskLevel.LOW
        assert op.generation_status == GenerationStatus.READY_FOR_VALIDATION


# 6. CURRENT_DIRECTORY
def test_6_current_directory(command_generator):
    """Test CURRENT_DIRECTORY generates Get-Location."""
    op = command_generator.generate(
        "what is my current directory",
        intent="CURRENT_DIRECTORY",
        entities={},
    )
    assert op.operation_type == OperationType.GET_LOCATION
    assert op.powershell_preview == "Get-Location"
    assert op.risk_level == RiskLevel.LOW


# 7. MOVE_FILE
def test_7_move_file(command_generator):
    """Test MOVE_FILE with source and destination."""
    op = command_generator.generate(
        "move data.csv to D:\\Backup",
        intent="MOVE_FILE",
        entities={"source": "data.csv", "destination": "D:\\Backup"},
    )
    assert op.operation_type == OperationType.MOVE
    assert op.powershell_preview == 'Move-Item -LiteralPath "data.csv" -Destination "D:\\Backup"'
    assert op.risk_level == RiskLevel.MEDIUM
    assert op.requires_confirmation is True


# 8. COPY_FILE
def test_8_copy_file(command_generator):
    """Test COPY_FILE with source and destination."""
    op = command_generator.generate(
        "copy notes.txt to C:\\Archive",
        intent="COPY_FILE",
        entities={"source": "notes.txt", "destination": "C:\\Archive"},
    )
    assert op.operation_type == OperationType.COPY
    assert op.powershell_preview == 'Copy-Item -LiteralPath "notes.txt" -Destination "C:\\Archive"'
    assert op.risk_level == RiskLevel.MEDIUM
    assert op.requires_confirmation is True


# 9. SYSTEM_INFO
def test_9_system_info_general(command_generator):
    """Test SYSTEM_INFO general query generates Get-ComputerInfo."""
    op = command_generator.generate(
        "show system hardware information",
        intent="SYSTEM_INFO",
        entities={},
    )
    assert op.operation_type == OperationType.SYSTEM_INFO
    assert op.powershell_preview == "Get-ComputerInfo"
    assert op.risk_level == RiskLevel.LOW


def test_9_system_info_os_specific(command_generator):
    """Test SYSTEM_INFO OS-specific query generates Get-CimInstance Win32_OperatingSystem."""
    op = command_generator.generate(
        "what is my operating system version",
        intent="SYSTEM_INFO",
        entities={},
    )
    assert "Win32_OperatingSystem" in op.powershell_preview


def test_9_system_info_hostname(command_generator):
    """Test SYSTEM_INFO hostname query generates hostname."""
    op = command_generator.generate(
        "what is the computer hostname",
        intent="SYSTEM_INFO",
        entities={},
    )
    assert op.powershell_preview == "hostname"


# 10. NETWORK_INFO
def test_10_network_info_ipconfig(command_generator):
    """Test NETWORK_INFO IP configuration generates ipconfig."""
    op = command_generator.generate(
        "show my ip address",
        intent="NETWORK_INFO",
        entities={},
    )
    assert op.operation_type == OperationType.NETWORK_INFO
    assert op.powershell_preview == "ipconfig"


def test_10_network_info_adapters(command_generator):
    """Test NETWORK_INFO adapter query generates Get-NetAdapter."""
    op = command_generator.generate(
        "show network adapter hardware",
        intent="NETWORK_INFO",
        entities={},
    )
    assert op.powershell_preview == "Get-NetAdapter"


def test_10_network_info_connections(command_generator):
    """Test NETWORK_INFO ports/connection query generates Get-NetTCPConnection."""
    op = command_generator.generate(
        "show listening ports and tcp connections",
        intent="NETWORK_INFO",
        entities={},
    )
    assert op.powershell_preview == "Get-NetTCPConnection"


# 11. CREATE_FILE
def test_11_create_file(command_generator):
    """Test CREATE_FILE generates New-Item -ItemType File."""
    op = command_generator.generate(
        "create a new file named output.log",
        intent="CREATE_FILE",
        entities={"file_name": "output.log"},
    )
    assert op.operation_type == OperationType.CREATE_FILE
    assert op.powershell_preview == 'New-Item -ItemType File -Path "output.log"'
    assert op.risk_level == RiskLevel.MEDIUM


# 12. UNKNOWN
def test_12_unknown_rejection(command_generator):
    """Test UNKNOWN intent produces no preview and UNSUPPORTED_INTENT status."""
    op = command_generator.generate(
        "tell me a joke",
        intent="UNKNOWN",
        entities={},
    )
    assert op.operation_type == OperationType.UNKNOWN
    assert op.powershell_preview is None
    assert op.generation_status == GenerationStatus.UNSUPPORTED_INTENT


# 13. Missing parameters
def test_13_missing_parameters_move_without_destination(command_generator):
    """Test MOVE_FILE without destination returns NEEDS_CLARIFICATION."""
    op = command_generator.generate(
        "move report.pdf",
        intent="MOVE_FILE",
        entities={"source": "report.pdf"},
    )
    assert op.powershell_preview is None
    assert op.generation_status == GenerationStatus.NEEDS_CLARIFICATION
    assert "destination" in op.missing_parameters
    assert op.clarification_prompt is not None


def test_13_missing_parameters_copy_without_destination(command_generator):
    """Test COPY_FILE without destination returns NEEDS_CLARIFICATION."""
    op = command_generator.generate(
        "copy notes.txt",
        intent="COPY_FILE",
        entities={"source": "notes.txt"},
    )
    assert op.powershell_preview is None
    assert op.generation_status == GenerationStatus.NEEDS_CLARIFICATION
    assert "destination" in op.missing_parameters


def test_13_missing_parameters_create_directory(command_generator):
    """Test CREATE_DIRECTORY without directory_name returns NEEDS_CLARIFICATION."""
    op = command_generator.generate(
        "create a new directory",
        intent="CREATE_DIRECTORY",
        entities={},
    )
    assert op.powershell_preview is None
    assert op.generation_status == GenerationStatus.NEEDS_CLARIFICATION
    assert "directory_name" in op.missing_parameters


def test_13_missing_parameters_delete_file(command_generator):
    """Test DELETE_FILE without file_name returns NEEDS_CLARIFICATION."""
    op = command_generator.generate(
        "delete a file",
        intent="DELETE_FILE",
        entities={},
    )
    assert op.powershell_preview is None
    assert op.generation_status == GenerationStatus.NEEDS_CLARIFICATION
    assert "file_name" in op.missing_parameters


def test_13_missing_parameters_check_version(command_generator):
    """Test CHECK_VERSION without version_name returns NEEDS_CLARIFICATION."""
    op = command_generator.generate(
        "check the installed version",
        intent="CHECK_VERSION",
        entities={},
    )
    assert op.powershell_preview is None
    assert op.generation_status == GenerationStatus.NEEDS_CLARIFICATION
    assert "version_name" in op.missing_parameters


# 14. Spaces in paths
def test_14_spaces_in_paths(command_generator):
    """Test paths containing spaces are preserved and correctly quoted."""
    op = command_generator.generate(
        "move report.pdf to C:\\My Documents\\Archived Reports",
        intent="MOVE_FILE",
        entities={"source": "annual report.pdf", "destination": "C:\\My Documents\\Archived Reports"},
    )
    assert op.powershell_preview == 'Move-Item -LiteralPath "annual report.pdf" -Destination "C:\\My Documents\\Archived Reports"'


# 15. Windows absolute paths
def test_15_windows_absolute_paths(command_generator):
    """Test absolute Windows drive paths are maintained."""
    op = command_generator.generate(
        "create directory C:\\Projects\\AI-Assistant",
        intent="CREATE_DIRECTORY",
        entities={"directory_name": "C:\\Projects\\AI-Assistant"},
    )
    assert op.powershell_preview == 'New-Item -ItemType Directory -Path "C:\\Projects\\AI-Assistant"'


# 16. Relative paths
def test_16_relative_paths(command_generator):
    """Test relative paths (. and ..) are preserved."""
    op = command_generator.generate(
        "copy .\\config.json to ..\\backup\\config.json",
        intent="COPY_FILE",
        entities={"source": ".\\config.json", "destination": "..\\backup\\config.json"},
    )
    assert op.powershell_preview == 'Copy-Item -LiteralPath ".\\config.json" -Destination "..\\backup\\config.json"'


# 17. Quoted filenames
def test_17_quoted_filenames_preservation(command_generator):
    """Test input with internal quotes is safely handled without double wrapping."""
    op = command_generator.generate(
        'delete "temp file.txt"',
        intent="DELETE_FILE",
        entities={"file_name": '"temp file.txt"'},
    )
    assert op.powershell_preview == 'Remove-Item -LiteralPath "temp file.txt"'


# 18. Special characters handling
def test_18_special_characters_handling(command_generator):
    """Test filenames with brackets, hyphens, and multiple dots."""
    op = command_generator.generate(
        "delete data[1]_v2.0.final.csv",
        intent="DELETE_FILE",
        entities={"file_name": "data[1]_v2.0.final.csv"},
    )
    assert op.powershell_preview == 'Remove-Item -LiteralPath "data[1]_v2.0.final.csv"'


# 19. Unsupported software names
def test_19_unsupported_software_name(command_generator):
    """Test unsupported software runtime rejection."""
    op = command_generator.generate(
        "check ruby version",
        intent="CHECK_VERSION",
        entities={"version_name": "ruby"},
    )
    assert op.powershell_preview is None
    assert op.generation_status == GenerationStatus.UNSUPPORTED_INTENT
    assert "allowlist" in op.clarification_prompt.lower()


# 20. Empty input
def test_20_empty_input(command_generator):
    """Test empty input string handling."""
    for empty_val in ("", "   ", None):
        op = command_generator.generate(empty_val)
        assert op.operation_type == OperationType.UNKNOWN
        assert op.powershell_preview is None
        assert op.generation_status == GenerationStatus.NEEDS_CLARIFICATION


# 21. Low-confidence prediction
def test_21_low_confidence_rejection(command_generator):
    """Test that low-confidence NLP outputs reject command preview."""
    # When intent is UNKNOWN from NLP pipeline
    op = command_generator.generate("asdfghjkl random noise 12345")
    assert op.powershell_preview is None
    assert op.generation_status in (GenerationStatus.UNSUPPORTED_INTENT, GenerationStatus.LOW_CONFIDENCE)


# 22. End-to-end NLP integration
def test_22_end_to_end_nlp_integration(command_generator):
    """Test end-to-end integration: query -> hybrid inference -> entity extraction -> command generation."""
    query = "where am I currently located"
    op = command_generator.generate(query)
    assert op.operation_type == OperationType.GET_LOCATION
    assert op.powershell_preview == "Get-Location"
    assert op.generation_status == GenerationStatus.READY_FOR_VALIDATION
    assert "confidence" in op.inference_metadata


# 23. Destructive-operation risk labeling
def test_23_risk_level_assignment(command_generator):
    """Test risk levels: DELETE -> HIGH, MOVE/COPY/CREATE -> MEDIUM, READ -> LOW."""
    op_del = command_generator.generate("delete a.txt", intent="DELETE_FILE", entities={"file_name": "a.txt"})
    assert op_del.risk_level == RiskLevel.HIGH
    assert op_del.requires_confirmation is True

    op_mov = command_generator.generate("move a.txt to b", intent="MOVE_FILE", entities={"source": "a.txt", "destination": "b"})
    assert op_mov.risk_level == RiskLevel.MEDIUM

    op_lst = command_generator.generate("list files", intent="LIST_FILES", entities={})
    assert op_lst.risk_level == RiskLevel.LOW
    assert op_lst.requires_confirmation is False


# 24. No command generation for incomplete inputs
def test_24_no_command_for_incomplete_inputs(command_generator):
    """Verify powershell_preview is strictly None when required arguments are missing."""
    incomplete_queries = [
        ("move file", "MOVE_FILE", {}),
        ("copy file", "COPY_FILE", {}),
        ("create folder", "CREATE_DIRECTORY", {}),
        ("delete file", "DELETE_FILE", {}),
        ("create file", "CREATE_FILE", {}),
    ]
    for q, intent, ents in incomplete_queries:
        op = command_generator.generate(q, intent=intent, entities=ents)
        assert op.powershell_preview is None, f"Expected None for incomplete query '{q}', got {op.powershell_preview}"
        assert op.generation_status == GenerationStatus.NEEDS_CLARIFICATION


# 25. No command execution during generation
def test_25_no_execution_occurs(command_generator, monkeypatch):
    """Verifies that subprocess and execution functions are never called during generation."""
    import subprocess
    import os

    mock_run = MagicMock()
    monkeypatch.setattr(subprocess, "run", mock_run)
    monkeypatch.setattr(subprocess, "Popen", mock_run)
    monkeypatch.setattr(os, "system", mock_run)

    # Generate various commands
    queries = [
        "list files",
        "delete confidential.txt",
        "move data.csv to C:\\Backup",
        "show ip address",
        "check python version",
    ]
    for q in queries:
        op = command_generator.generate(q)
        assert isinstance(op, CommandOperation)

    # Assert execution was never invoked
    mock_run.assert_not_called()
