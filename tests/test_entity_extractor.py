"""Unit tests for core.entity_extractor.EntityExtractor."""

import pytest
from core.entity_extractor import EntityExtractor, EntityExtractionResult, ExtractedEntity


@pytest.fixture(scope="module")
def extractor() -> EntityExtractor:
    """Fixture providing an EntityExtractor instance."""
    return EntityExtractor()


def test_1_entity_type_file_name(extractor: EntityExtractor) -> None:
    """1. Test extraction of exact filename."""
    res = extractor.extract("Delete file notes.txt", intent="DELETE_FILE")
    assert "FILE_NAME" in res.entities_by_type
    assert res.entities_by_type["FILE_NAME"] == ["notes.txt"]


def test_2_entity_type_file_extension(extractor: EntityExtractor) -> None:
    """2. Test extraction of exact file extension."""
    res = extractor.extract("Search for all .pdf documents", intent="SEARCH_FILES")
    assert "FILE_EXTENSION" in res.entities_by_type
    assert ".pdf" in res.entities_by_type["FILE_EXTENSION"]


def test_3_entity_type_directory_name(extractor: EntityExtractor) -> None:
    """3. Test extraction of directory name."""
    res = extractor.extract("Create a folder named MachineLearning", intent="CREATE_DIRECTORY")
    assert "DIRECTORY_NAME" in res.entities_by_type
    assert res.entities_by_type["DIRECTORY_NAME"] == ["MachineLearning"]


def test_4_entity_type_source_path(extractor: EntityExtractor) -> None:
    """4. Test extraction of source path."""
    res = extractor.extract("Move C:\\Projects\\report.pdf into D:\\Backup", intent="MOVE_FILE")
    assert "SOURCE_PATH" in res.entities_by_type
    assert res.entities_by_type["SOURCE_PATH"] == ["C:\\Projects\\report.pdf"]


def test_5_entity_type_destination_path(extractor: EntityExtractor) -> None:
    """5. Test extraction of destination path."""
    res = extractor.extract("Move report.pdf to C:\\Archive", intent="MOVE_FILE")
    assert "DESTINATION_PATH" in res.entities_by_type
    assert res.entities_by_type["DESTINATION_PATH"] == ["C:\\Archive"]


def test_6_entity_type_keyword(extractor: EntityExtractor) -> None:
    """6. Test extraction of search keyword."""
    res = extractor.extract("Find files containing keyword error", intent="SEARCH_FILES")
    assert "KEYWORD" in res.entities_by_type
    assert "error" in res.entities_by_type["KEYWORD"]


def test_7_entity_type_version_name(extractor: EntityExtractor) -> None:
    """7. Test extraction of version software names."""
    res = extractor.extract("Check installed Python version", intent="CHECK_VERSION")
    assert "VERSION_NAME" in res.entities_by_type
    assert "Python" in res.entities_by_type["VERSION_NAME"]


def test_8_entity_type_version_name_nodejs(extractor: EntityExtractor) -> None:
    """8. Test extraction of Node.js version name (not as .js filename)."""
    res = extractor.extract("Show the installed Node.js version", intent="CHECK_VERSION")
    assert "VERSION_NAME" in res.entities_by_type
    assert res.entities_by_type["VERSION_NAME"] == ["Node.js"]


def test_9_entity_type_number(extractor: EntityExtractor) -> None:
    """9. Test extraction of standalone numeric parameter."""
    res = extractor.extract("Show the first 10 files", intent="LIST_FILES")
    assert "NUMBER" in res.entities_by_type
    assert res.entities_by_type["NUMBER"] == ["10"]


def test_10_windows_absolute_path(extractor: EntityExtractor) -> None:
    """10. Test Windows absolute directory path extraction."""
    res = extractor.extract("List all files in C:\\Data\\Reports", intent="LIST_FILES")
    assert "DIRECTORY_NAME" in res.entities_by_type
    assert "C:\\Data\\Reports" in res.entities_by_type["DIRECTORY_NAME"]


def test_11_relative_path(extractor: EntityExtractor) -> None:
    """11. Test relative directory path extraction."""
    res = extractor.extract("Display items in .\\subfolder", intent="LIST_FILES")
    assert "DIRECTORY_NAME" in res.entities_by_type
    assert ".\\subfolder" in res.entities_by_type["DIRECTORY_NAME"]


def test_12_path_with_spaces(extractor: EntityExtractor) -> None:
    """12. Test path containing spaces preservation."""
    res = extractor.extract("Move Report.PDF to C:\\Users\\Amit\\My Documents", intent="MOVE_FILE")
    assert "DESTINATION_PATH" in res.entities_by_type
    assert res.entities_by_type["DESTINATION_PATH"] == ["C:\\Users\\Amit\\My Documents"]


def test_13_quoted_filename(extractor: EntityExtractor) -> None:
    """13. Test extraction of quoted filename with spaces."""
    res = extractor.extract("Delete file 'annual report.pdf'", intent="DELETE_FILE")
    assert "FILE_NAME" in res.entities_by_type
    assert "annual report.pdf" in res.entities_by_type["FILE_NAME"]


def test_14_quoted_directory(extractor: EntityExtractor) -> None:
    """14. Test extraction of quoted directory with spaces."""
    res = extractor.extract("Create directory \"Client Assets\"", intent="CREATE_DIRECTORY")
    assert "DIRECTORY_NAME" in res.entities_by_type
    assert "Client Assets" in res.entities_by_type["DIRECTORY_NAME"]


def test_15_directory_with_numbers(extractor: EntityExtractor) -> None:
    """15. Test directory name containing numbers."""
    res = extractor.extract("Make a directory called Financial Reports 2026", intent="CREATE_DIRECTORY")
    assert "DIRECTORY_NAME" in res.entities_by_type
    assert "Financial Reports 2026" in res.entities_by_type["DIRECTORY_NAME"]


def test_16_multiple_dot_filename(extractor: EntityExtractor) -> None:
    """16. Test extraction of filename containing multiple dots."""
    res = extractor.extract("Remove file data_v1.0.final.csv", intent="DELETE_FILE")
    assert "FILE_NAME" in res.entities_by_type
    assert "data_v1.0.final.csv" in res.entities_by_type["FILE_NAME"]
    assert ".csv" in res.entities_by_type["FILE_EXTENSION"]


def test_17_search_pattern_wildcard(extractor: EntityExtractor) -> None:
    """17. Test wildcard search pattern parsing."""
    res = extractor.extract("Search for files matching config_*.json", intent="SEARCH_FILES")
    assert "KEYWORD" in res.entities_by_type
    assert "config" in res.entities_by_type["KEYWORD"]
    assert "config_*.json" in res.entities_by_type["KEYWORD"]
    assert ".json" in res.entities_by_type["FILE_EXTENSION"]


def test_18_mapped_file_extension(extractor: EntityExtractor) -> None:
    """18. Test natural language mapping from technology word to file extension."""
    res = extractor.extract("Find all Python files", intent="SEARCH_FILES")
    assert "FILE_EXTENSION" in res.entities_by_type
    assert ".py" in res.entities_by_type["FILE_EXTENSION"]


def test_19_casing_preservation(extractor: EntityExtractor) -> None:
    """19. Test that original entity casing is preserved without lowercasing."""
    res = extractor.extract("Move ImportantData.XLSX to D:\\ArchiveFolder", intent="MOVE_FILE")
    assert res.entities_by_type["FILE_NAME"] == ["ImportantData.XLSX"]
    assert res.entities_by_type["DESTINATION_PATH"] == ["D:\\ArchiveFolder"]


def test_20_copy_file_extraction(extractor: EntityExtractor) -> None:
    """20. Test copy command entity extraction."""
    res = extractor.extract("Copy notes.txt to Documents", intent="COPY_FILE")
    assert res.entities_by_type["FILE_NAME"] == ["notes.txt"]
    assert res.entities_by_type["DESTINATION_PATH"] == ["Documents"]
    assert res.status == "SUCCESS"


def test_21_create_file_named(extractor: EntityExtractor) -> None:
    """21. Test file creation with full filename and extension."""
    res = extractor.extract("Create a file named setup.py", intent="CREATE_FILE")
    assert res.entities_by_type["FILE_NAME"] == ["setup.py"]
    assert res.entities_by_type["FILE_EXTENSION"] == [".py"]
    assert res.status == "SUCCESS"


def test_22_delete_file_direct(extractor: EntityExtractor) -> None:
    """22. Test delete file direct extraction."""
    res = extractor.extract("Delete obsolete_data.csv", intent="DELETE_FILE")
    assert res.entities_by_type["FILE_NAME"] == ["obsolete_data.csv"]
    assert res.entities_by_type["FILE_EXTENSION"] == [".csv"]


def test_23_missing_param_move_destination(extractor: EntityExtractor) -> None:
    """23. Test detection of missing destination in MOVE_FILE."""
    res = extractor.extract("Move report.pdf", intent="MOVE_FILE")
    assert "destination_path" in res.missing_parameters
    assert res.status == "INCOMPLETE_PARAMETERS"


def test_24_missing_param_copy_destination(extractor: EntityExtractor) -> None:
    """24. Test detection of missing destination in COPY_FILE."""
    res = extractor.extract("Copy notes.txt", intent="COPY_FILE")
    assert "destination_path" in res.missing_parameters
    assert res.status == "INCOMPLETE_PARAMETERS"


def test_25_missing_param_create_directory(extractor: EntityExtractor) -> None:
    """25. Test detection of missing directory name in CREATE_DIRECTORY."""
    res = extractor.extract("Create a folder", intent="CREATE_DIRECTORY")
    assert "directory_name" in res.missing_parameters
    assert res.status == "INCOMPLETE_PARAMETERS"


def test_26_missing_param_create_file(extractor: EntityExtractor) -> None:
    """26. Test detection of missing filename in CREATE_FILE."""
    res = extractor.extract("Create a file", intent="CREATE_FILE")
    assert "file_name" in res.missing_parameters
    assert res.status == "INCOMPLETE_PARAMETERS"


def test_27_missing_param_delete_file(extractor: EntityExtractor) -> None:
    """27. Test detection of missing filename in DELETE_FILE."""
    res = extractor.extract("Delete a file", intent="DELETE_FILE")
    assert "file_name" in res.missing_parameters
    assert res.status == "INCOMPLETE_PARAMETERS"


def test_28_missing_param_check_version(extractor: EntityExtractor) -> None:
    """28. Test detection of missing software in CHECK_VERSION."""
    res = extractor.extract("Check version", intent="CHECK_VERSION")
    assert "version_name" in res.missing_parameters
    assert res.status == "INCOMPLETE_PARAMETERS"


def test_29_false_positive_prevention(extractor: EntityExtractor) -> None:
    """29. Test that generic words like directory/file are not extracted as filenames."""
    res = extractor.extract("List all files in the current directory", intent="LIST_FILES")
    assert "FILE_NAME" not in res.entities_by_type
    assert res.status == "NO_ENTITIES"


def test_30_unsupported_query(extractor: EntityExtractor) -> None:
    """30. Test out-of-domain query produces no entities."""
    res = extractor.extract("Play some music for me", intent="UNKNOWN")
    assert len(res.entities) == 0
    assert res.status == "NO_ENTITIES"


def test_31_empty_and_none_input(extractor: EntityExtractor) -> None:
    """31. Test handling of None and empty string."""
    for empty_val in [None, "", "   \t\n "]:
        res = extractor.extract(empty_val)
        assert isinstance(res, EntityExtractionResult)
        assert len(res.entities) == 0
        assert res.status == "NO_ENTITIES"


def test_32_character_offsets_exactness(extractor: EntityExtractor) -> None:
    """32. Test that entity character offsets slice the exact original query text."""
    query = "Move C:\\Projects\\Report_2026.PDF to D:\\My Backup"
    res = extractor.extract(query, intent="MOVE_FILE")
    for ent in res.entities:
        assert query[ent.start:ent.end] == ent.value, (
            f"Offset mismatch for {ent.entity_type}: expected '{ent.value}', "
            f"got '{query[ent.start:ent.end]}'"
        )


def test_33_pydantic_serialization(extractor: EntityExtractor) -> None:
    """33. Test Pydantic model serialization and dictionary conversion."""
    res = extractor.extract("Delete obsolete_data.csv", intent="DELETE_FILE")
    dumped = res.model_dump()
    assert isinstance(dumped, dict)
    assert dumped["status"] == "SUCCESS"
    assert "FILE_NAME" in dumped["entities_by_type"]
    assert "obsolete_data.csv" in dumped["entities_by_type"]["FILE_NAME"]
