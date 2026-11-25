"""Entity Extractor module for AI Terminal Assistant.

Combines spaCy EntityRuler, regular expressions, and contextual rules to extract
parameters (FILE_NAME, FILE_EXTENSION, DIRECTORY_NAME, SOURCE_PATH, DESTINATION_PATH,
KEYWORD, VERSION_NAME, NUMBER) from natural language terminal queries.
"""

import os
import re
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple, Set

from pydantic import BaseModel, Field
import spacy
from spacy.pipeline import EntityRuler


class ExtractedEntity(BaseModel):
    """Structured representation of an extracted entity."""
    entity_type: str
    value: str
    start: int
    end: int
    confidence: float = 1.0
    context: Optional[str] = None


class EntityExtractionResult(BaseModel):
    """Structured result of entity extraction across a user query."""
    query: str
    intent: Optional[str] = None
    entities: List[ExtractedEntity] = Field(default_factory=list)
    entities_by_type: Dict[str, List[str]] = Field(default_factory=dict)
    missing_parameters: List[str] = Field(default_factory=list)
    missing_parameter_descriptions: Dict[str, str] = Field(default_factory=dict)
    status: str = "SUCCESS"  # "SUCCESS", "INCOMPLETE_PARAMETERS", "NO_ENTITIES"


class EntityExtractor:
    """Extracts structured terminal command parameters from raw natural language queries."""

    VERSION_TOOLS = [
        "python",
        "python3",
        "git",
        "node",
        "node.js",
        "nodejs",
        "docker",
        "powershell",
        "npm",
        "rust",
        "gcc",
        "go",
        "java",
        "dotnet",
        "code",
        "pip",
        "pip3",
    ]

    FILE_TYPE_MAPPINGS = {
        "python": ".py",
        "py": ".py",
        "text": ".txt",
        "txt": ".txt",
        "pdf": ".pdf",
        "json": ".json",
        "csv": ".csv",
        "spreadsheet": ".csv",
        "spreadsheets": ".csv",
        "excel": ".xlsx",
        "xlsx": ".xlsx",
        "word": ".docx",
        "docx": ".docx",
        "markdown": ".md",
        "md": ".md",
        "log": ".log",
        "logs": ".log",
        "javascript": ".js",
        "js": ".js",
        "html": ".html",
        "css": ".css",
        "image": ".png",
        "images": ".png",
        "png": ".png",
        "backup": ".bak",
        "zip": ".zip",
        "archive": ".zip",
    }

    # Filler words that must not be extracted as filenames or directories
    GENERIC_WORDS = {
        "file",
        "files",
        "folder",
        "folders",
        "directory",
        "directories",
        "path",
        "paths",
        "contents",
        "content",
        "item",
        "items",
        "document",
        "documents",
        "everything",
        "all",
        "this",
        "current",
        "present",
        "here",
        "there",
        "location",
        "system",
        "machine",
        "pc",
        "computer",
        "network",
        "connection",
        "ip",
        "address",
        "adapter",
        "version",
        "build",
        "info",
        "information",
        "keyword",
        "keywords",
        "me",
        "my",
        "you",
        "we",
        "it",
        "the",
        "a",
        "an",
    }

    def __init__(self) -> None:
        """Initializes the entity extraction engine with spaCy EntityRuler and custom rules."""
        self.nlp = spacy.blank("en")
        self.ruler: EntityRuler = self.nlp.add_pipe("entity_ruler")
        self._setup_entity_ruler_patterns()

    def _setup_entity_ruler_patterns(self) -> None:
        """Registers controlled pattern rules in spaCy EntityRuler."""
        patterns = []
        for tool in self.VERSION_TOOLS:
            patterns.append(
                {"label": "VERSION_NAME", "pattern": [{"LOWER": tool}]}
            )
        self.ruler.add_patterns(patterns)

    def extract(self, query: Optional[str], intent: Optional[str] = None) -> EntityExtractionResult:
        """Extracts all recognized entities from the original query, considering intent context.

        Args:
            query: Original natural-language query preserving casing.
            intent: Optional predicted intent string.

        Returns:
            Structured EntityExtractionResult.
        """
        if query is None or not query.strip():
            return EntityExtractionResult(
                query=query or "",
                intent=intent,
                entities=[],
                entities_by_type={},
                missing_parameters=[],
                missing_parameter_descriptions={},
                status="NO_ENTITIES",
            )

        extracted_entities: List[ExtractedEntity] = []

        def add_entity(entity: ExtractedEntity) -> bool:
            # Check for duplicate or conflicting entity spans
            for existing in extracted_entities:
                if (
                    existing.entity_type == entity.entity_type
                    and existing.start == entity.start
                    and existing.end == entity.end
                ):
                    return False
                # Do not let FILE_NAME or FILE_EXTENSION overwrite a VERSION_NAME e.g. Node.js
                if existing.entity_type == "VERSION_NAME" and entity.entity_type in ("FILE_NAME", "FILE_EXTENSION"):
                    if not (entity.end <= existing.start or entity.start >= existing.end):
                        return False
                # Do not let a partial DIRECTORY_NAME duplicate an encompassing DESTINATION_PATH or SOURCE_PATH
                if entity.entity_type == "DIRECTORY_NAME" and existing.entity_type in ("DESTINATION_PATH", "SOURCE_PATH"):
                    if existing.start <= entity.start and entity.end <= existing.end:
                        return False
            extracted_entities.append(entity)
            return True

        # -------------------------------------------------------------
        # 1. Version Names (for CHECK_VERSION or explicit version queries)
        # -------------------------------------------------------------
        if intent == "CHECK_VERSION" or re.search(r"\b(version|build|installed|runtime)\b", query, re.I):
            self._extract_version_names(query, intent, add_entity)

        # -------------------------------------------------------------
        # 2. Contextual Intent Extraction (MOVE_FILE, COPY_FILE)
        # -------------------------------------------------------------
        if intent in ("MOVE_FILE", "COPY_FILE") or re.search(r"\b(move|copy|transfer|shift|relocate|duplicate|clone)\b", query, re.I):
            self._extract_move_copy_entities(query, add_entity)

        # -------------------------------------------------------------
        # 3. Contextual Directory Creation (CREATE_DIRECTORY)
        # -------------------------------------------------------------
        if intent == "CREATE_DIRECTORY" or re.search(r"\b(create|make|set up|generate|initialize)\s+(?:a\s+)?(?:new\s+)?(?:directory|folder)\b", query, re.I):
            self._extract_create_directory_entity(query, add_entity)

        # -------------------------------------------------------------
        # 4. Contextual File Creation (CREATE_FILE)
        # -------------------------------------------------------------
        if intent == "CREATE_FILE" or re.search(r"\b(create|make|generate)\s+(?:a\s+)?(?:new\s+)?(?:blank\s+|empty\s+)?(?:file|document)\b", query, re.I):
            self._extract_create_file_entity(query, add_entity)

        # -------------------------------------------------------------
        # 5. Contextual File Deletion (DELETE_FILE)
        # -------------------------------------------------------------
        if intent == "DELETE_FILE" or re.search(r"\b(delete|remove|erase|trash|purge|get rid of)\b", query, re.I):
            self._extract_delete_file_entity(query, add_entity)

        # -------------------------------------------------------------
        # 6. Contextual File Search / Patterns (SEARCH_FILES)
        # -------------------------------------------------------------
        if intent == "SEARCH_FILES" or re.search(r"\b(find|search|look for|locate)\b", query, re.I):
            self._extract_search_entities(query, add_entity)

        # -------------------------------------------------------------
        # 7. General Structured Regex Extractions
        # -------------------------------------------------------------
        # A. Windows Absolute Paths (e.g. C:\Users\Documents, C:\Users\Amit\My Documents)
        self._extract_windows_paths(query, add_entity)

        # B. Relative Paths (e.g. .\logs\output.log, ..\backup)
        self._extract_relative_paths(query, add_entity)

        # C. Quoted Filenames and Directories
        self._extract_quoted_entities(query, intent, add_entity)

        # D. Filenames with explicit extensions (e.g. notes.txt, annual_report.pdf)
        self._extract_filenames_with_extensions(query, add_entity)

        # E. Standalone File Extensions (e.g. .py, .pdf, *.json)
        self._extract_standalone_extensions(query, add_entity)

        # F. Numbers (e.g. first 10 files)
        self._extract_numbers(query, add_entity)

        # Sort entities by start character offset
        extracted_entities.sort(key=lambda e: (e.start, e.end))

        # Build entities_by_type grouping (deduplicated preserving order)
        entities_by_type: Dict[str, List[str]] = {}
        for ent in extracted_entities:
            existing_vals = entities_by_type.setdefault(ent.entity_type, [])
            if ent.value not in existing_vals:
                existing_vals.append(ent.value)

        # -------------------------------------------------------------
        # 8. Missing Parameter Detection based on Intent
        # -------------------------------------------------------------
        missing_params, missing_descs = self._detect_missing_parameters(
            query=query,
            intent=intent,
            entities_by_type=entities_by_type,
        )

        status = "SUCCESS"
        if missing_params:
            status = "INCOMPLETE_PARAMETERS"
        elif not extracted_entities:
            status = "NO_ENTITIES"

        return EntityExtractionResult(
            query=query,
            intent=intent,
            entities=extracted_entities,
            entities_by_type=entities_by_type,
            missing_parameters=missing_params,
            missing_parameter_descriptions=missing_descs,
            status=status,
        )

    # -----------------------------------------------------------------
    # Helper Extraction Methods
    # -----------------------------------------------------------------

    def _extract_move_copy_entities(self, query: str, add_fn) -> None:
        """Extracts source and destination paths/files from move/copy queries."""
        pat = re.compile(
            r"\b(?:move|copy|transfer|shift|relocate|duplicate|clone)\s+"
            r"(?:the\s+)?(?:file|document|item\s+)?(?P<src>['\"][^'\"]+['\"]|[a-zA-Z]:\\[^\s]+|\.{1,2}[/\\][^\s]+|\S+)"
            r"(?:\s+(?:to|into|in)\s+(?P<dst>.+))?",
            re.IGNORECASE,
        )
        m = pat.search(query)
        if m:
            src_raw = m.group("src").strip().strip("'\"")
            dst_match = m.group("dst")

            # Validate source
            if src_raw.lower() not in self.GENERIC_WORDS:
                src_start = m.start("src")
                src_end = m.end("src")
                if "\\" in src_raw or "/" in src_raw or re.match(r"^[a-zA-Z]:", src_raw):
                    add_fn(
                        ExtractedEntity(
                            entity_type="SOURCE_PATH",
                            value=src_raw,
                            start=src_start,
                            end=src_end,
                            confidence=1.0,
                            context="move_copy_source_path",
                        )
                    )
                    fname = Path(src_raw).name
                    if "." in fname and fname.lower() not in self.GENERIC_WORDS:
                        f_start = query.find(fname, src_start)
                        add_fn(
                            ExtractedEntity(
                                entity_type="FILE_NAME",
                                value=fname,
                                start=f_start,
                                end=f_start + len(fname),
                                confidence=1.0,
                                context="source_filename",
                            )
                        )
                        ext = "." + fname.split(".")[-1]
                        e_start = query.find(ext, f_start)
                        add_fn(
                            ExtractedEntity(
                                entity_type="FILE_EXTENSION",
                                value=ext,
                                start=e_start,
                                end=e_start + len(ext),
                                confidence=1.0,
                                context="file_extension",
                            )
                        )
                else:
                    add_fn(
                        ExtractedEntity(
                            entity_type="FILE_NAME",
                            value=src_raw,
                            start=src_start,
                            end=src_end,
                            confidence=1.0,
                            context="move_copy_source_filename",
                        )
                    )
                    if "." in src_raw:
                        ext = "." + src_raw.split(".")[-1]
                        e_start = query.find(ext, src_start)
                        add_fn(
                            ExtractedEntity(
                                entity_type="FILE_EXTENSION",
                                value=ext,
                                start=e_start,
                                end=e_start + len(ext),
                                confidence=1.0,
                                context="file_extension",
                            )
                        )

            # Validate destination if present
            if dst_match:
                dst_raw = dst_match.strip().strip("'\"").rstrip(".,;!?")
                if dst_raw and dst_raw.lower() not in ("folder", "directory", "file"):
                    dst_start = query.find(dst_raw, m.start("dst"))
                    dst_end = dst_start + len(dst_raw)
                    add_fn(
                        ExtractedEntity(
                            entity_type="DESTINATION_PATH",
                            value=dst_raw,
                            start=dst_start,
                            end=dst_end,
                            confidence=1.0,
                            context="move_copy_destination",
                        )
                    )

    def _extract_create_directory_entity(self, query: str, add_fn) -> None:
        """Extracts directory name or path from create directory queries."""
        named_pat = re.compile(
            r"\b(?:named|called|at)\s+['\"]?([^'\"\n\r\?!,;]+)['\"]?",
            re.IGNORECASE,
        )
        m = named_pat.search(query)
        if m:
            val = m.group(1).strip().strip("'\"")
            if val.lower() not in self.GENERIC_WORDS:
                start = query.find(val, m.start(1))
                add_fn(
                    ExtractedEntity(
                        entity_type="DIRECTORY_NAME",
                        value=val,
                        start=start,
                        end=start + len(val),
                        confidence=1.0,
                        context="named_directory",
                    )
                )
                return

        direct_pat = re.compile(
            r"\b(?:create|make|generate|set up|initialize)\s+(?:a\s+)?(?:new\s+)?(?:directory|folder)\s+['\"]?([^'\"\n\r\?!,;]+)['\"]?",
            re.IGNORECASE,
        )
        m2 = direct_pat.search(query)
        if m2:
            val = m2.group(1).strip().strip("'\"")
            if val.lower() not in self.GENERIC_WORDS:
                start = query.find(val, m2.start(1))
                add_fn(
                    ExtractedEntity(
                        entity_type="DIRECTORY_NAME",
                        value=val,
                        start=start,
                        end=start + len(val),
                        confidence=1.0,
                        context="direct_directory",
                    )
                )

    def _extract_create_file_entity(self, query: str, add_fn) -> None:
        """Extracts filename and extension from create file queries."""
        named_pat = re.compile(
            r"\b(?:named|called|at)\s+['\"]?([^'\"\n\r\?!,;]+)['\"]?",
            re.IGNORECASE,
        )
        m = named_pat.search(query)
        if m:
            val = m.group(1).strip().strip("'\"")
            if val.lower() not in self.GENERIC_WORDS:
                start = query.find(val, m.start(1))
                add_fn(
                    ExtractedEntity(
                        entity_type="FILE_NAME",
                        value=val,
                        start=start,
                        end=start + len(val),
                        confidence=1.0,
                        context="named_create_file",
                    )
                )
                if "." in val:
                    ext = "." + val.split(".")[-1]
                    e_start = query.find(ext, start)
                    add_fn(
                        ExtractedEntity(
                            entity_type="FILE_EXTENSION",
                            value=ext,
                            start=e_start,
                            end=e_start + len(ext),
                            confidence=1.0,
                            context="file_extension",
                        )
                    )
                return

        direct_pat = re.compile(
            r"\b(?:create|make|generate)\s+(?:a\s+)?(?:new\s+)?(?:blank\s+|empty\s+)?(?:file|document)\s+['\"]?([^'\"\n\r\?!,;]+)['\"]?",
            re.IGNORECASE,
        )
        m2 = direct_pat.search(query)
        if m2:
            val = m2.group(1).strip().strip("'\"")
            if val.lower() not in self.GENERIC_WORDS:
                start = query.find(val, m2.start(1))
                add_fn(
                    ExtractedEntity(
                        entity_type="FILE_NAME",
                        value=val,
                        start=start,
                        end=start + len(val),
                        confidence=1.0,
                        context="direct_create_file",
                    )
                )
                if "." in val:
                    ext = "." + val.split(".")[-1]
                    e_start = query.find(ext, start)
                    add_fn(
                        ExtractedEntity(
                            entity_type="FILE_EXTENSION",
                            value=ext,
                            start=e_start,
                            end=e_start + len(ext),
                            confidence=1.0,
                            context="file_extension",
                        )
                    )

    def _extract_delete_file_entity(self, query: str, add_fn) -> None:
        """Extracts filename to delete."""
        named_pat = re.compile(
            r"\b(?:named|called)\s+['\"]?([^'\"\n\r\?!,;]+)['\"]?",
            re.IGNORECASE,
        )
        m = named_pat.search(query)
        if m:
            val = m.group(1).strip().strip("'\"")
            if val.lower() not in self.GENERIC_WORDS:
                start = query.find(val, m.start(1))
                add_fn(
                    ExtractedEntity(
                        entity_type="FILE_NAME",
                        value=val,
                        start=start,
                        end=start + len(val),
                        confidence=1.0,
                        context="named_delete_file",
                    )
                )
                if "." in val:
                    ext = "." + val.split(".")[-1]
                    e_start = query.find(ext, start)
                    add_fn(
                        ExtractedEntity(
                            entity_type="FILE_EXTENSION",
                            value=ext,
                            start=e_start,
                            end=e_start + len(ext),
                            confidence=1.0,
                            context="file_extension",
                        )
                    )
                return

        del_pat = re.compile(
            r"\b(?:delete|remove|erase|trash|purge|get rid of)\s+(?:the\s+)?(?:file|document|item|log file)?\s*['\"]?([^'\"\n\r\?!,;]+)['\"]?",
            re.IGNORECASE,
        )
        m2 = del_pat.search(query)
        if m2:
            val = m2.group(1).strip().strip("'\"")
            if val.lower() not in self.GENERIC_WORDS and not re.match(r"^(a|an|the|this|that|all)\b", val, re.I):
                start = query.find(val, m2.start(1))
                add_fn(
                    ExtractedEntity(
                        entity_type="FILE_NAME",
                        value=val,
                        start=start,
                        end=start + len(val),
                        confidence=1.0,
                        context="direct_delete_file",
                    )
                )
                if "." in val:
                    ext = "." + val.split(".")[-1]
                    e_start = query.find(ext, start)
                    add_fn(
                        ExtractedEntity(
                            entity_type="FILE_EXTENSION",
                            value=ext,
                            start=e_start,
                            end=e_start + len(ext),
                            confidence=1.0,
                            context="file_extension",
                        )
                    )

    def _extract_search_entities(self, query: str, add_fn) -> None:
        """Extracts search keywords, patterns, or language-mapped extensions."""
        pattern_regex = re.compile(r"\b([a-zA-Z0-9_\-\*\?]+\.[a-zA-Z0-9_\-\*\?]+)\b")
        for m in pattern_regex.finditer(query):
            pattern_str = m.group(1)
            # If pattern has prefix keyword before wildcard, extract keyword e.g. config from config_*.json
            if "_" in pattern_str:
                prefix = pattern_str.split("_")[0]
                if prefix and prefix not in ("*", "?"):
                    add_fn(
                        ExtractedEntity(
                            entity_type="KEYWORD",
                            value=prefix,
                            start=m.start(1),
                            end=m.start(1) + len(prefix),
                            confidence=1.0,
                            context="pattern_prefix_keyword",
                        )
                    )
            add_fn(
                ExtractedEntity(
                    entity_type="KEYWORD",
                    value=pattern_str,
                    start=m.start(1),
                    end=m.end(1),
                    confidence=1.0,
                    context="search_wildcard_pattern",
                )
            )
            if "." in pattern_str:
                ext = "." + pattern_str.split(".")[-1].replace("*", "").replace("?", "")
                if len(ext) > 1:
                    add_fn(
                        ExtractedEntity(
                            entity_type="FILE_EXTENSION",
                            value=ext,
                            start=m.start(1) + pattern_str.rfind("."),
                            end=m.end(1),
                            confidence=1.0,
                            context="pattern_extension",
                        )
                    )

        kw_pat = re.compile(
            r"\b(?:containing\s+keyword|keyword\s+named|keyword|containing|term)\s+['\"]?([a-zA-Z0-9_\-]+)['\"]?",
            re.I,
        )
        for m in kw_pat.finditer(query):
            kw = m.group(1).strip()
            if kw.lower() not in self.GENERIC_WORDS:
                add_fn(
                    ExtractedEntity(
                        entity_type="KEYWORD",
                        value=kw,
                        start=m.start(1),
                        end=m.end(1),
                        confidence=1.0,
                        context="explicit_keyword",
                    )
                )

        for type_word, ext in self.FILE_TYPE_MAPPINGS.items():
            type_pat = re.compile(rf"\b({type_word})\s+(?:files|file|documents|document)\b", re.I)
            m = type_pat.search(query)
            if m:
                add_fn(
                    ExtractedEntity(
                        entity_type="FILE_EXTENSION",
                        value=ext,
                        start=m.start(1),
                        end=m.end(1),
                        confidence=1.0,
                        context=f"mapped_from_{type_word}",
                    )
                )

    def _extract_windows_paths(self, query: str, add_fn) -> None:
        r"""Extracts Windows absolute paths (e.g. C:\Users\Documents, D:\Backup)."""
        win_path_pat = re.compile(
            r"\b([a-zA-Z]:[\\/][^\s\"'\n\r<>|?*]+)",
            re.IGNORECASE,
        )
        for m in win_path_pat.finditer(query):
            val = m.group(1).rstrip(".,;!?")
            if re.search(r"\.[a-zA-Z0-9]{1,10}$", val):
                add_fn(
                    ExtractedEntity(
                        entity_type="SOURCE_PATH",
                        value=val,
                        start=m.start(1),
                        end=m.start(1) + len(val),
                        confidence=1.0,
                        context="windows_file_path",
                    )
                )
                fname = Path(val).name
                if fname.lower() not in self.GENERIC_WORDS:
                    f_start = query.find(fname, m.start(1))
                    add_fn(
                        ExtractedEntity(
                            entity_type="FILE_NAME",
                            value=fname,
                            start=f_start,
                            end=f_start + len(fname),
                            confidence=1.0,
                            context="path_filename",
                        )
                    )
                    ext = "." + fname.split(".")[-1]
                    e_start = query.find(ext, f_start)
                    add_fn(
                        ExtractedEntity(
                            entity_type="FILE_EXTENSION",
                            value=ext,
                            start=e_start,
                            end=e_start + len(ext),
                            confidence=1.0,
                            context="file_extension",
                        )
                    )
            else:
                add_fn(
                    ExtractedEntity(
                        entity_type="DIRECTORY_NAME",
                        value=val,
                        start=m.start(1),
                        end=m.start(1) + len(val),
                        confidence=1.0,
                        context="windows_directory_path",
                    )
                )

    def _extract_relative_paths(self, query: str, add_fn) -> None:
        r"""Extracts relative paths (e.g. .\logs\output.log, ..\backup, ./src)."""
        rel_path_pat = re.compile(
            r"(\.{1,2}[\\/][^\"'\n\r\s<>|?*]+)",
            re.IGNORECASE,
        )
        for m in rel_path_pat.finditer(query):
            val = m.group(1).rstrip(".,;!?")
            if re.search(r"\.[a-zA-Z0-9]{1,10}$", val):
                add_fn(
                    ExtractedEntity(
                        entity_type="SOURCE_PATH",
                        value=val,
                        start=m.start(1),
                        end=m.start(1) + len(val),
                        confidence=1.0,
                        context="relative_file_path",
                    )
                )
                fname = Path(val).name
                if fname.lower() not in self.GENERIC_WORDS:
                    f_start = query.find(fname, m.start(1))
                    add_fn(
                        ExtractedEntity(
                            entity_type="FILE_NAME",
                            value=fname,
                            start=f_start,
                            end=f_start + len(fname),
                            confidence=1.0,
                            context="path_filename",
                        )
                    )
            else:
                add_fn(
                    ExtractedEntity(
                        entity_type="DIRECTORY_NAME",
                        value=val,
                        start=m.start(1),
                        end=m.start(1) + len(val),
                        confidence=1.0,
                        context="relative_directory_path",
                    )
                )

    def _extract_quoted_entities(self, query: str, intent: Optional[str], add_fn) -> None:
        """Extracts quoted strings as filenames, directories, or keywords."""
        quote_pat = re.compile(r"['\"]([^'\"\n\r]+)['\"]")
        for m in quote_pat.finditer(query):
            val = m.group(1).strip()
            if not val or val.lower() in self.GENERIC_WORDS:
                continue

            start = m.start(1)
            end = m.end(1)

            if "." in val and re.search(r"\.[a-zA-Z0-9]{1,10}$", val):
                add_fn(
                    ExtractedEntity(
                        entity_type="FILE_NAME",
                        value=val,
                        start=start,
                        end=end,
                        confidence=1.0,
                        context="quoted_filename",
                    )
                )
                ext = "." + val.split(".")[-1]
                e_start = query.find(ext, start)
                add_fn(
                    ExtractedEntity(
                        entity_type="FILE_EXTENSION",
                        value=ext,
                        start=e_start,
                        end=e_start + len(ext),
                        confidence=1.0,
                        context="file_extension",
                    )
                )
            elif intent == "CREATE_DIRECTORY":
                add_fn(
                    ExtractedEntity(
                        entity_type="DIRECTORY_NAME",
                        value=val,
                        start=start,
                        end=end,
                        confidence=1.0,
                        context="quoted_directory",
                    )
                )
            elif intent in ("MOVE_FILE", "COPY_FILE"):
                add_fn(
                    ExtractedEntity(
                        entity_type="DESTINATION_PATH",
                        value=val,
                        start=start,
                        end=end,
                        confidence=1.0,
                        context="quoted_path",
                    )
                )
            else:
                add_fn(
                    ExtractedEntity(
                        entity_type="KEYWORD",
                        value=val,
                        start=start,
                        end=end,
                        confidence=0.85,
                        context="quoted_keyword",
                    )
                )

    def _extract_filenames_with_extensions(self, query: str, add_fn) -> None:
        """Extracts unquoted filenames containing valid file extensions."""
        fname_pat = re.compile(
            r"\b([a-zA-Z0-9_\-][a-zA-Z0-9_\-\.]*\.[a-zA-Z0-9]{1,10})\b"
        )
        for m in fname_pat.finditer(query):
            val = m.group(1).rstrip(".,;!?")
            if val.lower() in self.GENERIC_WORDS or val.lower() in self.VERSION_TOOLS:
                continue

            ext = "." + val.split(".")[-1]
            start = m.start(1)
            end = start + len(val)

            add_fn(
                ExtractedEntity(
                    entity_type="FILE_NAME",
                    value=val,
                    start=start,
                    end=end,
                    confidence=1.0,
                    context="filename_with_extension",
                )
            )
            e_start = query.find(ext, start)
            add_fn(
                ExtractedEntity(
                    entity_type="FILE_EXTENSION",
                    value=ext,
                    start=e_start,
                    end=e_start + len(ext),
                    confidence=1.0,
                    context="file_extension",
                )
            )

    def _extract_standalone_extensions(self, query: str, add_fn) -> None:
        """Extracts standalone extensions (e.g. .py, .pdf, *.json)."""
        ext_pat = re.compile(r"(\.[a-zA-Z0-9]{1,10})\b")
        for m in ext_pat.finditer(query):
            val = m.group(1)
            if val.startswith(("./", ".\\")):
                continue
            add_fn(
                ExtractedEntity(
                    entity_type="FILE_EXTENSION",
                    value=val,
                    start=m.start(1),
                    end=m.end(1),
                    confidence=1.0,
                    context="standalone_extension",
                )
            )

    def _extract_version_names(self, query: str, intent: Optional[str], add_fn) -> None:
        """Uses spaCy EntityRuler to extract version tool names."""
        doc = self.nlp(query)
        for ent in doc.ents:
            if ent.label_ == "VERSION_NAME":
                val = ent.text.strip()
                canonical_map = {
                    "node.js": "Node.js",
                    "nodejs": "Node.js",
                    "node": "Node.js",
                    "python": "Python",
                    "python3": "Python",
                    "git": "git",
                    "docker": "docker",
                    "powershell": "powershell",
                    "rust": "rust",
                    "gcc": "gcc",
                    "npm": "npm",
                    "dotnet": "dotnet",
                    "java": "java",
                    "go": "go",
                    "pip": "pip",
                }
                display_val = canonical_map.get(val.lower(), val)
                add_fn(
                    ExtractedEntity(
                        entity_type="VERSION_NAME",
                        value=display_val,
                        start=ent.start_char,
                        end=ent.end_char,
                        confidence=1.0,
                        context="spacy_entity_ruler",
                    )
                )

    def _extract_numbers(self, query: str, add_fn) -> None:
        """Extracts standalone numeric arguments (e.g. 10 files)."""
        num_pat = re.compile(r"\b(\d+)\b")
        for m in num_pat.finditer(query):
            val = m.group(1)
            add_fn(
                ExtractedEntity(
                    entity_type="NUMBER",
                    value=val,
                    start=m.start(1),
                    end=m.end(1),
                    confidence=1.0,
                    context="numeric_argument",
                )
            )

    def _detect_missing_parameters(
        self,
        query: str,
        intent: Optional[str],
        entities_by_type: Dict[str, List[str]],
    ) -> Tuple[List[str], Dict[str, str]]:
        """Identifies mandatory parameters missing for the specific intent.

        Args:
            query: User's query string.
            intent: Target intent label.
            entities_by_type: Map of extracted entity types to values.

        Returns:
            (missing_parameter_names, missing_parameter_descriptions)
        """
        missing_params: List[str] = []
        missing_descs: Dict[str, str] = {}

        if not intent:
            return missing_params, missing_descs

        if intent == "CREATE_DIRECTORY":
            if "DIRECTORY_NAME" not in entities_by_type:
                missing_params.append("directory_name")
                missing_descs["directory_name"] = "Please specify the name or path of the directory to create."

        elif intent == "CREATE_FILE":
            if "FILE_NAME" not in entities_by_type:
                missing_params.append("file_name")
                missing_descs["file_name"] = "Please specify the name of the file to create."

        elif intent == "DELETE_FILE":
            if "FILE_NAME" not in entities_by_type and "SOURCE_PATH" not in entities_by_type:
                missing_params.append("file_name")
                missing_descs["file_name"] = "Please specify the file to delete."

        elif intent in ("MOVE_FILE", "COPY_FILE"):
            if "FILE_NAME" not in entities_by_type and "SOURCE_PATH" not in entities_by_type:
                missing_params.append("source")
                missing_descs["source"] = "Please specify the source file to move/copy."
            if "DESTINATION_PATH" not in entities_by_type:
                missing_params.append("destination_path")
                missing_descs["destination_path"] = "Please specify the target destination directory."

        elif intent == "CHECK_VERSION":
            if "VERSION_NAME" not in entities_by_type:
                missing_params.append("version_name")
                missing_descs["version_name"] = "Please specify which software or runtime version to check (e.g. Python, Git, Node.js)."

        return missing_params, missing_descs
