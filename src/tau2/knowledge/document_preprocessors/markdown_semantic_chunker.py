import re
from typing import Any, Dict, List, Tuple

from tau2.knowledge.document_preprocessors.base import BaseDocumentPreprocessor
from tau2.knowledge.document_preprocessors.search_text import normalize_document_id
from tau2.knowledge.registry import register_document_preprocessor

_HEADING = re.compile(r"^(#{1,6})\s+(.+?)\s*$")


@register_document_preprocessor("markdown_semantic_chunker")
class MarkdownSemanticChunker(BaseDocumentPreprocessor):
    """Split only long Markdown documents at structural boundaries."""

    def __init__(self, max_chars: int = 2000, **kwargs: Any):
        super().__init__(max_chars=max_chars, **kwargs)
        if max_chars < 1:
            raise ValueError("max_chars must be positive")
        self.max_chars = max_chars

    def process(
        self, documents: List[Dict[str, Any]], state: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        chunks = []
        parent_map = {}
        state["parent_doc_content_map"] = {}
        state["parent_doc_title_map"] = {}
        state["chunk_rerank_text_map"] = {}

        for document in documents:
            parent_id = document["id"]
            title = str(document.get("title", parent_id))
            content = str(document.get("text") or document.get("content") or "")
            state["parent_doc_content_map"][parent_id] = content
            state["parent_doc_title_map"][parent_id] = title

            grouped = (
                [("Document", content)]
                if len(content) <= self.max_chars
                else self.split_content(content)
            )
            for index, (heading_path, chunk_content) in enumerate(grouped, start=1):
                chunk_id = f"{parent_id}::chunk_{index:03d}"
                parent_map[chunk_id] = parent_id
                normalized_id = normalize_document_id(parent_id)
                chunks.append(
                    {
                        "id": chunk_id,
                        "normalized_id": normalized_id,
                        "title": title,
                        "section": heading_path,
                        "text": chunk_content,
                    }
                )
                state["chunk_rerank_text_map"][chunk_id] = (
                    f"ID: {normalized_id}\n"
                    f"Title: {title}\n"
                    f"Section: {heading_path}\n"
                    f"Content:\n{chunk_content}"
                )

        state["chunk_parent_map"] = parent_map
        return chunks

    def split_content(self, content: str) -> List[Tuple[str, str]]:
        """Return Markdown-aware sections grouped within the configured size."""
        sections = self._parse_sections(content)
        groups: List[Tuple[str, str]] = []
        current_paths: List[str] = []
        current_parts: List[str] = []
        current_length = 0

        for path, section in sections:
            for part in self._split_oversized_section(section):
                addition = len(part) + (2 if current_parts else 0)
                if current_parts and current_length + addition > self.max_chars:
                    groups.append(
                        (" | ".join(current_paths), "\n\n".join(current_parts))
                    )
                    current_paths, current_parts, current_length = [], [], 0
                current_paths.append(path)
                current_parts.append(part)
                current_length += len(part) + (2 if len(current_parts) > 1 else 0)

        if current_parts:
            groups.append((" | ".join(current_paths), "\n\n".join(current_parts)))
        return groups

    @staticmethod
    def _parse_sections(content: str) -> List[Tuple[str, str]]:
        sections = []
        headings: List[Tuple[int, str]] = []
        buffer: List[str] = []
        current_path = "Document"
        for line in content.splitlines():
            match = _HEADING.match(line)
            if match:
                if buffer:
                    sections.append((current_path, "\n".join(buffer).strip()))
                level = len(match.group(1))
                while headings and headings[-1][0] >= level:
                    headings.pop()
                headings.append((level, match.group(2)))
                current_path = " > ".join(title for _, title in headings)
                buffer = [line]
            else:
                buffer.append(line)
        if buffer:
            sections.append((current_path, "\n".join(buffer).strip()))
        return [(path, text) for path, text in sections if text]

    def _split_oversized_section(self, section: str) -> List[str]:
        if len(section) <= self.max_chars:
            return [section]
        paragraphs = re.split(r"\n\s*\n", section)
        parts, current = [], ""
        for paragraph in paragraphs:
            if current and len(current) + len(paragraph) + 2 > self.max_chars:
                parts.append(current)
                current = ""
            if len(paragraph) > self.max_chars:
                if current:
                    parts.append(current)
                    current = ""
                parts.extend(
                    paragraph[start : start + self.max_chars]
                    for start in range(0, len(paragraph), self.max_chars)
                )
            else:
                current = f"{current}\n\n{paragraph}" if current else paragraph
        if current:
            parts.append(current)
        return parts
