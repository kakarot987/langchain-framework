"""Two allowlisted, read-only tools with Pydantic argument validation."""

from langchain_core.tools import tool
from pydantic import BaseModel, Field

from .corpus import KnowledgeBase


class ReadDocumentInput(BaseModel):
    name: str = Field(min_length=1, max_length=180, description="Exact Markdown source name")


def make_tools(kb: KnowledgeBase):
    @tool
    def list_documents() -> str:
        """List names of the Markdown files available to answer questions."""
        return "\n".join(sorted(kb.files))

    @tool(args_schema=ReadDocumentInput)
    def read_document(name: str) -> str:
        """Read a bounded excerpt of a document by its exact listed name."""
        return f"Source: {name}\n{kb.read_excerpt(name)}"

    return [list_documents, read_document]
