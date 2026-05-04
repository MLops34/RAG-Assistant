"""
RAG interface for \"Chat with my syllabus\": chunk, embed, retrieve, answer.

Uses FAISS + OpenAI embeddings/chat by default. Set ``OPENAI_API_KEY``.
"""

from __future__ import annotations

import os
import sys
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from core.llm_provider import get_llm_provider_config

PathLike = Union[str, Path]

DEFAULT_CHUNK_SIZE = 1200
DEFAULT_CHUNK_OVERLAP = 200


def _ensure_llm_provider() -> None:
    get_llm_provider_config()


def _split_documents(documents: List[Document]) -> List[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=DEFAULT_CHUNK_SIZE,
        chunk_overlap=DEFAULT_CHUNK_OVERLAP,
        length_function=len,
    )
    return splitter.split_documents(documents)


def documents_from_pdf(path: PathLike, extra_metadata: Optional[Dict[str, Any]] = None) -> List[Document]:
    """Load a PDF as LangChain ``Document`` pages (then chunked by caller)."""
    from langchain_community.document_loaders import PyPDFLoader

    path = Path(path)
    loader = PyPDFLoader(str(path))
    docs = loader.load()
    meta = {"source": str(path.resolve()), **(extra_metadata or {})}
    for d in docs:
        d.metadata = {**d.metadata, **meta}
    return docs


def documents_from_text(
    text: str,
    *,
    source_label: str = "inline",
    extra_metadata: Optional[Dict[str, Any]] = None,
) -> List[Document]:
    """Wrap raw syllabus text as a single document for chunking."""
    meta = {"source": source_label, **(extra_metadata or {})}
    return [Document(page_content=text, metadata=meta)]


class SyllabusRAG:
    """
    In-memory FAISS store per instance. Typical flow:

    .. code-block:: python

        rag = SyllabusRAG()
        rag.ingest_pdf(\"data/course.pdf\")
        answer = rag.query(\"When is the midterm?\")
    """

    def __init__(
        self,
        *,
        embedding_model: Optional[str] = None,
        chat_model: Optional[str] = None,
        k_retrieve: int = 6,
    ) -> None:
        _ensure_llm_provider()
        provider = get_llm_provider_config()
        self._embedding_model = embedding_model or os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
        self._chat_model = chat_model or os.getenv("OPENAI_CHAT_MODEL", "mistralai/mistral-7b-instruct")
        self._k = k_retrieve
        self._embeddings = OpenAIEmbeddings(
            model=self._embedding_model,
            api_key=provider.api_key,
            base_url=provider.base_url,
        )
        self._vectorstore: Optional[FAISS] = None
        self._store_id: str = str(uuid.uuid4())
        self._provider = provider

    @property
    def store_id(self) -> str:
        return self._store_id

    def ingest_documents(self, documents: List[Document]) -> int:
        """Chunk, embed, and merge into the vector store. Returns number of chunks added."""
        chunks = _split_documents(documents)
        if not chunks:
            return 0
        if self._vectorstore is None:
            self._vectorstore = FAISS.from_documents(chunks, self._embeddings)
        else:
            self._vectorstore.add_documents(chunks)
        return len(chunks)

    def ingest_pdf(self, path: PathLike, **metadata: Any) -> int:
        """Load PDF from disk and ingest. Extra kwargs become document metadata."""
        docs = documents_from_pdf(path, extra_metadata=metadata or None)
        return self.ingest_documents(docs)

    def ingest_text(self, text: str, **metadata: Any) -> int:
        """Ingest raw syllabus text."""
        docs = documents_from_text(text, extra_metadata=metadata or None)
        return self.ingest_documents(docs)

    def as_retriever(self):
        if self._vectorstore is None:
            raise RuntimeError("No documents ingested. Call ingest_pdf or ingest_text first.")
        return self._vectorstore.as_retriever(search_kwargs={"k": self._k})

    def query(self, question: str) -> str:
        """Retrieve relevant chunks and return a grounded answer."""
        if self._vectorstore is None:
            raise RuntimeError("No documents ingested. Call ingest_pdf or ingest_text first.")

        retriever = self.as_retriever()
        llm = ChatOpenAI(
            model=self._chat_model,
            temperature=0,
            api_key=self._provider.api_key,
            base_url=self._provider.base_url,
        )

        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "You are a teaching assistant. Answer ONLY using the provided syllabus context. "
                    "If the answer is not in the context, say you do not find it in the syllabus and "
                    "suggest what section the student might check. Be concise.",
                ),
                ("human", "Context:\n{context}\n\nQuestion: {input}"),
            ]
        )

        docs = retriever.invoke(question)
        context = "\n\n".join(d.page_content for d in docs)
        msg = prompt.format_messages(context=context, input=question)
        response = llm.invoke(msg)
        return str(getattr(response, "content", response)).strip()

    def save_local(self, directory: PathLike) -> None:
        """Persist FAISS index + metadata to disk."""
        if self._vectorstore is None:
            raise RuntimeError("Nothing to save.")
        path = Path(directory)
        path.mkdir(parents=True, exist_ok=True)
        self._vectorstore.save_local(str(path))

    def load_local(self, directory: PathLike) -> None:
        """Load FAISS index from disk (same embedding model must be used)."""
        path = Path(directory)
        self._vectorstore = FAISS.load_local(
            str(path),
            self._embeddings,
            allow_dangerous_deserialization=True,
        )
