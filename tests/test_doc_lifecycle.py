import pytest
from unittest.mock import MagicMock, patch
from backend.storage.doc_lifecycle import mark_document_deleted, get_chunk_ids_by_filename


class TestMarkDocumentDeleted:

    @patch("backend.storage.doc_lifecycle.SessionLocal")
    def test_marks_chunks_as_deleted(self, mock_session_factory):
        mock_session = MagicMock()
        mock_session_factory.return_value.__enter__.return_value = mock_session

        mock_session.execute.return_value.rowcount = 2

        result = mark_document_deleted("test.pdf")

        assert result["filename"] == "test.pdf"
        assert result["affected_chunks"] == 2
        assert result["status"] == "soft_deleted"
        mock_session.commit.assert_called()

    @patch("backend.storage.doc_lifecycle.SessionLocal")
    def test_returns_empty_when_no_chunks(self, mock_session_factory):
        mock_session = MagicMock()
        mock_session_factory.return_value.__enter__.return_value = mock_session
        mock_session.execute.return_value.rowcount = 0

        result = mark_document_deleted("nonexist.pdf")

        assert result["affected_chunks"] == 0


class TestGetChunkIdsByFilename:

    @patch("backend.storage.doc_lifecycle.SessionLocal")
    def test_returns_chunk_ids(self, mock_session_factory):
        mock_session = MagicMock()
        mock_session_factory.return_value.__enter__.return_value = mock_session
        mock_session.execute.return_value.scalars.return_value.all.return_value = [
            ("ck1",), ("ck2",), ("ck3",)
        ]

        result = get_chunk_ids_by_filename("test.pdf")

        assert result == ["ck1", "ck2", "ck3"]
