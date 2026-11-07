"""Offline tests for RAG orchestration; no HTTP, E5 download or Chroma needed."""

from unittest.mock import Mock

from django.test import SimpleTestCase
from langchain_core.documents import Document as LangChainDocument
from langchain_core.messages import AIMessage

from documents.services.rag import NO_DOCUMENTS_ANSWER, answer_question


class RAGServiceTests(SimpleTestCase):
    def test_retrieves_context_and_returns_sources(self):
        chunk = LangChainDocument(
            page_content="Machine learning detects patterns in data.",
            metadata={"document_id": 2, "chunk_id": "2:0", "title": "AI notes"},
        )
        search = Mock(return_value=[chunk])
        llm = Mock()
        llm.invoke.return_value = AIMessage(content="Machine learning detects patterns [1].")

        result = answer_question(
            "What is machine learning?", search_fn=search, chat_model=llm, k=3
        )

        search.assert_called_once_with(
            "What is machine learning?", k=3, document_id=None
        )
        self.assertIn("patterns", result["answer"])
        self.assertEqual(result["sources"][0]["chunk_id"], "2:0")
        messages = llm.invoke.call_args.args[0]
        self.assertIn("[1] Document: AI notes", messages[1].content)
        self.assertIn("Machine learning detects patterns", messages[1].content)

    def test_empty_index_skips_llm(self):
        llm = Mock()
        result = answer_question(
            "Any question?", search_fn=Mock(return_value=[]), chat_model=llm
        )
        self.assertEqual(result["answer"], NO_DOCUMENTS_ANSWER)
        self.assertEqual(result["sources"], [])
        llm.invoke.assert_not_called()

    def test_invalid_inputs_fail_before_retrieval(self):
        search = Mock()
        for question in ["", "   ", None]:
            with self.subTest(question=question):
                with self.assertRaises(ValueError):
                    answer_question(question, search_fn=search)
        with self.assertRaises(ValueError):
            answer_question("hello", k=0, search_fn=search)
        with self.assertRaises(ValueError):
            answer_question("hello", k=9, search_fn=search)
        with self.assertRaises(ValueError):
            answer_question("hello", document_id=-3, search_fn=search)
        search.assert_not_called()

    def test_empty_model_response_is_error(self):
        llm = Mock()
        llm.invoke.return_value = AIMessage(content="")
        chunk = LangChainDocument(page_content="Text", metadata={})
        with self.assertRaisesRegex(RuntimeError, "no textual answer"):
            answer_question(
                "Question", search_fn=Mock(return_value=[chunk]), chat_model=llm
            )

    def test_document_id_filter_is_passed_to_retriever(self):
        search = Mock(return_value=[])
        answer_question("Question", document_id=2, search_fn=search)
        search.assert_called_once_with("Question", k=4, document_id=2)
