import pytest
from langchain_core.messages import AIMessage, ToolMessage

from docqa_lab.assistant import answer_question, estimate_cost, format_context


class FakeToolCallingModel:
    def __init__(self, responses):
        self.responses = iter(responses)
        self.seen_messages = []
        self.bound_tools = []

    def bind_tools(self, tools):
        self.bound_tools = [tool.name for tool in tools]
        return self

    def invoke(self, messages):
        self.seen_messages.append(list(messages))
        return next(self.responses)


def test_model_response_citations_and_grounding(kb):
    model = FakeToolCallingModel([AIMessage(content="Run pytest. [S1]")])
    result = answer_question(model, kb, "How to run pytest tests?")
    assert "[S1]" in result.text
    assert "build.md" in result.sources
    assert "read_document" in model.bound_tools
    assert "[S1]" in model.seen_messages[0][1].content


def test_tool_call_is_executed_then_response(kb):
    model = FakeToolCallingModel([
        AIMessage(content="", tool_calls=[{"name": "list_documents", "args": {}, "id": "c1"}]),
        AIMessage(content="The available files are build.md and deploy.md. [S1]"),
    ])
    result = answer_question(model, kb, "What documentation covers pytest?")
    assert result.tool_events == ["called: list_documents"]
    assert isinstance(model.seen_messages[1][-1], ToolMessage)
    assert "build.md" in model.seen_messages[1][-1].content


def test_unknown_tool_is_not_executed(kb):
    model = FakeToolCallingModel([
        AIMessage(
            content="",
            tool_calls=[{"name": "run_shell", "args": {"cmd": "pwd"}, "id": "c1"}],
        ),
        AIMessage(content="No shell operations are available."),
    ])
    result = answer_question(model, kb, "How to run pytest?")
    assert result.tool_events == ["rejected: unknown tool"]


def test_out_of_scope_retrieval_has_no_context(kb):
    assert format_context(kb.search("xylophone astronomer nebula")) == (
        "No matching document excerpts were found. Abstain unless a tool provides evidence."
    )


def test_hallucinated_citation_rejected(kb):
    model = FakeToolCallingModel([AIMessage(content="Incorrect claim [S99]")])
    with pytest.raises(RuntimeError, match="citations"):
        answer_question(model, kb, "How to run pytest?")


def test_model_failure_has_safe_error(kb):
    class Failing:
        def bind_tools(self, tools):
            return self

        def invoke(self, messages):
            raise ConnectionError("a secret payload")

    with pytest.raises(RuntimeError, match="Chat model failed") as exc:
        answer_question(Failing(), kb, "How to run tests?")
    assert "secret payload" not in str(exc.value)


def test_tool_call_budget(kb):
    def request_tool(call_id):
        return AIMessage(
            content="", tool_calls=[{"name": "list_documents", "args": {}, "id": call_id}]
        )

    model = FakeToolCallingModel([request_tool("1"), request_tool("2"), request_tool("3")])
    result = answer_question(model, kb, "How to run pytest?")
    assert "budget" in result.text
    assert result.tool_events.count("called: list_documents") == 2


def test_token_cost_estimation():
    assert estimate_cost({"input_tokens": 1_000, "output_tokens": 500}, (1.0, 4.0)) == 0.003
    assert estimate_cost({"total_tokens": 100}, (1.0, 4.0)) is None
