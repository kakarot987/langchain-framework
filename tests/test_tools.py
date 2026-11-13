import pytest

from docqa_lab.tools import make_tools


def test_tools_are_read_only_and_bounded(kb):
    listing, read = make_tools(kb)
    assert listing.invoke({}) == "build.md\ndeploy.md"
    assert "Run pytest" in read.invoke({"name": "build.md"})
    with pytest.raises(ValueError):
        read.invoke({"name": "../../private.txt"})


def test_tool_inputs_reject_oversized_names(kb):
    read = make_tools(kb)[1]
    with pytest.raises(ValueError):
        read.invoke({"name": "z" * 181})
