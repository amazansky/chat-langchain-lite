"""Scope guarantees for the LangChain-only knowledge-base tools."""

from agent.tools import OUT_OF_SCOPE, get_security_advice, get_setup_guide


def test_setup_guide_off_domain_deployment_is_out_of_scope():
    assert get_setup_guide.invoke({"topic": "deployment"}) == OUT_OF_SCOPE
    assert get_setup_guide.invoke({"topic": "kubernetes deployment"}) == OUT_OF_SCOPE
    assert get_setup_guide.invoke({"topic": "vercel deployment"}) == OUT_OF_SCOPE


def test_setup_guide_returns_guide_when_topic_names_the_product():
    result = get_setup_guide.invoke({"topic": "langgraph deployment"})
    assert "LangGraph Platform" in result
    assert "Installation guide" in get_setup_guide.invoke({"topic": "langchain installation"})


def test_setup_guide_does_not_substring_match_unrelated_topics():
    assert get_setup_guide.invoke({"topic": "django install"}) == OUT_OF_SCOPE
    assert "not found" in get_setup_guide.invoke({"topic": "langchain deploy"})


def test_security_advice_off_domain_query_is_out_of_scope():
    result = get_security_advice.invoke({"query": "how do I secure my S3 buckets?"})
    assert result == OUT_OF_SCOPE
    assert "S3" not in result


def test_security_advice_answers_in_scope_query():
    result = get_security_advice.invoke({"query": "langgraph checkpointer security"})
    assert "RECOMMENDED patterns" in result
