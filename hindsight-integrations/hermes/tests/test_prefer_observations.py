"""``prefer_observations`` forwarding (source: hermes-agent #64914, prefer_observations
half only — the min_scores half is officially covered by vectorize-io/hindsight#5087 and
is deliberately NOT re-added here).

The plugin reads the ``prefer_observations`` config knob and sends it to Hindsight
only when the user opts in, on both recall paths (the ``hindsight_recall`` tool and
the auto-recall prefetch); unset, the kwarg stays off the wire entirely so a default
config's recall request is byte-for-byte what it was before the flag existed."""

import json

import hindsight_hermes as plugin
from conftest import FakeClient


def test_prefer_observations_is_exposed_as_a_setting(provider):
    """Operators must be able to set it without editing code — and it defaults off."""
    instance, _ = provider()
    options = {option["key"]: option for option in instance.get_config_schema()}
    assert options["prefer_observations"]["default"] is False
    instance.shutdown()


def test_tool_recall_forwards_prefer_observations_when_opted_in(provider):
    instance, fake = provider(
        {"prefer_observations": True}, client=FakeClient(recall_texts=["fact one"])
    )
    instance.handle_tool_call("hindsight_recall", {"query": "who am I?"})
    assert fake.recalls[0]["prefer_observations"] is True
    instance.shutdown()


def test_prefetch_forwards_prefer_observations_when_opted_in(provider):
    instance, fake = provider(
        {"recall_sync": True, "prefer_observations": True},
        client=FakeClient(recall_texts=["fact one"]),
    )
    instance.prefetch("what do you know?")
    assert fake.recalls[0]["prefer_observations"] is True
    instance.shutdown()


def test_unset_prefer_observations_keeps_the_kwarg_off_the_wire(provider):
    """Default config: no opt-in — neither recall path sends the kwarg."""
    instance, fake = provider({}, client=FakeClient(recall_texts=["fact one"]))
    instance.handle_tool_call("hindsight_recall", {"query": "who am I?"})
    instance.prefetch("what do you know?")
    assert fake.recalls
    assert all("prefer_observations" not in call for call in fake.recalls)
    instance.shutdown()


def test_prefer_observations_explicit_false_keeps_the_kwarg_off_the_wire(provider):
    instance, fake = provider({"prefer_observations": False}, client=FakeClient(recall_texts=["x"]))
    instance.handle_tool_call("hindsight_recall", {"query": "q"})
    assert "prefer_observations" not in fake.recalls[0]
    instance.shutdown()


def test_prefer_observations_falsy_values_stay_disabled(provider):
    """The bool() read normalizes junk config values onto the disabled side instead
    of letting a truthy-looking string ride to the client (``""`` / ``null`` / ``0``)."""
    for bad in ("", None, 0):
        instance, fake = provider({"prefer_observations": bad}, client=FakeClient(recall_texts=["x"]))
        instance.handle_tool_call("hindsight_recall", {"query": "q"})
        assert "prefer_observations" not in fake.recalls[0], repr(bad)
        instance.shutdown()


def test_results_are_unaffected_without_the_opt_in(provider):
    instance, fake = provider({}, client=FakeClient(recall_texts=["fact one", "fact two"]))
    result = json.loads(instance.handle_tool_call("hindsight_recall", {"query": "who am I?"}))
    assert result["result"] == "1. fact one\n2. fact two"
    instance.shutdown()


def test_the_flag_message_never_leaks_into_the_result(provider):
    instance, fake = provider(
        {"prefer_observations": True}, client=FakeClient(recall_texts=["kept fact"])
    )
    result = json.loads(instance.handle_tool_call("hindsight_recall", {"query": "q"}))
    assert result["result"] == "1. kept fact"
    assert fake.recalls[0]["prefer_observations"] is True
    instance.shutdown()
