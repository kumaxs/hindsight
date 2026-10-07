"""The provider driven through the Hermes MemoryProvider interface, asserting what it
sends to Hindsight (a recording fake client stands in for the real SDK)."""

import asyncio
import json
import sys
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

import hindsight_hermes as plugin
from conftest import FakeClient, FakeResult
from hindsight_client_api.exceptions import NotFoundException


def _retain_item(fake: FakeClient, index: int = 0) -> dict:
    return fake.retains[index]["items"][0]


def _turns_of(fake: FakeClient, index: int = 0) -> list[list[str]]:
    """Message texts per turn in one retain. Content is ``"[" + ",".join(turns) + "]"``
    where each turn is itself a JSON array, so the whole payload is a list of turns."""
    return [[m["content"] for m in turn] for turn in json.loads(_retain_item(fake, index)["content"])]


def test_sync_turn_retains_the_turn(provider):
    instance, fake = provider({"bank_id": "team", "retain_tags": "hermes"})
    instance.sync_turn("what is my name?", "Ada.")
    instance.shutdown()

    assert len(fake.retains) == 1
    call = fake.retains[0]
    assert call["bank_id"] == "team"
    assert call["document_id"] == "session-1"  # stable id + append on a capable API
    item = _retain_item(fake)
    assert item["update_mode"] == "append"
    assert "hermes" in item["tags"] and "session:session-1" in item["tags"]
    messages = json.loads(item["content"][1:-1])
    assert [m["content"] for m in messages] == ["User: what is my name?", "Assistant: Ada."]


def test_retain_every_n_turns_buffers_then_ships_the_batch(provider):
    instance, fake = provider({"retain_every_n_turns": 2})
    instance.sync_turn("one", "1")
    assert fake.retains == []
    instance.sync_turn("two", "2")
    instance.shutdown()

    assert len(fake.retains) == 1
    assert _retain_item(fake)["metadata"]["message_count"] == "4"


def test_auto_retain_off_stores_nothing(provider):
    instance, fake = provider({"auto_retain": False})
    instance.sync_turn("hello", "hi")
    instance.shutdown()
    assert fake.retains == []


def test_recall_tool_queries_the_bank_and_formats_results(provider):
    instance, fake = provider(
        {"bank_id": "team", "recall_budget": "high"}, client=FakeClient(recall_texts=["fact one", "fact two"])
    )
    result = json.loads(instance.handle_tool_call("hindsight_recall", {"query": "who am I?"}))

    assert fake.recalls[0]["bank_id"] == "team"
    assert fake.recalls[0]["budget"] == "high"
    assert fake.recalls[0]["types"] == ["observation"]  # observation-only default
    assert result["result"] == "1. id=? fact one\n2. id=? fact two"
    instance.shutdown()


def test_recall_sends_no_score_floor_by_default(provider):
    instance, fake = provider({}, client=FakeClient(recall_texts=["fact one"]))
    instance.handle_tool_call("hindsight_recall", {"query": "who am I?"})
    assert "min_scores" not in fake.recalls[0]
    instance.shutdown()


def test_recall_min_scores_reaches_the_tool_and_the_prefetch(provider):
    instance, fake = provider(
        {"recall_sync": True, "recall_min_scores": {"reranker": 0.25}}, client=FakeClient(recall_texts=["fact one"])
    )
    instance.handle_tool_call("hindsight_recall", {"query": "who am I?"})
    instance.prefetch("what do you know?")
    assert [call["min_scores"] for call in fake.recalls] == [{"reranker": 0.25}] * 2
    instance.shutdown()


def test_recall_min_scores_accepts_the_json_string_the_setup_wizard_writes(provider):
    instance, fake = provider({"recall_min_scores": '{"reranker": 0.25}'}, client=FakeClient(recall_texts=["x"]))
    instance.handle_tool_call("hindsight_recall", {"query": "q"})
    assert fake.recalls[0]["min_scores"] == {"reranker": 0.25}
    instance.shutdown()


def test_recall_with_an_empty_answer_stays_an_empty_block(provider):
    # The server drops what falls under a reranker/final floor; an empty answer must stay empty.
    instance, fake = provider({"recall_sync": True, "recall_min_scores": {"reranker": 0.9}}, client=FakeClient())
    assert instance.prefetch("something off topic") == ""
    assert fake.recalls[0]["min_scores"] == {"reranker": 0.9}
    assert json.loads(instance.handle_tool_call("hindsight_recall", {"query": "q"}))["result"] == (
        "No relevant memories found."
    )
    instance.shutdown()


def test_recall_min_scores_is_enforced_on_each_result_the_server_returns(provider):
    # The server prunes only the retrieval arm a `semantic` floor names, so weak results (and ones
    # another arm found, with no semantic score) can still come back; the plugin drops them so the
    # floor really abstains.
    results = [("kept", {"semantic": 0.7}), ("weak", {"semantic": 0.3}), ("other arm", {"semantic": None})]
    instance, fake = provider({"recall_min_scores": {"semantic": 0.5}}, client=FakeClient(recall_texts=results))
    result = json.loads(instance.handle_tool_call("hindsight_recall", {"query": "q"}))
    assert result["result"] == "1. id=? kept"
    assert fake.recalls[0]["min_scores"] == {"semantic": 0.5}
    instance.shutdown()


def test_recall_min_scores_drops_the_whole_block_when_nothing_clears_it(provider):
    results = [("weak", {"semantic": 0.3}), ("other arm", {"semantic": None})]
    instance, _ = provider(
        {"recall_sync": True, "recall_min_scores": {"semantic": 0.5}}, client=FakeClient(recall_texts=results)
    )
    assert instance.prefetch("something off topic") == ""
    assert json.loads(instance.handle_tool_call("hindsight_recall", {"query": "q"}))["result"] == (
        "No relevant memories found."
    )
    instance.shutdown()


def test_reflect_tool_uses_reflect(provider):
    instance, fake = provider({}, client=FakeClient(reflect_text="You are Ada."))
    result = json.loads(instance.handle_tool_call("hindsight_reflect", {"query": "who am I?"}))
    assert fake.reflects[0]["query"] == "who am I?"
    assert result["result"] == "You are Ada."
    instance.shutdown()


def test_retain_tool_stores_content_with_per_call_tags(provider):
    instance, fake = provider({"retain_tags": "base"})
    instance.handle_tool_call("hindsight_retain", {"content": "Ada likes tea", "tags": ["drink"]})
    item = _retain_item(fake)
    assert item["content"] == "Ada likes tea"
    assert item["tags"] == ["base", "drink"]
    instance.shutdown()


def test_builtin_memory_adds_and_replaces_are_retained(provider):
    instance, fake = provider({"bank_id": "team", "retain_tags": "base"})
    instance.on_memory_write("add", "memory", "Deploys go through Fly.io")
    instance.on_memory_write("replace", "user", "Ada prefers tea", metadata={"previous_content": "Ada likes tea"})
    instance.on_memory_write("remove", "memory", "", metadata={"previous_content": "Deploys go through Fly.io"})
    instance.shutdown()

    assert [call["bank_id"] for call in fake.retains] == ["team", "team"]
    first, second = _retain_item(fake, 0), _retain_item(fake, 1)
    assert first["content"] == "Deploys go through Fly.io"
    assert first["tags"] == ["base", "builtin-memory", "builtin-target:memory", "builtin-action:add"]
    assert second["content"] == "Ada prefers tea"
    assert second["tags"] == ["base", "builtin-memory", "builtin-target:user", "builtin-action:replace"]


def test_tool_call_errors_are_reported_not_raised(provider):
    instance, _ = provider({})
    assert instance.handle_tool_call("hindsight_recall", {}).startswith("ERROR:")
    assert instance.handle_tool_call("nope", {"query": "x"}).startswith("ERROR:")
    instance.shutdown()


class _EmptyServerClient(FakeClient):
    async def arecall(self, **kwargs):
        raise NotFoundException(status=404, reason="Not Found")

    async def areflect(self, **kwargs):
        raise NotFoundException(status=404, reason="Not Found")


def test_searching_a_bank_nothing_was_saved_to_finds_nothing(provider):
    instance, _ = provider({}, client=_EmptyServerClient())
    for tool in ("hindsight_recall", "hindsight_reflect"):
        result = json.loads(instance.handle_tool_call(tool, {"query": "who am I?"}))
        assert result == {"result": "No relevant memories found."}
    instance.shutdown()


def test_prefetch_injects_recalled_memories(provider):
    instance, fake = provider({"recall_sync": True}, client=FakeClient(recall_texts=["fact one"]))
    block = instance.prefetch("what do you know?")
    assert "- fact one" in block
    status = instance.recall_status()
    assert status.count == 1 and status.provider_label == "Hindsight"
    instance.shutdown()


def test_context_mode_hides_tools_tools_mode_skips_recall(provider):
    context_only, _ = provider({"memory_mode": "context"})
    assert context_only.get_tool_schemas() == []
    context_only.shutdown()

    tools_only, fake = provider({"memory_mode": "tools", "recall_sync": True})
    assert [t["name"] for t in tools_only.get_tool_schemas()] == [
        "hindsight_retain",
        "hindsight_recall",
        "hindsight_reflect",
        "hindsight_invalidate",
    ]
    assert tools_only.prefetch("anything") == ""
    assert fake.recalls == []
    tools_only.shutdown()


def test_session_switch_starts_a_new_document(provider):
    instance, fake = provider({})
    instance.sync_turn("one", "1")
    instance.on_session_switch("session-2", reset=True)
    instance.sync_turn("two", "2")
    instance.shutdown()

    # The switch flushes the old session's buffer under the old document id first,
    # so the new session's turn can never land in the previous document. In append
    # mode the buffer is already empty here (sync_turn shipped and dropped the turn),
    # so there is nothing left to flush — previously this re-shipped the retained
    # turn under session-1 a second time, duplicating it in the document.
    assert [call["document_id"] for call in fake.retains] == ["session-1", "session-2"]


def test_register_exposes_the_provider_to_hermes():
    registered = []
    plugin.register(type("Ctx", (), {"register_memory_provider": lambda _self, p: registered.append(p)})())
    assert registered and registered[0].name == "hindsight"


def test_append_mode_drops_retained_turns_from_the_buffer(provider):
    """Append retains ship a delta, so keeping every turn would pin the whole session
    in memory on a long-running gateway (hermes-agent #62950).

    Append mode comes from the API capability probe, which the fixture pins on — it is
    not a config key.
    """
    instance, fake = provider({})
    instance.sync_turn("one", "1")
    instance.sync_turn("two", "2")

    # Buffer state is read before shutdown(); retains only land once the writer drains.
    assert instance._session_turns == []
    assert instance._last_retained_turn_count == 0
    instance.shutdown()

    # Each retain still carries only its own un-retained tail, never a replay.
    assert _turns_of(fake, 0) == [["User: one", "Assistant: 1"]]
    assert _turns_of(fake, 1) == [["User: two", "Assistant: 2"]]


def test_overwrite_mode_keeps_every_turn(provider, monkeypatch):
    """Overwrite resends the full session on each retain, so its buffer must NOT be
    cleared — only the append path drops shipped turns."""
    instance, fake = provider({})
    # An API without update_mode='append' support: the fixture pins the probe on, so
    # turn it back off to exercise the overwrite path.
    monkeypatch.setattr(plugin, "_check_api_supports_update_mode_append", lambda *a, **k: False)
    instance.sync_turn("one", "1")
    instance.sync_turn("two", "2")

    assert len(instance._session_turns) == 2  # one buffered entry per turn
    instance.shutdown()

    # The second retain resends the whole session, which is what overwrite means.
    assert _turns_of(fake, 1) == [["User: one", "Assistant: 1"], ["User: two", "Assistant: 2"]]


def test_root_warning_goes_through_the_hosts_warning_callback(provider, monkeypatch):
    """The 'cannot run as root' notice is an automatic startup diagnostic: hosts that
    wire a gated sink must receive it there, not on stderr (hermes-agent cd3de040ab9)."""
    seen = []
    instance, _ = provider({}, warning_callback=seen.append, platform="telegram")
    assert instance._platform == "telegram"

    monkeypatch.setattr(plugin.os, "geteuid", lambda: 0, raising=False)
    instance._mode = "local_embedded"
    instance._start_embedded_daemon()

    assert len(seen) == 1 and "cannot run as root" in seen[0]
    assert instance._mode == "disabled"
    instance.shutdown()


def test_warning_sink_defaults_exist_without_initialize():
    """_start_embedded_daemon reads these directly, and availability probes construct a
    provider without ever calling initialize() — so __init__ must supply both."""
    bare = plugin.HindsightMemoryProvider()
    assert bare._warning_callback is None
    assert bare._platform == "cli"


def test_system_prompt_guides_tool_choice_only_when_tools_exist(provider):
    blocks = {}
    for mode in ("context", "tools", "hybrid"):
        instance, _ = provider({"memory_mode": mode})
        blocks[mode] = instance.system_prompt_block()
        instance.shutdown()

    assert "session_search" not in blocks["context"]
    assert "automatically injected" in blocks["context"]
    for mode in ("tools", "hybrid"):
        assert "prefer hindsight_recall over session_search" in blocks[mode]
        assert "hindsight_reflect" in blocks[mode] and "hindsight_retain" in blocks[mode]
    assert "automatically injected" in blocks["hybrid"]
    assert "automatically injected" not in blocks["tools"]


def test_the_first_run_download_is_announced_through_the_warning_sink(provider, monkeypatch):
    """A first embedded start fetches the server through uvx, which took minutes with nothing on
    screen (hermes-agent#4936: a 6m23s reply that retained nothing and printed no error). The
    notice goes to the same gated sink as the root-refusal warning."""
    seen = []
    instance, _ = provider({}, warning_callback=seen.append, platform="telegram")
    monkeypatch.setattr(plugin, "_daemon_is_running", lambda profile: False)
    monkeypatch.setattr(plugin, "_installed_api_binary_exists", lambda: False)

    instance._announce_slow_first_start("hermes")

    assert len(seen) == 1 and "downloading its local memory server" in seen[0]
    instance.shutdown()


def test_no_announcement_when_the_server_is_already_there(provider, monkeypatch):
    seen = []
    instance, _ = provider({}, warning_callback=seen.append)
    monkeypatch.setattr(plugin, "_daemon_is_running", lambda profile: False)
    monkeypatch.setattr(plugin, "_installed_api_binary_exists", lambda: True)
    instance._announce_slow_first_start("hermes")

    monkeypatch.setattr(plugin, "_daemon_is_running", lambda profile: True)
    monkeypatch.setattr(plugin, "_installed_api_binary_exists", lambda: False)
    instance._announce_slow_first_start("hermes")

    assert seen == []
    instance.shutdown()


def test_concurrent_callers_start_the_daemon_and_build_the_client_once(provider, monkeypatch):
    """The start worker and the first memory operation both reach _get_client. Unguarded, each
    started a daemon and built a client, and the loser's client was dropped without being closed.

    No barrier inside the build: with the lock in place only one caller ever gets there, so the
    contention window is opened with a sleep instead.
    """
    import threading
    import time as _time

    instance, _ = provider({})
    instance._mode = "local_embedded"
    built = []

    def _slow_build(self):
        _time.sleep(0.2)  # as wide as a real daemon start, in miniature
        built.append(object())
        return built[-1]

    monkeypatch.setattr(type(instance), "_new_embedded_client", _slow_build)

    results = []
    threads = [threading.Thread(target=lambda: results.append(instance._get_client())) for _ in range(3)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(10)

    assert len(built) == 1, f"client built {len(built)} times"
    assert len({id(r) for r in results}) == 1  # every caller got the same client
    instance.shutdown()


def test_building_the_embedded_client_announces_before_it_waits(provider, monkeypatch):
    """The notice has to fire from the path that actually blocks — asserting the helper in
    isolation would keep passing if nothing called it."""
    seen = []
    # Mode is switched after init: configuring local_embedded would also start the background
    # daemon worker, which builds a client too and made this test announce twice at random.
    instance, _ = provider({"profile": "hermes"}, warning_callback=seen.append)
    instance._mode = "local_embedded"
    order = []
    from hindsight_hermes.embedded import LocalRuntimeStatus

    monkeypatch.setattr(plugin, "_check_local_runtime", lambda: LocalRuntimeStatus(available=True))
    monkeypatch.setattr(plugin, "_daemon_is_running", lambda profile: False)
    monkeypatch.setattr(plugin, "_installed_api_binary_exists", lambda: False)
    monkeypatch.setattr(plugin, "_build_embedded_profile_env", lambda cfg: {})
    monkeypatch.setattr(
        plugin, "_start_daemon", lambda config, profile: order.append("started") or "http://127.0.0.1:1"
    )
    monkeypatch.setattr(plugin, "Hindsight", lambda **kw: object(), raising=False)
    instance._warning_callback = lambda m: order.append("announced")

    instance._new_embedded_client()

    assert order == ["announced", "started"], order
    instance.shutdown()


def test_the_embedded_client_sends_the_daemons_tenant_key(provider, monkeypatch):
    """#5023: a daemon running ApiKeyTenantExtension answers 401 to a keyless client."""
    instance, _ = provider({"profile": "hermes"})
    instance._mode = "local_embedded"
    from hindsight_hermes.embedded import LocalRuntimeStatus, _embedded_profile_env_path

    profile_env = _embedded_profile_env_path({"profile": "hermes"})
    profile_env.parent.mkdir(parents=True, exist_ok=True)
    profile_env.write_text("HINDSIGHT_API_TENANT_API_KEY=tenant-secret\n")
    built = {}

    class _Client:
        def __init__(self, **kwargs):
            built.update(kwargs)

    monkeypatch.setattr(plugin, "_check_local_runtime", lambda: LocalRuntimeStatus(available=True))
    monkeypatch.setattr(plugin, "_start_daemon", lambda config, profile: "http://127.0.0.1:1")
    monkeypatch.setattr(plugin, "_installed_api_binary_exists", lambda: True)
    monkeypatch.setitem(sys.modules, "hindsight_client", SimpleNamespace(Hindsight=_Client))

    instance._new_embedded_client()

    assert built == {"base_url": "http://127.0.0.1:1", "api_key": "tenant-secret"}
    instance.shutdown()


def test_retain_omits_strategy_by_default(provider):
    """No strategy configured means no key on the item, so the bank keeps deciding."""
    instance, fake = provider()
    instance.sync_turn("hello", "hi")
    instance.shutdown()

    assert fake.retains
    for call in fake.retains:
        for item in call["items"]:
            assert "strategy" not in item


def test_retain_sends_the_configured_strategy(provider):
    """A configured strategy rides on every stored item."""
    instance, fake = provider({"retain_strategy": "agent-session"})
    instance.sync_turn("hello", "hi")
    instance.shutdown()

    assert fake.retains
    for call in fake.retains:
        for item in call["items"]:
            assert item["strategy"] == "agent-session"


def test_retain_strategy_is_exposed_as_a_setting(provider):
    """Operators must be able to set it without editing code."""
    instance, _ = provider()
    keys = {option["key"] for option in instance.get_config_schema()}
    assert "retain_strategy" in keys


# ---------------------------------------------------------------------------
# invalidate/restore memory curation (ported from hermes-agent#68206)
# ---------------------------------------------------------------------------


def _recalled(result: FakeResult, memory_id: str | None):
    """A FakeResult carrying the SDK's ``id`` attribute when the server supplies it."""
    result.id = memory_id
    return result


def _recall_with_ids(fake: FakeClient, *pairs: tuple[str, str | None]) -> None:
    """Give the fake's pending recall response results with SDK-like ``id`` attributes.

    ``fake._recall_texts`` only carries texts (or ``(text, scores)`` tuples) into
    FakeRecallResponse, which re-wraps each entry in a fresh FakeResult — so an id
    riding along on the original object is lost. Mock the arecall call directly with
    plain SimpleNamespace results the way the SDK would return them instead."""
    instance = SimpleNamespace(results=[_recalled(FakeResult(text), memory_id) for text, memory_id in pairs])
    fake.arecall = _mk_arecall(instance)


def _mk_arecall(resp):
    async def arecall(**kwargs):
        return resp

    return arecall


def test_get_tool_schemas_returns_four(provider):
    instance, _ = provider({})
    schemas = instance.get_tool_schemas()
    assert len(schemas) == 4
    names = {s["name"] for s in schemas}
    assert names == {"hindsight_retain", "hindsight_recall", "hindsight_reflect", "hindsight_invalidate"}
    instance.shutdown()


def test_recall_ids_are_full_never_truncated(provider):
    """Full IDs rule: recall prints the whole id, no 12-character prefix cut
    (the #68207 truncation was reviewed down)."""
    instance, fake = provider({})
    _recall_with_ids(
        fake,
        ("Memory 1", "5e79c849-f3b6-4a1e-b789-123456789abc"),
        ("Memory 2", "baee4d5b-84bd-4c3e-9f12-abcdef123456"),
    )
    result = json.loads(instance.handle_tool_call("hindsight_recall", {"query": "test"}))
    assert "id=5e79c849-f3b6-4a1e-b789-123456789abc" in result["result"]
    assert "id=baee4d5b-84bd-4c3e-9f12-abcdef123456" in result["result"]
    instance.shutdown()


def test_recall_missing_id_shows_question_mark(provider):
    """A result without an id attribute degrades to ``id=?`` gracefully."""
    instance, fake = provider({})
    _recall_with_ids(fake, ("No ID memory", None))
    result = json.loads(instance.handle_tool_call("hindsight_recall", {"query": "test"}))
    assert "id=?" in result["result"]
    assert "No ID memory" in result["result"]
    instance.shutdown()


def test_recall_types_override_hits_the_client_kwargs(provider):
    """Agent-provided types param overrides the default recall_types."""
    instance, fake = provider({})
    instance.handle_tool_call("hindsight_recall", {"query": "test", "types": ["world", "experience"]})
    assert fake.recalls[0]["types"] == ["world", "experience"]
    instance.shutdown()


def test_recall_without_types_uses_default(provider):
    """When types is omitted, the configured default observation filter applies."""
    instance, fake = provider({})
    instance.handle_tool_call("hindsight_recall", {"query": "test"})
    assert fake.recalls[0]["types"] == ["observation"]
    instance.shutdown()


def test_recall_types_override_applies_to_one_call_only(provider):
    instance, fake = provider({})
    instance.handle_tool_call("hindsight_recall", {"query": "q", "types": ["world"]})
    instance.handle_tool_call("hindsight_recall", {"query": "q"})
    assert [c["types"] for c in fake.recalls] == [["world"], ["observation"]]
    instance.shutdown()


def test_recall_types_override_reaches_the_prefetch_path_too(provider):
    """The per-call override is a tool argument; the prefetch keeps the config default."""
    instance, fake = provider({"recall_sync": True})
    instance.prefetch("anything")
    assert fake.recalls[0]["types"] == ["observation"]
    instance.shutdown()


class _RecordingUpdateMemory:
    """Stand-in for the SDK path that records what the plugin asked the API to do."""

    def __init__(self) -> None:
        self.calls: list[dict] = []

    def __call__(self, **kwargs):
        request = kwargs.get("update_memory_request")
        self.calls.append(
            kwargs | {"state": getattr(request, "state", None), "reason": getattr(request, "reason", None)}
        )
        return SimpleNamespace()


def _wire_sdk_update(instance, monkeypatch) -> _RecordingUpdateMemory:
    recording = _RecordingUpdateMemory()

    class _Req:
        def __init__(self, state=None, **kwargs):
            self.state = state
            self.reason = None

    monkeypatch.setattr(instance, "_try_import_update_memory_request", lambda: _Req)

    def _fake_op(operation):
        client = SimpleNamespace(memory=SimpleNamespace(update_memory=recording))
        # The real operation awaits client.memory.update_memory(...); run that call
        # the way _run_sync schedules it, without a real event loop dependency.
        coroutine = operation(client)
        if asyncio.iscoroutine(coroutine):
            return asyncio.run(coroutine)
        return coroutine  # a recording stand-in that returned synchronously

    monkeypatch.setattr(instance, "_run_hindsight_operation", _fake_op)
    return recording


def test_invalidate_via_the_sdk_path(provider, monkeypatch):
    """UpdateMemoryRequest(state='invalidated') rides client.memory.update_memory."""
    instance, _ = provider({})
    recording = _wire_sdk_update(instance, monkeypatch)

    result = json.loads(
        instance.handle_tool_call("hindsight_invalidate", {"memory_id": "abc-123-def", "reason": "stale info"})
    )
    assert recording.calls == [
        {
            "bank_id": instance._bank_id,
            "memory_id": "abc-123-def",
            "update_memory_request": recording.calls[0]["update_memory_request"],
            "state": "invalidated",
            "reason": "stale info",
        }
    ]
    assert "invalidated" in result["result"]
    assert "abc-123-def" in result["result"]
    instance.shutdown()


def test_invalidate_sdk_path_carries_the_reason_for_the_audit_trail(provider, monkeypatch):
    """reason rides on the UpdateMemoryRequest, not just the log line."""
    instance, _ = provider({})
    recording = _wire_sdk_update(instance, monkeypatch)

    instance.handle_tool_call("hindsight_invalidate", {"memory_id": "abc-123", "reason": "superseded by doc X"})
    request = recording.calls[0]["update_memory_request"]
    assert request.state == "invalidated"
    assert request.reason == "superseded by doc X"
    instance.shutdown()


def test_invalidate_restore_sets_state_valid(provider, monkeypatch):
    """restore=true → state=valid; restore never needs a reason."""
    instance, _ = provider({})
    recording = _wire_sdk_update(instance, monkeypatch)

    result = json.loads(instance.handle_tool_call("hindsight_invalidate", {"memory_id": "abc-123", "restore": True}))
    assert recording.calls[0]["update_memory_request"].state == "valid"
    assert "restored" in result["result"]
    instance.shutdown()


def test_invalidate_string_restore_false_means_invalidate(provider, monkeypatch):
    """A string 'false' must never act as a truthy flag — it means invalidate."""
    instance, _ = provider({})
    recording = _wire_sdk_update(instance, monkeypatch)

    result = json.loads(
        instance.handle_tool_call(
            "hindsight_invalidate", {"memory_id": "abc-123", "restore": "false", "reason": "stale"}
        )
    )
    assert recording.calls[0]["update_memory_request"].state == "invalidated"
    assert "invalidated" in result["result"]
    instance.shutdown()


def test_invalidate_requires_reason(provider, monkeypatch):
    """Invalidating without a reason returns an error (audit-trail rule)."""
    instance, _ = provider({})
    monkeypatch.setattr(instance, "_try_import_update_memory_request", lambda: object)
    result = json.loads(instance.handle_tool_call("hindsight_invalidate", {"memory_id": "abc-123-def"}))
    assert "error" in result
    assert "reason is required" in result["error"]
    instance.shutdown()


def test_invalidate_reason_whitespace_only_fails_too(provider, monkeypatch):
    instance, _ = provider({})
    monkeypatch.setattr(instance, "_try_import_update_memory_request", lambda: object)
    result = json.loads(instance.handle_tool_call("hindsight_invalidate", {"memory_id": "abc", "reason": "   "}))
    assert "error" in result
    assert "reason is required" in result["error"]
    instance.shutdown()


def test_invalidate_missing_params(provider):
    """Neither query nor memory_id returns an error payload."""
    instance, _ = provider({})
    result = json.loads(instance.handle_tool_call("hindsight_invalidate", {}))
    assert "error" in result
    instance.shutdown()


def test_invalidate_query_mode_lists_ids_and_texts(provider, monkeypatch):
    instance, _ = provider({})
    monkeypatch.setattr(
        instance,
        "_http_list_invalidated",
        MagicMock(
            return_value=[
                {"id": "abc-123", "text": "old server address"},
                {"id": "def-456", "text": "deprecated config"},
            ]
        ),
    )
    result = json.loads(instance.handle_tool_call("hindsight_invalidate", {"query": "server"}))
    assert "abc-123" in result["result"]
    assert "old server address" in result["result"]
    assert result["ids"] == ["abc-123", "def-456"]
    instance.shutdown()


def test_invalidate_discovery_sends_full_ids(provider, monkeypatch):
    """Discovery surfaces the full ids array — the model reads them straight back."""
    instance, _ = provider({})
    ids = ["5e79c849-f3b6-4a1e-b789-123456789abc", "baee4d5b-84bd-4c3e-9f12-abcdef123456"]
    monkeypatch.setattr(
        instance,
        "_http_list_invalidated",
        MagicMock(return_value=[{"id": ids[0], "text": "Memory 1"}, {"id": ids[1], "text": "Memory 2"}]),
    )
    result = json.loads(instance.handle_tool_call("hindsight_invalidate", {"query": "memory"}))
    assert result["ids"] == ids
    instance.shutdown()


def test_invalidate_query_no_results(provider, monkeypatch):
    instance, _ = provider({})
    monkeypatch.setattr(instance, "_http_list_invalidated", MagicMock(return_value=[]))
    result = json.loads(instance.handle_tool_call("hindsight_invalidate", {"query": "nonexistent"}))
    assert "No invalidated memories" in result["result"]
    instance.shutdown()


def test_invalidate_query_and_id_mutually_exclusive(provider):
    instance, _ = provider({})
    result = json.loads(instance.handle_tool_call("hindsight_invalidate", {"query": "test", "memory_id": "abc-123"}))
    assert "error" in result
    instance.shutdown()


def test_invalidate_http_fallback_when_the_sdk_lacks_update_memory(provider, monkeypatch):
    instance, _ = provider({})
    monkeypatch.setattr(instance, "_try_import_update_memory_request", lambda: None)
    patcher = MagicMock()
    monkeypatch.setattr(instance, "_http_patch_memory", patcher)
    result = json.loads(
        instance.handle_tool_call("hindsight_invalidate", {"memory_id": "abc-123-def", "reason": "stale info"})
    )
    patcher.assert_called_once_with("abc-123-def", "invalidated", reason="stale info")
    assert "invalidated" in result["result"]
    instance.shutdown()


def test_invalidate_api_errors_are_surfaced(provider, monkeypatch):
    instance, _ = provider({})
    monkeypatch.setattr(instance, "_try_import_update_memory_request", lambda: None)
    monkeypatch.setattr(
        instance,
        "_http_patch_memory",
        MagicMock(side_effect=RuntimeError("HTTP 422: observation type cannot be invalidated")),
    )
    result = json.loads(instance.handle_tool_call("hindsight_invalidate", {"memory_id": "obs-123", "reason": "stale"}))
    assert "error" in result
    assert "422" in result["error"]
    instance.shutdown()


def test_invalidate_observation_refusal_redirects_to_source_facts(provider, monkeypatch):
    """Observations regenerate from their source memories: name them instead."""
    instance, _ = provider({})
    monkeypatch.setattr(instance, "_try_import_update_memory_request", lambda: None)
    monkeypatch.setattr(
        instance,
        "_http_patch_memory",
        MagicMock(
            side_effect=RuntimeError(
                "HTTP 422: only world/experience facts can be curated; "
                "observations are derived and regenerate from their sources"
            )
        ),
    )
    monkeypatch.setattr(
        instance,
        "_http_get_memory",
        MagicMock(
            return_value={
                "source_memories": [
                    {
                        "id": "d658bf5a-1111",
                        "text": "User meets the vendor for contract review next Wednesday",
                        "type": "world",
                    },
                    {"id": "d658bf5a-2222", "text": "another source fact", "type": "world"},
                ]
            }
        ),
    )
    result = json.loads(instance.handle_tool_call("hindsight_invalidate", {"memory_id": "obs-123", "reason": "stale"}))
    assert "error" in result
    assert "Retire one of its source facts instead" in result["error"]
    assert "d658bf5a-1111" in result["error"]
    instance.shutdown()


def test_invalidate_observation_hint_names_every_source(provider, monkeypatch):
    instance, _ = provider({})
    monkeypatch.setattr(instance, "_try_import_update_memory_request", lambda: None)
    monkeypatch.setattr(
        instance,
        "_http_patch_memory",
        MagicMock(side_effect=RuntimeError("HTTP 422: observations are derived (world/experience only)")),
    )
    monkeypatch.setattr(
        instance,
        "_http_get_memory",
        MagicMock(
            return_value={
                "source_memories": [
                    {"id": "src-1", "text": "source one"},
                    {"id": "src-2", "text": "source two"},
                ]
            }
        ),
    )
    result = json.loads(instance.handle_tool_call("hindsight_invalidate", {"memory_id": "obs-1", "reason": "stale"}))
    assert "src-1" in result["error"] and "src-2" in result["error"]
    instance.shutdown()


def test_invalidate_unrelated_error_gets_no_source_hint(provider, monkeypatch):
    """An error about something else must not trigger the source-facts lookup."""
    instance, _ = provider({})
    monkeypatch.setattr(instance, "_try_import_update_memory_request", lambda: None)
    monkeypatch.setattr(instance, "_http_patch_memory", MagicMock(side_effect=RuntimeError("HTTP 500: internal")))
    get_memory = MagicMock()
    monkeypatch.setattr(instance, "_http_get_memory", get_memory)
    result = json.loads(instance.handle_tool_call("hindsight_invalidate", {"memory_id": "abc", "reason": "x"}))
    assert "error" in result
    get_memory.assert_not_called()
    instance.shutdown()


def test_invalidate_observation_without_sources_hint_stays_empty(provider, monkeypatch):
    instance, _ = provider({})
    monkeypatch.setattr(instance, "_try_import_update_memory_request", lambda: None)
    monkeypatch.setattr(
        instance,
        "_http_patch_memory",
        MagicMock(
            side_effect=RuntimeError("HTTP 422: only world/experience facts can be curated; observations derive")
        ),
    )
    monkeypatch.setattr(instance, "_http_get_memory", MagicMock(return_value={"source_memories": []}))
    result = json.loads(instance.handle_tool_call("hindsight_invalidate", {"memory_id": "obs-1", "reason": "x"}))
    assert "error" in result
    assert "Retire one of its source facts" not in result["error"]
    instance.shutdown()


def test_invalidate_targets_the_primary_bank_only(provider, monkeypatch):
    """Curation targets the primary bank — extra banks are mirrors, not owners."""
    instance, _ = provider({"additional_banks": "mirror-a, mirror-b"})
    recording = _wire_sdk_update(instance, monkeypatch)

    instance.handle_tool_call("hindsight_invalidate", {"memory_id": "abc", "reason": "stale"})
    assert recording.calls[0]["bank_id"] == instance._bank_id
    assert len(recording.calls) == 1
    instance.shutdown()


def test_coerce_bool_accepts_the_usual_spellings():
    for value, expected in {
        "true": True,
        "TRUE": True,
        "1": True,
        "yes": True,
        "on": True,
        "false": False,
        "0": False,
        "no": False,
        "off": False,
        " False ": False,
        None: None,
        True: True,
        False: False,
        "garbage": None,
    }.items():
        assert plugin._coerce_bool(value) is expected


def test_system_prompt_names_the_invalidate_tool(provider):
    instance, _ = provider({})
    block = instance.system_prompt_block()
    assert "hindsight_invalidate" in block
    instance.shutdown()


class TestHttpHelpersWire:
    """Wire-level tests for _http_list_invalidated/_http_patch_memory/_http_get_memory:
    a local ThreadingHTTPServer records the requests the plugin actually sends."""

    @pytest.fixture()
    def _http_server(self):
        import threading
        from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

        captured = {"path": "", "method": "", "body": b"", "headers": {}}

        class _Handler(BaseHTTPRequestHandler):
            def do_GET(self):
                captured["path"] = self.path
                captured["method"] = "GET"
                captured["headers"] = dict(self.headers)
                resp = json.dumps(
                    {
                        "items": [
                            {"id": "abc-123", "text": "old server address"},
                        ],
                        "total": 1,
                        "limit": 50,
                        "offset": 0,
                    }
                ).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(resp)))
                self.end_headers()
                self.wfile.write(resp)

            def do_PATCH(self):
                captured["path"] = self.path
                captured["method"] = "PATCH"
                captured["headers"] = dict(self.headers)
                length = int(self.headers.get("Content-Length", 0))
                captured["body"] = self.rfile.read(length)
                self.send_response(200)
                self.end_headers()

            def log_message(self, *args):
                pass

        server = ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
        port = server.server_address[1]
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        yield f"http://127.0.0.1:{port}", captured
        server.shutdown()

    def test_list_invalidated_endpoint_with_query_state_and_limit(self, provider, _http_server):
        """_http_list_invalidated hits /memories/list with q=, state=invalidated, limit=50."""
        instance, _ = provider({})
        base_url, captured = _http_server
        instance._api_url = base_url
        instance._timeout = 5
        result = instance._http_list_invalidated("server")
        assert result == [{"id": "abc-123", "text": "old server address"}]
        assert "/memories/list" in captured["path"]
        assert "q=server" in captured["path"]
        assert "state=invalidated" in captured["path"]
        assert "limit=50" in captured["path"]
        assert captured["method"] == "GET"
        instance.shutdown()

    def test_wire_list_sends_the_bearer_token_only_when_a_key_exists(self, provider, _http_server):
        """Bearer is added only when _api_key holds a value."""
        instance, _ = provider({})
        base_url, captured = _http_server
        instance._api_url = base_url
        instance._timeout = 5
        instance._http_list_invalidated("q")
        assert captured["headers"].get("Authorization") == "Bearer test-key"
        instance.shutdown()

    def test_wire_list_without_a_key_ships_no_authorization_header(self, provider, _http_server):
        instance, _ = provider({})
        base_url, captured = _http_server
        instance._api_url = base_url
        instance._timeout = 5
        instance._api_key = ""
        instance._http_list_invalidated("q")
        assert "Authorization" not in captured["headers"]
        instance.shutdown()

    def test_patch_memory_encodes_the_id_and_carries_state_and_reason(self, provider, _http_server):
        instance, _ = provider({})
        base_url, captured = _http_server
        instance._api_url = base_url
        instance._timeout = 5
        instance._http_patch_memory("abc#frag?x=1", "invalidated", reason="stale info")
        assert captured["method"] == "PATCH"
        assert "abc%23frag%3Fx%3D1" in captured["path"]
        assert "#" not in captured["path"]
        body = json.loads(captured["body"].decode("utf-8"))
        assert body == {"state": "invalidated", "reason": "stale info"}
        instance.shutdown()

    def test_patch_memory_omits_reason_when_there_is_none(self, provider, _http_server):
        instance, _ = provider({})
        base_url, captured = _http_server
        instance._api_url = base_url
        instance._timeout = 5
        instance._http_patch_memory("abc", "valid", reason=None)
        body = json.loads(captured["body"].decode("utf-8"))
        assert body == {"state": "valid"}
        instance.shutdown()

    def test_get_memory_hits_the_singular_memory_path(self, provider, _http_server):
        """_http_get_memory must hit /memories/{id}, NOT /memories/list."""
        instance, _ = provider({})
        base_url, captured = _http_server
        instance._api_url = base_url
        instance._timeout = 5
        body = instance._http_get_memory("abc#frag")
        assert captured["method"] == "GET"
        assert "/memories/abc%23frag" in captured["path"]
        assert "/memories/list" not in captured["path"]
        assert isinstance(body, dict)
        instance.shutdown()

    def test_patch_memory_encodes_a_hostile_bank_id(self, provider, _http_server):
        """A bank_id with /?#= cannot change the PATCH path/query boundary."""
        instance, _ = provider({})
        base_url, captured = _http_server
        instance._api_url = base_url
        instance._timeout = 5
        instance._bank_id = "../../other-bank?x="
        instance._http_patch_memory("abc", "invalidated", reason="r")
        assert captured["path"] == "/v1/default/banks/..%2F..%2Fother-bank%3Fx%3D/memories/abc"
        instance.shutdown()

    def test_list_invalidated_encodes_a_hostile_bank_id(self, provider, _http_server):
        """A bank_id with ?# cannot break out of the /memories/list path segment."""
        instance, _ = provider({})
        base_url, captured = _http_server
        instance._api_url = base_url
        instance._timeout = 5
        instance._bank_id = "b?#"
        instance._http_list_invalidated("server")
        assert captured["path"] == "/v1/default/banks/b%3F%23/memories/list?q=server&state=invalidated&limit=50"
        instance.shutdown()

    def test_get_memory_encodes_a_hostile_bank_id(self, provider, _http_server):
        """A bank_id with / stays a single path segment on GET /memories/{id}."""
        instance, _ = provider({})
        base_url, captured = _http_server
        instance._api_url = base_url
        instance._timeout = 5
        instance._bank_id = "a/b"
        instance._http_get_memory("abc")
        assert captured["path"] == "/v1/default/banks/a%2Fb/memories/abc"
        instance.shutdown()

    def test_normal_bank_id_stays_unencoded(self, provider, _http_server):
        """No regression: a plain bank_id wires through verbatim on all three paths."""
        instance, _ = provider({})
        base_url, captured = _http_server
        instance._api_url = base_url
        instance._timeout = 5
        instance._bank_id = "hermes"
        instance._http_patch_memory("abc", "valid")
        assert "/v1/default/banks/hermes/memories/abc" in captured["path"]
        instance._http_get_memory("abc")
        assert "/v1/default/banks/hermes/memories/abc" in captured["path"]
        instance._http_list_invalidated("q")
        assert "/v1/default/banks/hermes/memories/list" in captured["path"]
        assert "%2F" not in captured["path"]
        instance.shutdown()

    def test_blank_bank_id_fails_closed_on_the_wire(self, provider, _http_server):
        """An empty bank_id raises before any request goes out (fails closed)."""
        instance, _ = provider({})
        base_url, captured = _http_server
        instance._api_url = base_url
        instance._timeout = 5
        instance._bank_id = ""
        with pytest.raises(ValueError, match="bank_id"):
            instance._http_patch_memory("abc", "valid")
        with pytest.raises(ValueError, match="bank_id"):
            instance._http_list_invalidated("q")
        with pytest.raises(ValueError, match="bank_id"):
            instance._http_get_memory("abc")
        assert captured["method"] == ""  # the dummy server saw no request
        instance.shutdown()

    def test_safe_url_path_segment_rejects_blank_and_nul(self):
        """Unit contract of _safe_url_path_segment: encode the injection face,
        refuse what encoding cannot express."""
        assert plugin._safe_url_path_segment("a/b?c#d\ne", field="bank_id") == "a%2Fb%3Fc%23d%0Ae"
        with pytest.raises(ValueError, match="bank_id"):
            plugin._safe_url_path_segment("", field="bank_id")
        with pytest.raises(ValueError, match="bank_id"):
            plugin._safe_url_path_segment("   ", field="bank_id")
        with pytest.raises(ValueError, match="NUL"):
            plugin._safe_url_path_segment("a\x00b", field="bank_id")

    def test_http_list_surfaces_http_error_bodies(self, provider, _http_server, monkeypatch):
        """HTTPError → RuntimeError with the code and a body excerpt."""
        import urllib.error

        instance, _ = provider({})
        base_url, _ = _http_server
        instance._api_url = base_url
        instance._timeout = 5

        def _raise(url, **kwargs):  # noqa: ARG001
            raise urllib.error.HTTPError(url, 403, "forbidden", hdrs=None, fp=None)

        monkeypatch.setattr("urllib.request.urlopen", _raise)
        with pytest.raises(RuntimeError, match="HTTP 403"):
            instance._http_list_invalidated("q")
        instance.shutdown()

    def test_probe_url_prefers_the_running_embedded_client(self, provider):
        """Embedded dynamic-port resolution: client URL → daemon URL → api_url (the
        _resolve_retain_target ordering); cloud mode ignores the client entirely."""
        instance, _ = provider({})
        client = SimpleNamespace(url="http://127.0.0.1:12345")
        instance._mode = "local_embedded"
        instance._client = client
        instance._embedded_url = "http://127.0.0.1:99999"
        instance._api_url = "https://api.hindsight.vectorize.io"
        assert instance._probe_url() == "http://127.0.0.1:12345"
        # No URL on the client → the daemon URL captured at start.
        client.url = ""
        assert instance._probe_url() == "http://127.0.0.1:99999"
        # Both empty → api_url.
        instance._embedded_url = ""
        assert instance._probe_url() == "https://api.hindsight.vectorize.io"
        # Cloud mode ignores the embedded client entirely.
        instance._mode = "cloud"
        assert instance._probe_url() == "https://api.hindsight.vectorize.io"
        instance.shutdown()


class TestInvalidateSchemaContract:
    """The schema itself: two-mode description, reason wording, restore flag, types enum."""

    def test_schema_names_and_shape(self, provider):
        instance, _ = provider({})
        schema = next(s for s in instance.get_tool_schemas() if s["name"] == "hindsight_invalidate")
        assert schema["parameters"]["type"] == "object"
        props = schema["parameters"]["properties"]
        assert set(props) == {"query", "memory_id", "reason", "restore"}
        assert props["restore"]["default"] is False
        instance.shutdown()

    def test_schema_carries_the_two_modes_and_reason_wording(self, provider):
        instance, _ = provider({})
        schema = next(s for s in instance.get_tool_schemas() if s["name"] == "hindsight_invalidate")
        description = schema["description"]
        assert "TWO MODES" in description
        assert "memory_id" in description and "query" in description
        assert "audit" in description
        reason_description = schema["parameters"]["properties"]["reason"]["description"]
        assert "REQUIRED" in reason_description
        assert "invalidation_reason" in reason_description
        assert "Not needed" in reason_description  # restoring needs no reason
        instance.shutdown()

    def test_recall_schema_advertises_the_types_override(self, provider):
        instance, _ = provider({})
        schema = next(s for s in instance.get_tool_schemas() if s["name"] == "hindsight_recall")
        types_param = schema["parameters"]["properties"]["types"]
        assert types_param["items"]["enum"] == ["world", "experience", "observation"]
        assert "Overrides the configured default" in types_param["description"]
        assert schema["parameters"]["required"] == ["query"]
        instance.shutdown()

    def test_recall_schema_warns_only_world_experience_can_be_invalidated(self, provider):
        instance, _ = provider({})
        schema = next(s for s in instance.get_tool_schemas() if s["name"] == "hindsight_recall")
        assert "Only world/experience facts can be invalidated" in schema["description"]
        instance.shutdown()


class TestSdkGuard:
    """_try_import_update_memory_request: the lazy-import compatibility guard."""

    def test_the_real_import_returns_the_class_on_the_current_floor(self, provider):
        instance, _ = provider({})
        from hindsight_client_api.models.update_memory_request import UpdateMemoryRequest as Real

        assert instance._try_import_update_memory_request() is Real
        instance.shutdown()

    def test_a_missing_module_returns_none_and_falls_back_to_http(self, provider, monkeypatch):
        instance, _ = provider({})
        import builtins

        real_import = builtins.__import__

        def _blocked(name, *args, **kwargs):
            if name.startswith("hindsight_client_api.models.update_memory_request"):
                raise ImportError(name)
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", _blocked)
        assert instance._try_import_update_memory_request() is None
        # And the tool call takes the HTTP path end to end:
        patcher = MagicMock()
        monkeypatch.setattr(instance, "_http_patch_memory", patcher)
        result = json.loads(instance.handle_tool_call("hindsight_invalidate", {"memory_id": "abc", "reason": "stale"}))
        assert patcher.called
        assert "invalidated" in result["result"]
        instance.shutdown()
