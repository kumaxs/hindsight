"""The provider driven through the Hermes MemoryProvider interface, asserting what it
sends to Hindsight (a recording fake client stands in for the real SDK)."""

import inspect
import json
import sys
from types import SimpleNamespace

import hindsight_hermes as plugin
import pytest
from conftest import FakeClient
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
    assert result["result"] == "1. fact one\n2. fact two"
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
    assert result["result"] == "1. kept"
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
    assert "a2a_tag" in keys


# ---------------------------------------------------------------------------
# turn_author — bot-authored (agent-to-agent) turns
# ---------------------------------------------------------------------------


BOT_AUTHOR = {"id": "bot:fron", "name": "@fron", "is_bot": True}
HUMAN_AUTHOR = {"id": "u1", "name": "Wang", "is_bot": False}
BOT_DM = "Message from 🤖 fron (@fron): t2 done"


class TestTurnAuthor:
    """A bot's DM arrives as role=user but was written by a peer agent. ``turn_author``
    carries ``is_bot``: the turn is retained in full and tagged (``a2a_tag``) so recall
    can leave agent-to-agent traffic out. Nothing is dropped, no text is sniffed."""

    def test_bot_authored_turn_carries_a2a_tag(self, provider):
        instance, fake = provider()
        instance.sync_turn(BOT_DM, "ack", turn_author=BOT_AUTHOR)
        instance.shutdown()

        item = _retain_item(fake)
        assert item["tags"] == ["session:session-1", "source:bot"]
        # Tagged, not dropped: the peer's turn is still in the retained payload.
        assert BOT_DM in item["content"]

    def test_human_turn_carries_no_a2a_tag(self, provider):
        instance, fake = provider()
        instance.sync_turn("hello", "hi", turn_author=HUMAN_AUTHOR)
        instance.shutdown()

        assert _retain_item(fake)["tags"] == ["session:session-1"]

    def test_turn_without_author_carries_no_a2a_tag(self, provider):
        """Older callers omit turn_author entirely — unchanged behavior."""
        instance, fake = provider()
        instance.sync_turn("hello", "hi")
        instance.shutdown()

        assert _retain_item(fake)["tags"] == ["session:session-1"]

    def test_custom_a2a_tag_respected(self, provider):
        instance, fake = provider({"a2a_tag": "x:y"})
        instance.sync_turn(BOT_DM, "ack", turn_author=BOT_AUTHOR)
        instance.shutdown()

        tags = _retain_item(fake)["tags"]
        assert "x:y" in tags
        assert "source:bot" not in tags

    @pytest.mark.parametrize("blank", ["", "   ", None])
    def test_blank_a2a_tag_disables_tagging(self, provider, blank):
        """An empty string, whitespace or a null YAML value disables tagging; none of
        them may ship as a literal "None" tag."""
        instance, fake = provider({"a2a_tag": blank})
        instance.sync_turn(BOT_DM, "ack", turn_author=BOT_AUTHOR)
        instance.shutdown()

        assert _retain_item(fake)["tags"] == ["session:session-1"]

    def test_a2a_tag_defaults_and_only_fires_for_bots(self, provider):
        instance, _ = provider()
        assert instance._a2a_tag == "source:bot"
        assert instance._a2a_tag_for_turn(BOT_AUTHOR) == "source:bot"
        assert instance._a2a_tag_for_turn(HUMAN_AUTHOR) == ""
        assert instance._a2a_tag_for_turn({"id": "u1", "is_bot": False}) == ""
        assert instance._a2a_tag_for_turn(None) == ""
        instance.shutdown()

    def test_bot_turn_is_buffered_like_any_other(self, provider):
        """No special short-circuit: bot turns ship like any other — counted,
        session-stamped, and tagged en route (append trim then clears the buffer)."""
        instance, fake = provider()
        instance.sync_turn(BOT_DM, "ack", session_id="s-new", turn_author=BOT_AUTHOR)

        assert instance._session_id == "s-new"
        assert instance._turn_counter == 1
        instance.shutdown()

        # The append retain shipped the turn with its tag; nothing was dropped.
        assert _retain_item(fake, -1)["tags"] == ["session:s-new", "source:bot"]
        assert BOT_DM in _retain_item(fake, -1)["content"]

    def test_mixed_batch_splits_by_source_tag(self, provider):
        """retain_every_n_turns > 1 (or a legacy overwrite server) can pack both kinds
        into one batch: each job then carries only its own tag, so a peer's turn is never
        mislabelled as the human's and a human turn never ships as agent-to-agent."""
        instance, fake = provider({"retain_every_n_turns": 2})
        instance.sync_turn(BOT_DM, "ack", turn_author=BOT_AUTHOR)  # turn 1: buffered
        instance.sync_turn("hello", "hi", turn_author=HUMAN_AUTHOR)  # turn 2: flushes both
        instance.shutdown()

        assert len(fake.retains) == 2
        bot_bodies = [
            json.dumps(_retain_item(fake, i)["content"], ensure_ascii=False)
            for i in range(2)
            if "source:bot" in _retain_item(fake, i)["tags"]
        ]
        human_bodies = [
            json.dumps(_retain_item(fake, i)["content"], ensure_ascii=False)
            for i in range(2)
            if "source:bot" not in _retain_item(fake, i)["tags"]
        ]
        assert len(bot_bodies) == 1 and len(human_bodies) == 1
        assert BOT_DM in bot_bodies[0] and "hello" not in bot_bodies[0]
        assert "hello" in human_bodies[0] and BOT_DM not in human_bodies[0]

    def test_append_retain_keeps_turn_tags_aligned_after_trim(self, provider, monkeypatch):
        """Append-mode retains drop shipped turns from ``_session_turns`` (#62950); the
        parallel tag list must go with them. Otherwise the next batch pairs a turn with
        an earlier turn's tag — a peer bot's turn ships untagged (into the user's recall)
        while a human turn ships as agent-to-agent traffic."""
        instance, fake = provider({"retain_every_n_turns": 2})  # both turns ship in the same flush
        instance.sync_turn("hello", "hi", turn_author=HUMAN_AUTHOR)  # buffered
        instance.sync_turn(BOT_DM, "ack", turn_author=BOT_AUTHOR)  # flushes both, split by tag
        instance.shutdown()

        assert len(fake.retains) == 2  # one job per source tag
        human_items = [r for r in fake.retains if "source:bot" not in r["items"][0]["tags"]]
        bot_items = [r for r in fake.retains if "source:bot" in r["items"][0]["tags"]]
        assert len(human_items) == 1 and len(bot_items) == 1
        assert human_items[0]["items"][0]["tags"] == ["session:session-1"]
        assert "hello" in human_items[0]["items"][0]["content"]
        assert BOT_DM not in human_items[0]["items"][0]["content"]
        assert bot_items[0]["items"][0]["tags"] == ["session:session-1", "source:bot"]
        assert BOT_DM in bot_items[0]["items"][0]["content"]
        assert "hello" not in bot_items[0]["items"][0]["content"]

    def test_append_retain_human_turn_after_bot_is_not_labelled_a2a(self, provider):
        """Mirror of the trim-alignment case on the live append path: after a bot turn
        is shipped and trimmed, the next human turn must not inherit its tag."""
        instance, fake = provider()
        instance.sync_turn(BOT_DM, "ack", turn_author=BOT_AUTHOR)  # retain + trim
        # The parallel tag list was trimmed in lockstep with _session_turns (append).
        assert instance._session_turns == []
        assert instance._session_turn_tags == []
        instance.sync_turn("hello", "hi", turn_author=HUMAN_AUTHOR)  # next turn stays human
        instance.shutdown()

        human_items = [r for r in fake.retains if "source:bot" not in r["items"][0]["tags"]]
        bot_items = [r for r in fake.retains if "source:bot" in r["items"][0]["tags"]]
        assert len(human_items) == 1 and len(bot_items) == 1
        assert human_items[0]["items"][0]["tags"] == ["session:session-1"]
        assert "hello" in human_items[0]["items"][0]["content"]
        assert BOT_DM not in human_items[0]["items"][0]["content"]
        assert bot_items[0]["items"][0]["tags"] == ["session:session-1", "source:bot"]
        assert BOT_DM in bot_items[0]["items"][0]["content"]
        assert "hello" not in bot_items[0]["items"][0]["content"]

    def test_signature_accepts_turn_author_for_core_dispatch(self, provider):
        """MemoryManager only forwards turn_author to sync_turn signatures that
        name it, so dropping the kwarg silently re-opens this bug — pin it."""
        instance, _ = provider()
        assert "turn_author" in inspect.signature(instance.sync_turn).parameters
        instance.shutdown()

    def test_a2a_tag_is_exposed_as_a_setting(self, provider):
        """Operators must be able to set it without editing code."""
        instance, _ = provider()
        keys = {option["key"] for option in instance.get_config_schema()}
        assert "a2a_tag" in keys
        instance.shutdown()
