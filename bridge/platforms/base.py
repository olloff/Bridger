"""
Abstract base class that every platform must implement.
"""

from abc import ABC, abstractmethod
from bridge.utils.config import bridge_sources, bridge_targets, bridge_all_platforms

class BasePlatform(ABC):
    """Interface contract for a bridge platform."""

    name: str
    def __init__(self, name: str, platform_config: dict, bridges: list[dict], router=None):
        """
        Parameters
        ----------
        name : str
            Platform key as it appears in ``settings.yaml`` (e.g. ``"telegram"``).
        platform_config : dict
            The platform-specific credentials / options from ``settings["platforms"][name]``.
        bridges : list[dict]
            The full list of bridge definitions. Each platform should filter
            for bridges that include itself.
        router : Router | None
            The central message router.  Set after construction via
            :pyattr:`router` so that the router can be created after all
            platforms are instantiated.
        """
        self.name = name
        self.platform_config = platform_config
        self.bridges = bridges
        self._bridge_by_name: dict[str, dict] = {}
        self._bridge_for_chat: dict[object, str] = {}
        self._bridge_for_source_chat: dict[object, str] = {}
        self._targets_by_bridge: dict[str, list] = {}
        self._rebuild_bridge_indexes()
        self.router = router

    # Lifecycle
    @abstractmethod
    async def start(self) -> None:
        """Connect / authenticate and begin receiving events.

        This coroutine should run **indefinitely** (e.g. keep an event-loop
        alive) so that it can be used with ``asyncio.gather``.
        """

    @abstractmethod
    async def stop(self) -> None:
        """Gracefully disconnect."""

    # Outbound messaging
    @abstractmethod
    async def send_text(
        self,
        chat_id,
        content: str,
        sender: str,
        reply_to_native_id=None,
        *,
        trusted_internal: bool = False,
    ) -> int | None:
        """Send a text message to *chat_id*.
        Returns the **native message ID** of the sent message, or ``None``
        if sending failed. ``trusted_internal`` is an explicit rendering
        boundary: it may only be set for bridge-generated, destination-safe
        markup. It must never be inferred from user-controlled strings.
        """

    @abstractmethod
    async def send_file(
        self,
        chat_id,
        file_path: str,
        file_ext: str,
        sender: str,
        reply_to_native_id=None,
        *,
        trusted_internal: bool = False,
    ) -> int | None:
        """Send a file/attachment to *chat_id*.
        Returns the **native message ID** of the sent message, or ``None``
        if sending failed.
        """

    # Outbound text safety
    def escape_user_text(self, s: str) -> str:
        """Escape *s* for safe interpolation into this platform's outbound
        message format.

        Default: identity (the platform's send_text doesn't render markup,
        or its rendering is benign). Telegram overrides this with HTML
        escaping because its outbound parse mode is HTML and untrusted
        text could otherwise inject tags or markdown link syntax.
        """
        return s

    def format_digest_body(self, entries: list[tuple[float, str, str]]) -> str:
        """Render digest entries as a body string in this platform's native
        markup. Default is platform-agnostic plain text. Subclasses override
        to produce richer formatting safely (with user content escaped).

        Consecutive messages from the same sender are grouped under a
        single ``Name · HH:MM`` header — keeps the digest readable when
        someone sends a burst of messages at once."""
        import time
        # group: list of [sender, first_ts, [content_lines]]
        groups: list[list] = []
        for ts, sender, content in entries:
            if not content or not content.strip(): continue
            lines = content.splitlines() or [content]
            if groups and groups[-1][0] == sender:
                groups[-1][2].extend(lines)
            else:
                groups.append([sender, ts, list(lines)])
        out: list[str] = []
        for sender, ts, lines in groups:
            ts_str = time.strftime("%H:%M", time.localtime(ts))
            out.append(f"{sender} · {ts_str}")
            for line in lines:
                out.append(f"  {line}")
            out.append("")
        return "\n".join(out).rstrip()

    # Hot-reload hook
    def apply_config(self, platform_config: dict) -> None:
        """Refresh mutable platform settings from a re-read YAML file.

        Default implementation only updates the stored ``platform_config``.
        Subclasses should override to reapply settings such as filters.
        Credentials cannot be reloaded — those are baked into the client.
        """
        self.platform_config = platform_config

    # Helpers
    def set_bridges(self, bridges: list[dict]) -> None:
        """Atomically replace bridge definitions and rebuild lookup indexes."""
        self.bridges = bridges
        self._rebuild_bridge_indexes()

    def _rebuild_bridge_indexes(self) -> None:
        self._bridge_by_name = {}
        self._bridge_for_chat = {}
        self._bridge_for_source_chat = {}
        self._targets_by_bridge = {}
        for bridge in self.bridges:
            name = bridge.get("name")
            if not name:
                continue
            self._bridge_by_name[name] = bridge
            for chat_id in bridge_all_platforms(bridge).get(self.name, []):
                self._bridge_for_chat.setdefault(chat_id, name)
            for chat_id in bridge_sources(bridge).get(self.name, []):
                self._bridge_for_source_chat.setdefault(chat_id, name)
            self._targets_by_bridge[name] = list(bridge_targets(bridge).get(self.name, []))

    def bridge_name_for_chat(self, chat_id) -> str | None:
        """Return the bridge name that involves *chat_id* on this platform —
        regardless of direction. Used by outbound formatting paths that just
        need to know which bridge a chat belongs to."""
        return self._bridge_for_chat.get(chat_id)

    def bridge_name_for_source_chat(self, chat_id) -> str | None:
        """Return the bridge name where *chat_id* is a **source** for this
        platform. Used by inbound dispatch — messages on a target-only chat
        have no bridge to forward into."""
        return self._bridge_for_source_chat.get(chat_id)

    def chat_ids_for_bridge(self, bridge_name: str) -> list:
        """Return this platform's *target* chat IDs for a given bridge."""
        return list(self._targets_by_bridge.get(bridge_name, []))

    def my_chat_ids(self) -> list:
        """Return all chat IDs this platform should listen to (its source chats)."""
        return list(self._bridge_for_source_chat.keys())

    def find_bridge_by_name(self, name: str) -> dict | None:
        """Return the bridge dict by name, or ``None`` if not found."""
        return self._bridge_by_name.get(name)
