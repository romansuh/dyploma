"""Regex-based normalization of log lines.

Replaces volatile technical entities (IPs, paths, hex values, exception class
names, session IDs, numbers) with stable placeholders. The placeholders are the
custom special tokens that we will later add to the BERT tokenizer in notebook
02, so the textual form here matches the tokenizer vocabulary.
"""

from __future__ import annotations

import re


class RegexNormalizer:
    """Replace technical entities in a log line with semantic placeholders.

    Patterns are applied in a fixed order (see `normalize`). The order matters:
    IPs and hex values are handled before bare integers, so the digits inside
    them are not separately replaced with `<NUM>`. Session IDs are detected
    after hex normalization, so the pattern matches `session <HEX>`.
    """

    EXC_PATTERN: re.Pattern[str] = re.compile(
        r"\b(?:[a-z][a-z0-9_]*\.)+[A-Z][A-Za-z0-9_]*(?:Exception|Error)\b"
    )
    IP_PATTERN: re.Pattern[str] = re.compile(
        r"\b(?:\d{1,3}\.){3}\d{1,3}(?::\d+)?\b"
    )
    UUID_PATTERN: re.Pattern[str] = re.compile(
        r"\b[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}\b"
    )
    PATH_UNIX_PATTERN: re.Pattern[str] = re.compile(
        r"(?<![A-Za-z0-9_])/(?:[\w.\-]+/)+[\w.\-]*"
    )
    PATH_WINDOWS_PATTERN: re.Pattern[str] = re.compile(
        r"\b[A-Za-z]:\\(?:[\w.\-]+\\)*[\w.\-]*"
    )
    HEX_PATTERN: re.Pattern[str] = re.compile(r"\b0x[0-9a-fA-F]+\b")
    SESSION_PATTERN: re.Pattern[str] = re.compile(
        r"(session(?:id)?[:\s]+)<HEX>", re.IGNORECASE
    )
    NUM_PATTERN: re.Pattern[str] = re.compile(r"\b\d+\b")
    LEVEL_PATTERN: re.Pattern[str] = re.compile(
        r"^\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3} - "
        r"(INFO|WARN|ERROR|DEBUG|TRACE|FATAL)\b"
    )

    def __init__(self) -> None:
        # Patterns are class-level and pre-compiled; nothing to do here, but
        # the explicit constructor lets callers swap an instance later if they
        # ever want a configurable subclass.
        pass

    def normalize(self, log_line: str) -> str:
        """Apply all replacements in fixed order and return the result."""
        s = log_line
        s = self.EXC_PATTERN.sub("<EXC>", s)
        s = self.IP_PATTERN.sub("<IP>", s)
        s = self.UUID_PATTERN.sub("<UUID>", s)
        s = self.PATH_UNIX_PATTERN.sub("<PATH>", s)
        s = self.PATH_WINDOWS_PATTERN.sub("<PATH>", s)
        s = self.HEX_PATTERN.sub("<HEX>", s)
        s = self.SESSION_PATTERN.sub(r"\1<SESSION>", s)
        s = self.NUM_PATTERN.sub("<NUM>", s)
        return s

    def extract_level(self, log_line: str) -> str | None:
        """Return the logging level from a ZooKeeper log line, or None.

        Expected line shape: ``YYYY-MM-DD HH:MM:SS,sss - LEVEL [...] - <content>``.
        Recognized levels: INFO, WARN, ERROR, DEBUG, TRACE, FATAL.
        """
        m = self.LEVEL_PATTERN.match(log_line)
        return m.group(1) if m else None


if __name__ == "__main__":
    n = RegexNormalizer()

    assert n.normalize("Caught java.lang.NullPointerException at start") == \
        "Caught <EXC> at start"

    assert n.normalize("Connected from 192.168.1.10:2181 ok") == \
        "Connected from <IP> ok"

    assert n.normalize("trace 550e8400-e29b-41d4-a716-446655440000 done") == \
        "trace <UUID> done"

    assert n.normalize("read /var/log/zookeeper/zk.log fast") == \
        "read <PATH> fast"

    assert n.normalize("read C:\\logs\\zk.log fast") == \
        "read <PATH> fast"

    assert n.normalize("address 0xdeadbeef found") == \
        "address <HEX> found"

    assert n.normalize("Established session 0x10000a3e7d0001 with timeout 30000") == \
        "Established session <SESSION> with timeout <NUM>"

    # ZooKeeper also writes "sessionid:0x..." (no space, colon-separated).
    assert n.normalize("client closing session, sessionid:0x10000a3e7d0001") == \
        "client closing session, sessionid:<SESSION>"

    assert n.normalize("Got 42 votes from peer 7") == \
        "Got <NUM> votes from peer <NUM>"

    assert n.extract_level(
        "2015-07-29 17:41:41,536 - INFO  [main:QuorumPeerConfig@101] - Reading cfg"
    ) == "INFO"

    assert n.extract_level(
        "2015-07-29 17:41:44,747 - WARN  [QuorumPeer[myid=1]/0:0:0:0:0:0:0:0:2181:Follower@89] - Got zxid"
    ) == "WARN"

    assert n.extract_level("garbage line with no timestamp") is None

    print("OK")
