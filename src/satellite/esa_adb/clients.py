"""ESA-ADB 8-client federated partition — real subsystem/group structure
only, never a random row split (per the user's explicit instruction).

Built directly from the real per-(Subsystem, Group) channel counts among
the 58 Target=YES channels (verified via channels.csv, see
docs/research/experiments/esa-adb/README.md Entry 2 and Entry 3):

    (subsystem_1, group 14):  3 channels
    (subsystem_3, group 18):  7 channels
    (subsystem_5, group  8):  6 channels
    (subsystem_6, group  3):  8 channels
    (subsystem_6, group  4):  5 channels
    (subsystem_6, group  5):  6 channels
    (subsystem_6, group  6):  5 channels
    (subsystem_6, group  7):  5 channels
    (subsystem_6, group  9):  3 channels
    (subsystem_6, group 10):  3 channels
    (subsystem_6, group 13):  4 channels
    (subsystem_6, group 15):  2 channels
    (subsystem_6, group 16):  1 channel

subsystem_1, subsystem_3, and subsystem_5 are each small enough to form
one client directly. subsystem_6 (42 of the 58 channels) is split by its
10 real groups into 5 sub-clients, grouped to balance channel counts
(8/9/9/8/8) while never splitting a real group across two clients.
"""

from __future__ import annotations

CLIENT_CHANNELS: dict[str, list[str]] = {
    "client_1": [f"channel_{i}" for i in (41, 42, 43, 44, 45, 46)],  # subsystem_5, group 8
    "client_2": [f"channel_{i}" for i in (70, 71, 72, 73, 74, 75, 76)],  # subsystem_3, group 18
    "client_3": [f"channel_{i}" for i in (61, 62, 63)],  # subsystem_1, group 14
    "client_4": [f"channel_{i}" for i in (12, 13, 19, 20, 27, 28, 36, 37)],  # subsystem_6, group 3
    "client_5": [f"channel_{i}" for i in (17, 18, 25, 26, 34, 35, 66, 64, 65)],  # subsystem_6, groups {5, 16, 15}
    "client_6": [f"channel_{i}" for i in (14, 21, 29, 30, 38, 57, 58, 59, 60)],  # subsystem_6, groups {4, 13}
    "client_7": [f"channel_{i}" for i in (16, 24, 32, 33, 40, 47, 48, 49)],  # subsystem_6, groups {6, 9}
    "client_8": [f"channel_{i}" for i in (15, 22, 23, 31, 39, 50, 51, 52)],  # subsystem_6, groups {7, 10}
}

CLIENT_SUBSYSTEM_GROUPS: dict[str, list[tuple[str, str]]] = {
    "client_1": [("subsystem_5", "8")],
    "client_2": [("subsystem_3", "18")],
    "client_3": [("subsystem_1", "14")],
    "client_4": [("subsystem_6", "3")],
    "client_5": [("subsystem_6", "5"), ("subsystem_6", "16"), ("subsystem_6", "15")],
    "client_6": [("subsystem_6", "4"), ("subsystem_6", "13")],
    "client_7": [("subsystem_6", "6"), ("subsystem_6", "9")],
    "client_8": [("subsystem_6", "7"), ("subsystem_6", "10")],
}


def validate_partition(target_yes_channels: list[str]) -> None:
    """Verify every approved Target=YES channel belongs to EXACTLY one
    client, and no client-assigned channel is outside the approved set.
    Raises AssertionError with a specific message on any violation —
    never silently repairs a bad partition."""
    assigned = [c for chans in CLIENT_CHANNELS.values() for c in chans]

    duplicates = {c for c in assigned if assigned.count(c) > 1}
    if duplicates:
        raise AssertionError(f"Channels assigned to more than one client: {sorted(duplicates)}")

    assigned_set = set(assigned)
    approved_set = set(target_yes_channels)
    missing = approved_set - assigned_set
    if missing:
        raise AssertionError(f"Approved Target=YES channels missing from the 8-client partition: {sorted(missing)}")
    extra = assigned_set - approved_set
    if extra:
        raise AssertionError(f"Client partition includes channels outside the approved Target=YES set: {sorted(extra)}")
