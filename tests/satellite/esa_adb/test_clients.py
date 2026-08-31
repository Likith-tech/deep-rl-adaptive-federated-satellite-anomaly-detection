"""8-client partition correctness (category 10) and no-duplicate-
channels-across-clients (a precondition for category 11, no duplicate
sequences across clients — sequences are built per channel, so disjoint
channels guarantee disjoint sequences)."""

from __future__ import annotations

import pytest

from src.satellite.esa_adb import clients as clients_mod


def test_eight_clients_defined():
    assert len(clients_mod.CLIENT_CHANNELS) == 8


def test_fifty_eight_channels_total_across_clients():
    total = sum(len(chans) for chans in clients_mod.CLIENT_CHANNELS.values())
    assert total == 58


def test_no_channel_assigned_to_more_than_one_client():
    seen = set()
    for chans in clients_mod.CLIENT_CHANNELS.values():
        for c in chans:
            assert c not in seen, f"{c} assigned to more than one client"
            seen.add(c)


def test_validate_partition_passes_for_the_approved_channel_set():
    approved = sorted({c for chans in clients_mod.CLIENT_CHANNELS.values() for c in chans})
    clients_mod.validate_partition(approved)  # should not raise


def test_validate_partition_rejects_missing_channel():
    """A channel present in the approved Target=YES list but absent
    from every client's assignment must be caught."""
    original = dict(clients_mod.CLIENT_CHANNELS)
    try:
        clients_mod.CLIENT_CHANNELS["client_1"] = clients_mod.CLIENT_CHANNELS["client_1"][1:]  # drop one channel
        approved = sorted({c for chans in original.values() for c in chans})  # the FULL, correct approved set
        with pytest.raises(AssertionError, match="missing"):
            clients_mod.validate_partition(approved)
    finally:
        clients_mod.CLIENT_CHANNELS.clear()
        clients_mod.CLIENT_CHANNELS.update(original)


def test_validate_partition_rejects_extra_channel():
    """A channel assigned to a client but outside the approved
    Target=YES list must be caught."""
    original = dict(clients_mod.CLIENT_CHANNELS)
    try:
        clients_mod.CLIENT_CHANNELS["client_1"] = clients_mod.CLIENT_CHANNELS["client_1"] + ["channel_999"]
        approved = sorted({c for chans in original.values() for c in chans})  # the FULL, correct approved set
        with pytest.raises(AssertionError, match="outside the approved"):
            clients_mod.validate_partition(approved)
    finally:
        clients_mod.CLIENT_CHANNELS.clear()
        clients_mod.CLIENT_CHANNELS.update(original)


def test_validate_partition_rejects_a_duplicated_channel():
    original = dict(clients_mod.CLIENT_CHANNELS)
    try:
        clients_mod.CLIENT_CHANNELS["client_1"] = clients_mod.CLIENT_CHANNELS["client_1"] + ["channel_70"]
        approved = sorted({c for chans in clients_mod.CLIENT_CHANNELS.values() for c in chans})
        with pytest.raises(AssertionError, match="more than one client"):
            clients_mod.validate_partition(approved)
    finally:
        clients_mod.CLIENT_CHANNELS.clear()
        clients_mod.CLIENT_CHANNELS.update(original)


def test_every_client_has_at_least_one_channel():
    for client, chans in clients_mod.CLIENT_CHANNELS.items():
        assert len(chans) > 0, f"{client} has no channels"


def test_client_subsystem_groups_defined_for_every_client():
    assert set(clients_mod.CLIENT_SUBSYSTEM_GROUPS.keys()) == set(clients_mod.CLIENT_CHANNELS.keys())
