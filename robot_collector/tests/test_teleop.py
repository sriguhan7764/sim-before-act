import pytest

from collector.teleop import (ADDR_GOAL, ADDR_TORQUE, build_sync_goal, build_write, initial_unwrapped, plan_goal,
                              wrap_delta)


def test_write_allowlist_refuses_other_registers():
    build_write(1, ADDR_TORQUE, b"\x01")
    build_write(1, ADDR_GOAL, b"\x00\x08")
    for addr in (31, 33, 48, 55):  # homing offset, mode, limits, lock
        with pytest.raises(PermissionError):
            build_write(1, addr, b"\x00")


def test_sync_goal_packet_checksum_and_wrap():
    p = build_sync_goal({1: 4100, 2: 10})
    assert p[:2] == b"\xff\xff" and p[2] == 0xFE and p[4] == 0x83
    assert (~sum(p[2:-1])) & 0xFF == p[-1]
    assert p[7:10] == bytes([1, 4, 0])  # 4100 % 4096 = 4


def test_wrap_delta_and_start_unwrap():
    assert wrap_delta(5, 4090) == 11 and wrap_delta(4090, 5) == -11
    assert initial_unwrapped(100, (2340, 4561), 100) == 4196
    with pytest.raises(ValueError):
        initial_unwrapped(1000, (2340, 4561), 100)


def test_plan_goal_no_jump_clamp_and_slew():
    cmd, _ = plan_goal(2000, 2000, 500, 500, 1, 1.0, (1800, 2500), 0, 12)
    assert cmd == 2000  # leader at its reference -> follower stays
    cmd, tgt = plan_goal(2000, 2000, 500, 900, 1, 1.0, (1800, 2500), 0, 12)
    assert cmd == 2012 and tgt == 2400  # slew limited
    cmd, _ = plan_goal(2499, 2000, 500, 3000, 1, 1.0, (1800, 2500), 0, 12)
    assert cmd == 2500  # clamped to range
    cmd, _ = plan_goal(2000, 2000, 500, 400, -1, 1.0, None, 0, 1000)
    assert cmd == 2100  # inverted
