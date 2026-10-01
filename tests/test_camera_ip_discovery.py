"""Tests for MAC-based camera IP auto-recovery (discovery.py + frame_buffer wiring)."""

import socket
import uuid
from types import SimpleNamespace

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from app.modules.cameras import discovery


# ---------------------------------------------------------------------------
# discovery.py — pure helpers
# ---------------------------------------------------------------------------

def test_normalize_mac():
    assert discovery.normalize_mac("AA-BB-CC-DD-EE-FF") == "aa:bb:cc:dd:ee:ff"
    assert discovery.normalize_mac(" aa:bb:cc:dd:ee:ff ") == "aa:bb:cc:dd:ee:ff"


def test_is_valid_mac():
    assert discovery.is_valid_mac("AA:BB:CC:DD:EE:FF")
    assert not discovery.is_valid_mac("not-a-mac")
    assert not discovery.is_valid_mac("AA:BB:CC:DD:EE")


def test_subnet_from_ip():
    assert discovery.subnet_from_ip("192.168.0.84") == "192.168.0.0/24"


def _fake_snic(family, address, netmask):
    return SimpleNamespace(family=family, address=address, netmask=netmask, ptp=None, broadcast=None)


def test_local_subnet_for_ip_uses_real_interface_netmask():
    """A /23 network (two /24s) must resolve to its real /23 boundary, not
    get truncated to an assumed /24 - this is the exact gap that made a
    camera on the far half of a real /23 permanently unfindable."""
    fake_addrs = {
        "en0": [_fake_snic(socket.AF_INET, "192.168.1.32", "255.255.254.0")],
    }
    with patch.object(discovery.psutil, "net_if_addrs", return_value=fake_addrs):
        assert discovery.local_subnet_for_ip("192.168.1.21") == "192.168.0.0/23"
        # An IP on the "other half" of the same /23 that a /24 guess would miss.
        assert discovery.local_subnet_for_ip("192.168.0.55") == "192.168.0.0/23"


def test_local_subnet_for_ip_falls_back_when_no_interface_matches():
    fake_addrs = {
        "en0": [_fake_snic(socket.AF_INET, "10.0.0.5", "255.255.255.0")],
    }
    with patch.object(discovery.psutil, "net_if_addrs", return_value=fake_addrs):
        assert discovery.local_subnet_for_ip("192.168.5.9") == "192.168.5.0/24"


def test_local_subnet_for_ip_falls_back_on_interface_read_error():
    with patch.object(discovery.psutil, "net_if_addrs", side_effect=OSError("no interfaces")):
        assert discovery.local_subnet_for_ip("192.168.5.9") == "192.168.5.0/24"


def test_extract_host():
    assert discovery.extract_host("rtsp://admin:pass@192.168.0.84:554/stream1") == "192.168.0.84"
    assert discovery.extract_host("rtsp://192.168.0.84/stream1") == "192.168.0.84"
    assert discovery.extract_host("not-a-url") is None


def test_rebuild_rtsp_url_swaps_host_keeps_creds_port_path():
    old = "rtsp://admin:pass@192.168.0.84:554/stream1"
    new = discovery.rebuild_rtsp_url(old, "192.168.0.120")
    assert new == "rtsp://admin:pass@192.168.0.120:554/stream1"


def test_rebuild_rtsp_url_no_creds():
    old = "rtsp://192.168.0.84/stream1"
    new = discovery.rebuild_rtsp_url(old, "192.168.0.120")
    assert new == "rtsp://192.168.0.120/stream1"


def test_rebuild_rtsp_url_rejects_bad_input():
    with pytest.raises(ValueError):
        discovery.rebuild_rtsp_url("not-a-url", "192.168.0.120")


# ---------------------------------------------------------------------------
# resolve_ip_by_mac_sync / resolve_mac_by_ip_sync — the real scan logic.
# Plain sync functions, no event loop involved - patch the sync ping/arp
# helpers directly and call with no thread/asyncio machinery needed.
# ---------------------------------------------------------------------------

def test_resolve_ip_by_mac_sync_found():
    with patch.object(discovery, "_ping_sweep_sync"), \
         patch.object(discovery, "_read_arp_table_sync",
                       return_value={"aa:bb:cc:dd:ee:ff": "192.168.0.120"}):
        ip = discovery.resolve_ip_by_mac_sync("AA:BB:CC:DD:EE:FF", "192.168.0.0/24")
        assert ip == "192.168.0.120"


def test_resolve_ip_by_mac_sync_not_found():
    with patch.object(discovery, "_ping_sweep_sync"), \
         patch.object(discovery, "_read_arp_table_sync", return_value={}):
        ip = discovery.resolve_ip_by_mac_sync("AA:BB:CC:DD:EE:FF", "192.168.0.0/24")
        assert ip is None


def test_resolve_ip_by_mac_sync_invalid_mac_short_circuits():
    with patch.object(discovery, "_ping_sweep_sync") as sweep:
        ip = discovery.resolve_ip_by_mac_sync("garbage", "192.168.0.0/24")
        assert ip is None
        sweep.assert_not_called()


def test_resolve_mac_by_ip_sync_found():
    with patch.object(discovery, "_ping_sync"), \
         patch.object(discovery, "_read_arp_table_sync",
                       return_value={"aa:bb:cc:dd:ee:ff": "192.168.0.120"}):
        mac = discovery.resolve_mac_by_ip_sync("192.168.0.120")
        assert mac == "aa:bb:cc:dd:ee:ff"


def test_resolve_mac_by_ip_sync_not_found():
    with patch.object(discovery, "_ping_sync"), \
         patch.object(discovery, "_read_arp_table_sync", return_value={}):
        mac = discovery.resolve_mac_by_ip_sync("192.168.0.120")
        assert mac is None


# ---------------------------------------------------------------------------
# async wrappers — used from FastAPI request handlers (already inside a
# running loop); must delegate to the sync implementation via a thread,
# never create their own event loop.
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_resolve_ip_by_mac_async_wrapper_delegates_to_sync():
    with patch.object(discovery, "resolve_ip_by_mac_sync", return_value="192.168.0.120") as sync_fn:
        ip = await discovery.resolve_ip_by_mac("AA:BB:CC:DD:EE:FF", "192.168.0.0/24")
        assert ip == "192.168.0.120"
        sync_fn.assert_called_once_with("AA:BB:CC:DD:EE:FF", "192.168.0.0/24")


@pytest.mark.asyncio
async def test_resolve_mac_by_ip_async_wrapper_delegates_to_sync():
    with patch.object(discovery, "resolve_mac_by_ip_sync", return_value="aa:bb:cc:dd:ee:ff") as sync_fn:
        mac = await discovery.resolve_mac_by_ip("192.168.0.120")
        assert mac == "aa:bb:cc:dd:ee:ff"
        sync_fn.assert_called_once_with("192.168.0.120")


# ---------------------------------------------------------------------------
# ONVIF (WS-Discovery) - fallback identity for a camera whose MAC rotates
# (Wi-Fi privacy addressing). Its endpoint UUID is assigned at the
# firmware/hardware level and doesn't change with the MAC.
# ---------------------------------------------------------------------------

_PROBE_MATCH_XML = """<?xml version="1.0" encoding="UTF-8"?>
<soap:Envelope xmlns:soap="http://www.w3.org/2003/05/soap-envelope"
    xmlns:wsa="http://schemas.xmlsoap.org/ws/2004/08/addressing"
    xmlns:wsd="http://schemas.xmlsoap.org/ws/2005/04/discovery">
  <soap:Body>
    <wsd:ProbeMatches>
      <wsd:ProbeMatch>
        <wsa:EndpointReference>
          <wsa:Address>urn:uuid:09501d80-190d-11f0-99bf-001c272683ee</wsa:Address>
        </wsa:EndpointReference>
        <wsd:XAddrs>http://192.168.1.21/onvif/device_service</wsd:XAddrs>
      </wsd:ProbeMatch>
    </wsd:ProbeMatches>
  </soap:Body>
</soap:Envelope>"""


def test_parse_probe_match_extracts_uuid_and_xaddr():
    result = discovery._parse_probe_match(_PROBE_MATCH_XML.encode("utf-8"))
    assert result == ("urn:uuid:09501d80-190d-11f0-99bf-001c272683ee", "http://192.168.1.21/onvif/device_service")


def test_parse_probe_match_rejects_garbage():
    assert discovery._parse_probe_match(b"not xml at all") is None
    assert discovery._parse_probe_match(b"<empty/>") is None


def test_resolve_ip_by_onvif_id_sync_found():
    with patch.object(
        discovery, "_ws_discovery_probe",
        return_value={"urn:uuid:abc": "http://192.168.1.21:8080/onvif/device_service"},
    ):
        ip = discovery.resolve_ip_by_onvif_id_sync("urn:uuid:abc")
        assert ip == "192.168.1.21"


def test_resolve_ip_by_onvif_id_sync_not_found():
    with patch.object(discovery, "_ws_discovery_probe", return_value={}):
        assert discovery.resolve_ip_by_onvif_id_sync("urn:uuid:abc") is None


def test_resolve_ip_by_onvif_id_sync_empty_id_short_circuits():
    with patch.object(discovery, "_ws_discovery_probe") as probe:
        assert discovery.resolve_ip_by_onvif_id_sync("") is None
        probe.assert_not_called()


def test_resolve_onvif_id_by_ip_sync_found():
    with patch.object(
        discovery, "_ws_discovery_probe",
        return_value={"urn:uuid:abc": "http://192.168.1.21:8080/onvif/device_service"},
    ):
        onvif_id = discovery.resolve_onvif_id_by_ip_sync("192.168.1.21")
        assert onvif_id == "urn:uuid:abc"


def test_resolve_onvif_id_by_ip_sync_not_found():
    with patch.object(discovery, "_ws_discovery_probe", return_value={}):
        assert discovery.resolve_onvif_id_by_ip_sync("192.168.1.21") is None


@pytest.mark.asyncio
async def test_resolve_ip_by_onvif_id_async_wrapper_delegates_to_sync():
    with patch.object(discovery, "resolve_ip_by_onvif_id_sync", return_value="192.168.1.21") as sync_fn:
        ip = await discovery.resolve_ip_by_onvif_id("urn:uuid:abc")
        assert ip == "192.168.1.21"
        sync_fn.assert_called_once()


@pytest.mark.asyncio
async def test_resolve_onvif_id_by_ip_async_wrapper_delegates_to_sync():
    with patch.object(discovery, "resolve_onvif_id_by_ip_sync", return_value="urn:uuid:abc") as sync_fn:
        onvif_id = await discovery.resolve_onvif_id_by_ip("192.168.1.21")
        assert onvif_id == "urn:uuid:abc"
        sync_fn.assert_called_once()


# ---------------------------------------------------------------------------
# frame_buffer._try_resolve_new_ip — reconnect wiring.
#
# This method must never create an asyncio event loop: it's called on every
# failed reconnect from a plain background thread with no loop, and it used
# to call asyncio.run() per call - on macOS that leaks file descriptors
# across repeated loop creation, and enough retries during an extended
# camera outage exhausts the whole process's fd limit and takes the entire
# server down, not just this camera. It now calls resolve_ip_by_mac_sync
# directly (plain function call, no event loop), so these tests call it
# straight from the test thread same as production's capture thread does -
# no asyncio involved anywhere in this path any more.
# ---------------------------------------------------------------------------

def _make_buffer(mac_address=None, onvif_id=None, on_ip_resolved=None):
    from app.modules.ai_runtime.frame_buffer import LatestFrameBuffer
    return LatestFrameBuffer(
        "rtsp://admin:pass@192.168.0.84:554/stream1",
        mac_address=mac_address,
        onvif_id=onvif_id,
        on_ip_resolved=on_ip_resolved,
    )


def test_try_resolve_new_ip_noop_without_mac():
    buf = _make_buffer(mac_address=None)
    assert buf._try_resolve_new_ip() is False
    assert buf.rtsp_url == "rtsp://admin:pass@192.168.0.84:554/stream1"


def test_try_resolve_new_ip_patches_url_and_fires_callback():
    callback = MagicMock()
    buf = _make_buffer(mac_address="AA:BB:CC:DD:EE:FF", on_ip_resolved=callback)

    with patch(
        "app.modules.cameras.discovery.resolve_ip_by_mac_sync",
        return_value="192.168.0.120",
    ):
        resolved = buf._try_resolve_new_ip()

    assert resolved is True
    assert buf.rtsp_url == "rtsp://admin:pass@192.168.0.120:554/stream1"
    callback.assert_called_once_with("rtsp://admin:pass@192.168.0.120:554/stream1")


def test_try_resolve_new_ip_false_when_mac_not_found():
    callback = MagicMock()
    buf = _make_buffer(mac_address="AA:BB:CC:DD:EE:FF", on_ip_resolved=callback)

    with patch(
        "app.modules.cameras.discovery.resolve_ip_by_mac_sync",
        return_value=None,
    ):
        resolved = buf._try_resolve_new_ip()

    assert resolved is False
    assert buf.rtsp_url == "rtsp://admin:pass@192.168.0.84:554/stream1"
    callback.assert_not_called()


def test_try_resolve_new_ip_false_when_ip_unchanged():
    """MAC found but at the same IP - nothing to patch, don't report success."""
    callback = MagicMock()
    buf = _make_buffer(mac_address="AA:BB:CC:DD:EE:FF", on_ip_resolved=callback)

    with patch(
        "app.modules.cameras.discovery.resolve_ip_by_mac_sync",
        return_value="192.168.0.84",
    ):
        resolved = buf._try_resolve_new_ip()

    assert resolved is False
    callback.assert_not_called()


def test_try_resolve_new_ip_callback_failure_does_not_crash():
    """A broken on_ip_resolved callback (e.g. DB write scheduling failure)
    must not prevent the buffer itself from using the corrected URL."""
    callback = MagicMock(side_effect=RuntimeError("db down"))
    buf = _make_buffer(mac_address="AA:BB:CC:DD:EE:FF", on_ip_resolved=callback)

    with patch(
        "app.modules.cameras.discovery.resolve_ip_by_mac_sync",
        return_value="192.168.0.120",
    ):
        resolved = buf._try_resolve_new_ip()

    assert resolved is True
    assert buf.rtsp_url == "rtsp://admin:pass@192.168.0.120:554/stream1"


def test_try_resolve_new_ip_falls_back_to_onvif_when_mac_search_empty():
    """The case ONVIF exists for: a Wi-Fi camera whose MAC has rotated, so
    the MAC-based search finds nothing even though the camera is up - the
    ONVIF endpoint UUID it forwards to hasn't changed and should still
    locate it."""
    callback = MagicMock()
    buf = _make_buffer(mac_address="AA:BB:CC:DD:EE:FF", onvif_id="urn:uuid:abc", on_ip_resolved=callback)

    with patch("app.modules.cameras.discovery.resolve_ip_by_mac_sync", return_value=None) as mac_fn, \
         patch("app.modules.cameras.discovery.resolve_ip_by_onvif_id_sync", return_value="192.168.0.120") as onvif_fn:
        resolved = buf._try_resolve_new_ip()

    assert resolved is True
    assert buf.rtsp_url == "rtsp://admin:pass@192.168.0.120:554/stream1"
    callback.assert_called_once_with("rtsp://admin:pass@192.168.0.120:554/stream1")
    mac_fn.assert_called_once()
    onvif_fn.assert_called_once()


def test_try_resolve_new_ip_skips_onvif_when_mac_search_succeeds():
    """ONVIF (a full WS-Discovery multicast round trip) should never fire
    when the cheaper MAC-based search already found the answer."""
    buf = _make_buffer(mac_address="AA:BB:CC:DD:EE:FF", onvif_id="urn:uuid:abc")

    with patch("app.modules.cameras.discovery.resolve_ip_by_mac_sync", return_value="192.168.0.120"), \
         patch("app.modules.cameras.discovery.resolve_ip_by_onvif_id_sync") as onvif_fn:
        resolved = buf._try_resolve_new_ip()

    assert resolved is True
    onvif_fn.assert_not_called()


def test_try_resolve_new_ip_onvif_only_camera_with_no_mac():
    """A camera onboarded with only an ONVIF id on file (no MAC captured)
    should still be findable via ONVIF alone."""
    buf = _make_buffer(mac_address=None, onvif_id="urn:uuid:abc")

    with patch("app.modules.cameras.discovery.resolve_ip_by_onvif_id_sync", return_value="192.168.0.120"):
        resolved = buf._try_resolve_new_ip()

    assert resolved is True
    assert buf.rtsp_url == "rtsp://admin:pass@192.168.0.120:554/stream1"


def test_try_resolve_new_ip_false_when_both_mac_and_onvif_fail():
    buf = _make_buffer(mac_address="AA:BB:CC:DD:EE:FF", onvif_id="urn:uuid:abc")

    with patch("app.modules.cameras.discovery.resolve_ip_by_mac_sync", return_value=None), \
         patch("app.modules.cameras.discovery.resolve_ip_by_onvif_id_sync", return_value=None):
        resolved = buf._try_resolve_new_ip()

    assert resolved is False


def test_try_resolve_new_ip_does_not_touch_asyncio():
    """Regression guard for the fd-leak bug: this path must never create an
    event loop. Calling it many times back-to-back on a bare thread with no
    loop (like the real capture thread) must never raise - asyncio.run()
    would eventually misbehave under repeated same-thread invocation in
    ways this simple loop is enough to catch if the old behavior came back."""
    buf = _make_buffer(mac_address="AA:BB:CC:DD:EE:FF")
    with patch(
        "app.modules.cameras.discovery.resolve_ip_by_mac_sync",
        return_value=None,
    ):
        for _ in range(20):
            assert buf._try_resolve_new_ip() is False


# ---------------------------------------------------------------------------
# camera_worker — zone/store identity must survive an IP-recovery persist
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_persist_resolved_rtsp_url_only_touches_rtsp_url():
    """The DB update on IP-recovery must be scoped to rtsp_url alone - the
    camera row's id/zone/store (and therefore its zone assignment) must
    never be touched by this path."""
    with patch("app.modules.ai_runtime.camera_worker.LatestFrameBuffer"), \
         patch("app.modules.ai_runtime.camera_worker.get_camera_detector"):
        from app.modules.ai_runtime.camera_worker import CameraWorker

        camera_config = {
            "id": uuid.uuid4(),
            "rtsp_url": "rtsp://192.168.0.84:554/stream1",
            "role": "general",
            "fps_target": 10,
            "reid_enabled": False,
            "demographic_enabled": False,
            "frame_rotation": 0,
            "mac_address": "AA:BB:CC:DD:EE:FF",
        }
        worker = CameraWorker(camera_config, {"zones": [], "views": [], "rules": []})

        mock_db = AsyncMock()
        mock_session_cm = AsyncMock()
        mock_session_cm.__aenter__.return_value = mock_db
        mock_session_cm.__aexit__.return_value = False

        with patch(
            "app.modules.ai_runtime.camera_worker.AsyncSessionLocal",
            return_value=mock_session_cm,
        ):
            await worker._persist_resolved_rtsp_url("rtsp://192.168.0.120:554/stream1")

        mock_db.execute.assert_called_once()
        stmt = mock_db.execute.call_args[0][0]
        # sqlalchemy Update construct: only rtsp_url in the SET values, camera
        # id only appears in the WHERE clause (identity untouched).
        assert {c.name for c in stmt._values.keys()} == {"rtsp_url"}
        mock_db.commit.assert_called_once()
        assert worker.camera_config["rtsp_url"] == "rtsp://192.168.0.120:554/stream1"
