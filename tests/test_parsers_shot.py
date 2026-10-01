"""Tests for .slog binary shot file parser."""

import struct
from pathlib import Path

import pytest

from gaggimate_mcp.parsers.shot import parse_binary_shot


class TestBinaryShotParser:
    """Test binary shot file parsing."""

    def test_parse_minimal_shot_v4(self):
        """Test parsing a minimal V4 shot file."""
        # Create minimal valid V4 shot file
        magic = 0x544F4853  # 'SHOT'
        version = 4
        reserved0 = 0
        header_size = 128
        sample_interval = 100  # milliseconds
        reserved1 = 0
        fields_mask = 0b1111  # T, TT, CT, TP (4 fields)
        sample_count = 2
        duration = 20000  # 20 seconds
        timestamp = 1640000000
        profile_id = b'test_profile\x00' + b'\x00' * 19
        profile_name = b'Test Profile\x00' + b'\x00' * 35
        weight = 360  # 36.0 grams

        # Build header
        header = struct.pack(
            '<IB B H H H I I I I 32s 48s H',
            magic, version, reserved0, header_size,
            sample_interval, reserved1,
            fields_mask, sample_count, duration, timestamp,
            profile_id, profile_name, weight
        )

        # Pad header to 128 bytes
        header = header + b'\x00' * (128 - len(header))

        # Sample data: 2 samples, 4 fields each (8 bytes per sample)
        # Sample 1: t=1000ms, tt=93.0°C, ct=92.5°C, tp=9.0 bar
        sample1 = struct.pack('<HHHH', 10, 930, 925, 90)

        # Sample 2: t=2000ms, tt=93.0°C, ct=93.0°C, tp=9.0 bar
        sample2 = struct.pack('<HHHH', 20, 930, 930, 90)

        binary_data = header + sample1 + sample2

        # Parse
        shot = parse_binary_shot(binary_data, "000001")

        # Verify
        assert shot.id == "000001"
        assert shot.version == 4
        assert shot.fields_mask == 0b1111
        assert shot.sample_count == 2
        assert shot.sample_interval == 100
        assert shot.profile_id == "test_profile"
        assert shot.profile_name == "Test Profile"
        assert shot.timestamp == 1640000000
        assert shot.duration == 20000
        assert shot.weight == 36.0
        assert shot.incomplete == False
        assert len(shot.samples) == 2
        assert len(shot.phases) == 0  # V4 has no phase data

        # Check first sample
        assert shot.samples[0]['t'] == 1000  # 10 * 100ms
        assert shot.samples[0]['tt'] == 93.0  # 930 / 10
        assert shot.samples[0]['ct'] == 92.5  # 925 / 10
        assert shot.samples[0]['tp'] == 9.0  # 90 / 10

    def test_parse_shot_invalid_magic(self):
        """Test parsing shot file with invalid magic number."""
        invalid_data = struct.pack('<I', 0xDEADBEEF) + b'\x00' * 124

        try:
            parse_binary_shot(invalid_data, "000001")
            assert False, "Should have raised ValueError"
        except ValueError as e:
            assert "Invalid shot magic" in str(e)

    def test_parse_shot_html_response_gives_friendly_error(self):
        """Test that HTML response (firmware 1.8.0 issue) gives a clear error."""
        html_data = b'<!DOCTYPE html><html><body>Not Found</body></html>'
        html_data = html_data + b'\x00' * (128 - len(html_data))

        try:
            parse_binary_shot(html_data, "000195")
            assert False, "Should have raised ValueError"
        except ValueError as e:
            assert "HTML" in str(e)
            assert "overloaded" in str(e)

    def test_parse_shot_short_html_detected_before_size_check(self):
        """Short HTML responses should get HTML error, not 'too small' error."""
        short_html = b'<html>err</html>'  # < 128 bytes
        try:
            parse_binary_shot(short_html, "000195")
            assert False, "Should have raised ValueError"
        except ValueError as e:
            assert "HTML" in str(e)
            assert "too small" not in str(e)

    def test_parse_shot_firmware_1_8_0_magic_bytes(self):
        """Test the specific magic bytes from the 1.8.0 error report."""
        # 0x6f64213c = "<!do" in little-endian ASCII
        data = b'<!doctype html>' + b'\x00' * 113

        try:
            parse_binary_shot(data, "000195")
            assert False, "Should have raised ValueError"
        except ValueError as e:
            assert "HTML" in str(e)

    def test_parse_shot_too_small(self):
        """Test parsing shot file that's too small."""
        too_small = b'\x00' * 50

        try:
            parse_binary_shot(too_small, "000001")
            assert False, "Should have raised ValueError"
        except ValueError as e:
            assert "too small" in str(e)

    def test_parse_shot_with_phase_transitions(self):
        """Test parsing V5 shot file with phase transitions."""
        magic = 0x544F4853
        version = 5
        reserved0 = 0
        header_size = 512
        sample_interval = 100
        reserved1 = 0
        fields_mask = 0b11  # T, TT (2 fields)
        sample_count = 3
        duration = 30000
        timestamp = 1640000000
        profile_id = b'v5_profile\x00' + b'\x00' * 21
        profile_name = b'V5 Profile\x00' + b'\x00' * 37
        weight = 400  # 40.0 grams

        # Build header up to weight (110 bytes)
        header_part1 = struct.pack(
            '<IB B H H H I I I I 32s 48s H',
            magic, version, reserved0, header_size,
            sample_interval, reserved1,
            fields_mask, sample_count, duration, timestamp,
            profile_id, profile_name, weight
        )

        # Phase transitions: 2 transitions
        # Transition 1: sample 0, phase 0, "Preinfusion"
        transition1 = struct.pack('<HB x 25s', 0, 0, b'Preinfusion\x00' + b'\x00' * 13)

        # Transition 2: sample 2, phase 1, "Extraction"
        transition2 = struct.pack('<HB x 25s', 2, 1, b'Extraction\x00' + b'\x00' * 14)

        # Pad to transition count position (458 bytes)
        padding = b'\x00' * (458 - len(header_part1) - len(transition1) - len(transition2))

        # Transition count
        transition_count = struct.pack('B', 2)

        # Pad to 512 bytes
        header = header_part1 + transition1 + transition2 + padding + transition_count
        header = header + b'\x00' * (512 - len(header))

        # Sample data: 3 samples, 2 fields each
        sample1 = struct.pack('<HH', 10, 900)  # t=1000ms, tt=90.0°C
        sample2 = struct.pack('<HH', 20, 920)  # t=2000ms, tt=92.0°C
        sample3 = struct.pack('<HH', 30, 930)  # t=3000ms, tt=93.0°C

        binary_data = header + sample1 + sample2 + sample3

        # Parse
        shot = parse_binary_shot(binary_data, "000002")

        # Verify
        assert shot.version == 5
        assert len(shot.phases) == 2
        assert shot.phases[0].sample_index == 0
        assert shot.phases[0].phase_number == 0
        assert shot.phases[0].phase_name == "Preinfusion"
        assert shot.phases[1].sample_index == 2
        assert shot.phases[1].phase_number == 1
        assert shot.phases[1].phase_name == "Extraction"

        # Check phase numbers in samples
        assert shot.samples[0].get('phase') == 0
        assert shot.samples[1].get('phase') == 0
        assert shot.samples[2].get('phase') == 1

    def test_parse_incomplete_shot(self):
        """Test parsing a shot file with incomplete samples."""
        magic = 0x544F4853
        version = 4
        reserved0 = 0
        header_size = 128
        sample_interval = 100
        reserved1 = 0
        fields_mask = 0b11  # T, TT (2 fields)
        sample_count = 10  # Claims 10 samples
        duration = 10000
        timestamp = 1640000000
        profile_id = b'incomplete\x00' + b'\x00' * 21
        profile_name = b'Incomplete\x00' + b'\x00' * 37
        weight = 0

        header = struct.pack(
            '<IB B H H H I I I I 32s 48s H',
            magic, version, reserved0, header_size,
            sample_interval, reserved1,
            fields_mask, sample_count, duration, timestamp,
            profile_id, profile_name, weight
        )
        header = header + b'\x00' * (128 - len(header))

        # Only provide 2 samples worth of data (but claims 10)
        sample1 = struct.pack('<HH', 10, 900)
        sample2 = struct.pack('<HH', 20, 920)

        binary_data = header + sample1 + sample2

        # Parse
        shot = parse_binary_shot(binary_data, "000003")

        # Should only parse the 2 samples that exist
        assert shot.sample_count == 2
        assert shot.incomplete == True
        assert len(shot.samples) == 2


class TestShotFormatV6V7:
    """Real v7 shot (firmware with 30-byte samples, uint32 ms timestamps)."""

    FIXTURE = str(Path(__file__).parent / 'fixtures' / 'shot_33_v7_adaptive.slog')

    def _parse(self):
        with open(self.FIXTURE, 'rb') as f:
            return parse_binary_shot(f.read(), "000033")

    def test_v7_sample_stride_and_count(self):
        shot = self._parse()
        assert shot.version == 7
        assert shot.sample_count == 152
        assert shot.incomplete is False

    def test_v7_timestamps_are_elapsed_ms(self):
        shot = self._parse()
        ts = [s['t'] for s in shot.samples]
        assert ts == sorted(ts)
        assert ts[0] < 1000
        assert 37000 <= ts[-1] <= 38500  # header duration is 38154 ms

    def test_v7_values_physically_plausible(self):
        shot = self._parse()
        for s in shot.samples:
            assert 0 <= s['ct'] <= 100
            assert 0 <= s['cp'] <= 15
            assert 0 <= s['v'] < 60
        assert shot.samples[0]['tt'] == 93.0
        assert shot.samples[-1]['ct'] == 86.8

    def test_v7_exposes_water_pumped(self):
        shot = self._parse()
        wps = [s['wp'] for s in shot.samples]
        assert wps == sorted(wps)  # cumulative

    def test_v7_phases_keep_firmware_numbering(self):
        shot = self._parse()
        assert [(p.phase_number, p.phase_name) for p in shot.phases] == [
            (0, 'Prefill'), (2, 'Compressing'), (3, 'Dripping'),
            (4, 'Pressurize'), (5, 'Extraction'),
        ]

    def test_unknown_future_version_rejected(self):
        with open(self.FIXTURE, 'rb') as f:
            data = bytearray(f.read())
        data[4] = 99
        with pytest.raises(ValueError, match="version"):
            parse_binary_shot(bytes(data), "000033")

    def test_unknown_field_bit_rejected(self):
        with open(self.FIXTURE, 'rb') as f:
            data = bytearray(f.read())
        struct.pack_into('<I', data, 12, 0x7fff)  # bit 14 not defined
        with pytest.raises(ValueError, match="field"):
            parse_binary_shot(bytes(data), "000033")


def _build_slog(version, mask, rows, declared_sample_size=None, count=None):
    """Build a synthetic .slog: rows are tuples of raw ints in mask-bit order."""
    widths = {0: 4 if version >= 6 else 2}
    fmts = {0: '<I' if version >= 6 else '<H'}
    for bit in range(1, 14):
        fmts[bit] = '<h' if bit in (5, 6, 7, 8) else '<H'
        widths[bit] = 2
    bits = [b for b in range(14) if mask & (1 << b)]
    stride = sum(widths[b] for b in bits)
    header_size = 512 if version >= 5 else 128
    hdr = bytearray(header_size)
    struct.pack_into('<I', hdr, 0, 0x544F4853)
    hdr[4] = version
    if version >= 5:
        hdr[5] = stride if declared_sample_size is None else declared_sample_size
        struct.pack_into('<H', hdr, 6, header_size)
    struct.pack_into('<H', hdr, 8, 250)
    struct.pack_into('<I', hdr, 12, mask)
    struct.pack_into('<I', hdr, 16, len(rows) if count is None else count)
    body = b''.join(
        b''.join(struct.pack(fmts[b], v) for b, v in zip(bits, row)) for row in rows
    )
    return bytes(hdr) + body, stride


class TestSyntheticVersions:
    # bits: t, tt, ct (v5/v6) and t, ct, wp (v7)
    def test_v5_t_is_index_times_interval(self):
        data, stride = _build_slog(5, 0b111, [(0, 930, 900), (1, 930, 910)])
        shot = parse_binary_shot(data, "x")
        assert stride == 6
        assert [s['t'] for s in shot.samples] == [0, 250]
        assert 'wp' not in shot.samples[0]

    def test_v6_t_is_uint32_ms_without_wp(self):
        data, stride = _build_slog(6, 0b111, [(70000, 930, 900), (70123, 930, 910)])
        shot = parse_binary_shot(data, "x")
        assert stride == 8
        assert [s['t'] for s in shot.samples] == [70000, 70123]
        assert shot.samples[1]['ct'] == 91.0
        assert 'wp' not in shot.samples[0]

    def test_v7_decodes_wp(self):
        mask = (1 << 0) | (1 << 2) | (1 << 13)
        data, stride = _build_slog(7, mask, [(100, 900, 0), (350, 910, 12)])
        shot = parse_binary_shot(data, "x")
        assert stride == 8
        assert [s['wp'] for s in shot.samples] == [0, 12]
        assert [s['t'] for s in shot.samples] == [100, 350]

    def test_v7_truncated_mid_sample_drops_partial(self):
        mask = (1 << 0) | (1 << 2) | (1 << 13)
        data, stride = _build_slog(7, mask, [(100, 900, 0), (350, 910, 12), (600, 920, 20)])
        shot = parse_binary_shot(data[:-3], "x")
        assert shot.sample_count == 2
        assert shot.incomplete is True

    def test_wp_bit_rejected_before_v7(self):
        data, _ = _build_slog(5, (1 << 0) | (1 << 13), [(0, 1)])
        with pytest.raises(ValueError, match="wp"):
            parse_binary_shot(data, "x")

    def test_header_sample_size_mismatch_rejected(self):
        data, _ = _build_slog(5, 0b111, [(0, 930, 900)], declared_sample_size=26)
        with pytest.raises(ValueError, match="Sample size mismatch"):
            parse_binary_shot(data, "x")
