"""Parser for .slog binary shot files.

Mirrors shot_log_format.h from the Gaggimate firmware.
"""

import struct
from dataclasses import dataclass, field
from typing import Optional


# Constants
HEADER_SIZE_V4 = 128
HEADER_SIZE_V5 = 512
MAGIC = 0x544F4853  # 'SHOT'
MAX_SUPPORTED_VERSION = 7
# v6+ stores `t` as uint32 elapsed milliseconds (v1-v5: uint16 sample index)
T_MS_VERSION = 6

# Scaling factors
TEMP_SCALE = 10
PRESSURE_SCALE = 10
FLOW_SCALE = 100
WEIGHT_SCALE = 10
RESISTANCE_SCALE = 100

# Field bit positions (must match shot_log_format.h)
FIELD_BITS = {
    'T': 0,    # tick
    'TT': 1,   # target temp
    'CT': 2,   # current temp
    'TP': 3,   # target pressure
    'CP': 4,   # current pressure
    'FL': 5,   # pump flow
    'TF': 6,   # target flow
    'PF': 7,   # puck flow
    'VF': 8,   # volumetric flow
    'V': 9,    # volumetric weight
    'EV': 10,  # estimated weight
    'PR': 11,  # puck resistance
    'SI': 12,  # system info (v2+)
    'WP': 13,  # cumulative water pumped (v7+)
}
KNOWN_FIELD_MASK = (1 << (max(FIELD_BITS.values()) + 1)) - 1

STRUCT_FORMATS = {'uint16': '<H', 'int16': '<h', 'uint32': '<I'}
FIELD_WIDTHS = {name: struct.calcsize(fmt) for name, fmt in STRUCT_FORMATS.items()}
WP_VERSION = 7  # `wp` (bit 13) first appears in v7


@dataclass
class FieldDef:
    """Field definition with parsing info."""
    name: str
    field_type: str  # 'uint8', 'uint16', 'int16'
    scale: Optional[float] = None
    transform: Optional[callable] = None


# Field definitions
FIELD_DEFS = {
    FIELD_BITS['T']: FieldDef(
        name='t',
        field_type='uint16',
        transform=lambda val, sample_interval: val * sample_interval
    ),
    FIELD_BITS['TT']: FieldDef(name='tt', field_type='uint16', scale=TEMP_SCALE),
    FIELD_BITS['CT']: FieldDef(name='ct', field_type='uint16', scale=TEMP_SCALE),
    FIELD_BITS['TP']: FieldDef(name='tp', field_type='uint16', scale=PRESSURE_SCALE),
    FIELD_BITS['CP']: FieldDef(name='cp', field_type='uint16', scale=PRESSURE_SCALE),
    FIELD_BITS['FL']: FieldDef(name='fl', field_type='int16', scale=FLOW_SCALE),
    FIELD_BITS['TF']: FieldDef(name='tf', field_type='int16', scale=FLOW_SCALE),
    FIELD_BITS['PF']: FieldDef(name='pf', field_type='int16', scale=FLOW_SCALE),
    FIELD_BITS['VF']: FieldDef(name='vf', field_type='int16', scale=FLOW_SCALE),
    FIELD_BITS['V']: FieldDef(name='v', field_type='uint16', scale=WEIGHT_SCALE),
    FIELD_BITS['EV']: FieldDef(name='ev', field_type='uint16', scale=WEIGHT_SCALE),
    FIELD_BITS['PR']: FieldDef(name='pr', field_type='uint16', scale=RESISTANCE_SCALE),
    FIELD_BITS['SI']: FieldDef(
        name='systemInfo',
        field_type='uint16',
        transform=lambda val, _: {
            'raw': val,
            'shot_started_volumetric': bool(val & 0x0001),
            'currently_volumetric': bool(val & 0x0002),
            'bluetooth_scale_connected': bool(val & 0x0004),
            'volumetric_available': bool(val & 0x0008),
            'extended_recording': bool(val & 0x0010),
        }
    ),
    # Unit/scale not documented in firmware header; exposed raw.
    FIELD_BITS['WP']: FieldDef(name='wp', field_type='uint16'),
}


def field_defs_for_version(version: int) -> dict[int, FieldDef]:
    """Field definitions for a log version (v6+ widens `t` to uint32 ms)."""
    if version < T_MS_VERSION:
        return FIELD_DEFS
    return {
        **FIELD_DEFS,
        FIELD_BITS['T']: FieldDef(name='t', field_type='uint32'),
    }


@dataclass
class PhaseTransition:
    """Phase transition marker."""
    sample_index: int
    phase_number: int
    phase_name: str


@dataclass
class ShotData:
    """Parsed shot data."""
    id: str
    version: int
    fields_mask: int
    sample_count: int
    sample_interval: int
    profile_id: str
    profile_name: str
    timestamp: int
    rating: int
    duration: int
    weight: Optional[float]
    samples: list[dict] = field(default_factory=list)
    phases: list[PhaseTransition] = field(default_factory=list)
    incomplete: bool = False


def is_html_response(data: bytes) -> bool:
    """Check if data looks like HTML instead of binary shot data.

    Some firmware versions return HTML error pages instead of binary data
    when the device is overloaded or the endpoint is temporarily unavailable.
    """
    if len(data) < 4:
        return False
    prefix = data[:15].lower()
    return prefix.startswith(b'<!doc') or prefix.startswith(b'<html')


def decode_c_string(data: bytes) -> str:
    """Decode null-terminated C string."""
    null_pos = data.find(b'\x00')
    if null_pos == -1:
        return data.decode('utf-8', errors='ignore')
    return data[:null_pos].decode('utf-8', errors='ignore')


def parse_binary_shot(data: bytes, shot_id: str) -> ShotData:
    """Parse binary shot file (.slog format).

    Args:
        data: Binary data from .slog file
        shot_id: Shot identifier

    Returns:
        Parsed shot data

    Raises:
        ValueError: If file format is invalid
    """
    # Check for HTML responses before size validation — the device may return
    # a short HTML error page that would otherwise trigger "Shot file too small"
    if is_html_response(data):
        raise ValueError(
            "Device returned an HTML page instead of binary shot data. "
            "This usually means the device is overloaded or the firmware's "
            "web server is serving its UI instead of the API endpoint."
        )

    if len(data) < HEADER_SIZE_V4:
        raise ValueError(f"Shot file too small: {len(data)} bytes")

    # Parse header
    magic = struct.unpack_from('<I', data, 0)[0]
    if magic != MAGIC:
        raise ValueError(f"Invalid shot magic: 0x{magic:08x} (expected 0x{MAGIC:08x})")

    version = struct.unpack_from('<B', data, 4)[0]
    if version < 1 or version > MAX_SUPPORTED_VERSION:
        raise ValueError(
            f"Unsupported shot log version {version} "
            f"(parser supports 1-{MAX_SUPPORTED_VERSION})"
        )
    header_size = HEADER_SIZE_V5 if version >= 5 else HEADER_SIZE_V4

    if len(data) < header_size:
        raise ValueError(f"Shot file too small for version {version}: {len(data)} bytes")

    # Parse header fields
    sample_interval = struct.unpack_from('<H', data, 8)[0]
    fields_mask = struct.unpack_from('<I', data, 12)[0]
    sample_count = struct.unpack_from('<I', data, 16)[0]
    duration = struct.unpack_from('<I', data, 20)[0]
    timestamp = struct.unpack_from('<I', data, 24)[0]

    profile_id_bytes = data[28:60]
    profile_name_bytes = data[60:108]
    weight_raw = struct.unpack_from('<H', data, 108)[0]

    profile_id = decode_c_string(profile_id_bytes)
    profile_name = decode_c_string(profile_name_bytes)
    weight = weight_raw / WEIGHT_SCALE if weight_raw > 0 else None

    # Parse phase transitions (V5+)
    phases: list[PhaseTransition] = []
    if version >= 5:
        transition_count = struct.unpack_from('<B', data, 458)[0]
        base_offset = 110

        for i in range(min(transition_count, 12)):
            offset = base_offset + i * 29
            sample_index = struct.unpack_from('<H', data, offset)[0]
            phase_number = struct.unpack_from('<B', data, offset + 2)[0]
            phase_name_bytes = data[offset + 4:offset + 29]
            phase_name = decode_c_string(phase_name_bytes)

            phases.append(PhaseTransition(
                sample_index=sample_index,
                phase_number=phase_number,
                phase_name=phase_name
            ))

    # Build field order and per-sample stride from the mask. Widths are
    # version-dependent, so an unknown bit means we cannot know the stride.
    if fields_mask == 0:
        raise ValueError("Invalid shot file: fields_mask is 0 (no fields recorded)")
    unknown_bits = fields_mask & ~KNOWN_FIELD_MASK
    if unknown_bits:
        raise ValueError(
            f"Unknown field bits in fields_mask: 0x{unknown_bits:08x} "
            "(parser cannot determine sample layout)"
        )

    if version < WP_VERSION and fields_mask & (1 << FIELD_BITS['WP']):
        raise ValueError(
            f"Field 'wp' (bit {FIELD_BITS['WP']}) is not valid in shot log version {version}"
        )

    field_defs = field_defs_for_version(version)
    field_order = [bit for bit in range(KNOWN_FIELD_MASK.bit_length()) if fields_mask & (1 << bit)]
    sample_data_size = sum(FIELD_WIDTHS[field_defs[bit].field_type] for bit in field_order)

    # Cross-check the stride against the layout the header declares (v5+:
    # sample size at offset 5, header size at offset 6). Zero means unset.
    if version >= 5:
        declared_sample_size = data[5]
        if declared_sample_size and declared_sample_size != sample_data_size:
            raise ValueError(
                f"Sample size mismatch: header declares {declared_sample_size} bytes, "
                f"fields_mask 0x{fields_mask:x} implies {sample_data_size} (version {version})"
            )
        declared_header_size = struct.unpack_from('<H', data, 6)[0]
        if declared_header_size and declared_header_size != header_size:
            raise ValueError(
                f"Header size mismatch: header declares {declared_header_size} bytes, "
                f"version {version} implies {header_size}"
            )

    # Determine actual samples (handle truncated files)
    available_data = len(data) - header_size
    actual_sample_count = min(sample_count, available_data // sample_data_size)
    incomplete = actual_sample_count < sample_count

    # Parse samples
    samples = []
    for i in range(actual_sample_count):
        sample = {}
        field_offset = header_size + i * sample_data_size

        # Add phase number based on transitions
        for phase in reversed(phases):
            if i >= phase.sample_index:
                sample['phase'] = phase.phase_number
                break

        # Parse each field in this sample
        for field_bit in field_order:
            field_def = field_defs[field_bit]
            value = struct.unpack_from(STRUCT_FORMATS[field_def.field_type], data, field_offset)[0]
            field_offset += FIELD_WIDTHS[field_def.field_type]

            # Apply transform or scale
            if field_def.transform:
                sample[field_def.name] = field_def.transform(value, sample_interval)
            elif field_def.scale:
                sample[field_def.name] = value / field_def.scale
            else:
                sample[field_def.name] = value

        samples.append(sample)

    return ShotData(
        id=shot_id,
        version=version,
        fields_mask=fields_mask,
        sample_count=actual_sample_count,
        sample_interval=sample_interval,
        profile_id=profile_id,
        profile_name=profile_name,
        timestamp=timestamp,
        rating=0,  # Rating not stored in binary file
        duration=duration,
        weight=weight,
        samples=samples,
        phases=phases,
        incomplete=incomplete,
    )
