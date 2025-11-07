from __future__ import annotations
import io
import struct
from dataclasses import dataclass

LE = "<"  # little-endian


def read_u32(br: io.BufferedReader) -> int:
    return struct.unpack(LE + "I", br.read(4))[0]


def read_u8(br: io.BufferedReader) -> int:
    return struct.unpack(LE + "B", br.read(1))[0]


def read_f32(br: io.BufferedReader) -> float:
    return struct.unpack(LE + "f", br.read(4))[0]


def read_bool(br: io.BufferedReader) -> bool:
    return struct.unpack(LE + "?", br.read(1))[0]


def write_u32(bw: io.BufferedWriter, val: int) -> None:
    bw.write(struct.pack(LE + "I", val))


def write_u8(bw: io.BufferedWriter, val: int) -> None:
    bw.write(struct.pack(LE + "B", val))


def write_f32(bw: io.BufferedWriter, val: float) -> None:
    bw.write(struct.pack(LE + "f", val))


def write_bool(bw: io.BufferedWriter, val: bool) -> None:
    bw.write(struct.pack(LE + "?", val))


def read_color3(br: io.BufferedReader) -> tuple[float, float, float]:
    return read_f32(br), read_f32(br), read_f32(br)


def write_color3(bw: io.BufferedWriter, c: tuple[float, float, float]) -> None:
    r, g, b = c
    write_f32(bw, r)
    write_f32(bw, g)
    write_f32(bw, b)


def color3_to_u32(c: tuple[float, float, float]) -> int:
    r, g, b = c
    ri = max(0, min(255, int(round(r * 255.0))))
    gi = max(0, min(255, int(round(g * 255.0))))
    bi = max(0, min(255, int(round(b * 255.0))))
    return (ri << 16) | (gi << 8) | bi | (255 << 24)


def u32_to_color3(argb: int) -> tuple[float, float, float]:
    r = ((argb >> 16) & 0xFF) / 255.0
    g = ((argb >> 8) & 0xFF) / 255.0
    b = (argb & 0xFF) / 255.0
    return (r, g, b)


def read_string(br: io.BufferedReader) -> str:
    length = read_u32(br)
    if length == 0:
        return ""
    data = br.read(length)
    return data.decode("utf-8", errors="strict")


def write_string(bw: io.BufferedWriter, s: str | None) -> None:
    if not s:
        write_u32(bw, 0)
        return
    b = s.encode("utf-8")
    write_u32(bw, len(b))
    bw.write(b)


@dataclass
class BaseHeader:
    signature: int
    version: int
    tile_u: bool
    tile_v: bool
    u_offset: float
    v_offset: float
    u_scale: float
    v_scale: float
    alpha: float
    alpha_blend_mode0: int
    alpha_blend_mode1: int
    alpha_blend_mode2: int
    alpha_test_ref: int
    alpha_test: bool
    zbuffer_write: bool
    zbuffer_test: bool
    ssr: bool
    wet_ssr: bool
    decal: bool
    two_sided: bool
    decal_nofade: bool
    non_occluder: bool
    refraction: bool
    refraction_falloff: bool
    refraction_power: float
    # Either env mapping + mask scale (v<10) or depth bias (v>=10)
    env_mapping: bool | None
    env_mapping_mask_scale: float | None
    depth_bias: bool | None
    grayscale_to_palette_color: bool
    mask_writes: int | None

    @staticmethod
    def read(br: io.BufferedReader, expected_signature: int) -> "BaseHeader":
        sig = read_u32(br)
        if sig != expected_signature:
            raise ValueError("Invalid signature: 0x%08X" % sig)
        version = read_u32(br)
        tile_flags = read_u32(br)
        tile_u = (tile_flags & 2) != 0
        tile_v = (tile_flags & 1) != 0
        u_offset = read_f32(br)
        v_offset = read_f32(br)
        u_scale = read_f32(br)
        v_scale = read_f32(br)
        alpha = read_f32(br)
        ab0 = read_u8(br)
        ab1 = read_u32(br)
        ab2 = read_u32(br)
        alpha_test_ref = read_u8(br)
        alpha_test = read_bool(br)
        zbuffer_write = read_bool(br)
        zbuffer_test = read_bool(br)
        ssr = read_bool(br)
        wet_ssr = read_bool(br)
        decal = read_bool(br)
        two_sided = read_bool(br)
        decal_nofade = read_bool(br)
        non_occluder = read_bool(br)
        refraction = read_bool(br)
        refraction_falloff = read_bool(br)
        refraction_power = read_f32(br)
        env_mapping = None
        env_mapping_mask_scale = None
        depth_bias = None
        if version < 10:
            env_mapping = read_bool(br)
            env_mapping_mask_scale = read_f32(br)
        else:
            depth_bias = read_bool(br)
        grayscale_to_palette_color = read_bool(br)
        mask_writes = None
        if version >= 6:
            mask_writes = read_u8(br)
        return BaseHeader(sig, version, tile_u, tile_v, u_offset, v_offset, u_scale, v_scale, alpha,
                          ab0, ab1, ab2, alpha_test_ref, alpha_test, zbuffer_write, zbuffer_test, ssr, wet_ssr,
                          decal, two_sided, decal_nofade, non_occluder, refraction, refraction_falloff,
                          refraction_power, env_mapping, env_mapping_mask_scale, depth_bias,
                          grayscale_to_palette_color, mask_writes)

    def write(self, bw: io.BufferedWriter) -> None:
        write_u32(bw, self.signature)
        write_u32(bw, self.version)
        tile_flags = (2 if self.tile_u else 0) | (1 if self.tile_v else 0)
        write_u32(bw, tile_flags)
        write_f32(bw, self.u_offset)
        write_f32(bw, self.v_offset)
        write_f32(bw, self.u_scale)
        write_f32(bw, self.v_scale)
        write_f32(bw, self.alpha)
        write_u8(bw, self.alpha_blend_mode0)
        write_u32(bw, self.alpha_blend_mode1)
        write_u32(bw, self.alpha_blend_mode2)
        write_u8(bw, self.alpha_test_ref)
        write_bool(bw, self.alpha_test)
        write_bool(bw, self.zbuffer_write)
        write_bool(bw, self.zbuffer_test)
        write_bool(bw, self.ssr)
        write_bool(bw, self.wet_ssr)
        write_bool(bw, self.decal)
        write_bool(bw, self.two_sided)
        write_bool(bw, self.decal_nofade)
        write_bool(bw, self.non_occluder)
        write_bool(bw, self.refraction)
        write_bool(bw, self.refraction_falloff)
        write_f32(bw, self.refraction_power)
        if self.version < 10:
            write_bool(bw, bool(self.env_mapping))
            write_f32(bw, float(self.env_mapping_mask_scale or 0.0))
        else:
            write_bool(bw, bool(self.depth_bias))
        write_bool(bw, self.grayscale_to_palette_color)
        if self.version >= 6:
            write_u8(bw, int(self.mask_writes or 0))
