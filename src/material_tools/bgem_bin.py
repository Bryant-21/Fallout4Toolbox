from __future__ import annotations
import io
from dataclasses import dataclass
from .base import (
    BaseHeader,
    read_bool,
    read_f32,
    read_string,
    write_bool,
    write_f32,
    write_string,
)

BGEM_SIGNATURE = 0x4D454742  # 'BGEM'


@dataclass
class BGEMData:
    header: BaseHeader
    BaseTexture: str
    GrayscaleTexture: str
    EnvmapTexture: str
    NormalTexture: str
    EnvmapMaskTexture: str
    SpecularTexture: str | None
    LightingTexture: str | None
    GlowTexture: str | None
    GlassRoughnessScratch: str | None
    GlassDirtOverlay: str | None
    GlassEnabled: bool | None
    GlassFresnelColor: tuple[float, float, float] | None
    GlassBlurScaleBase: float | None
    GlassBlurScaleFactor: float | None
    GlassRefractionScaleBase: float | None
    # version >= 10
    EnvironmentMapping: bool | None
    EnvironmentMappingMaskScale: float | None
    # flags and floats
    BloodEnabled: bool
    EffectLightingEnabled: bool
    FalloffEnabled: bool
    FalloffColorEnabled: bool
    GrayscaleToPaletteAlpha: bool
    SoftEnabled: bool
    BaseColor: tuple[float, float, float]
    BaseColorScale: float
    FalloffStartAngle: float
    FalloffStopAngle: float
    FalloffStartOpacity: float
    FalloffStopOpacity: float
    LightingInfluence: float
    EnvmapMinLOD: int
    SoftDepth: float
    EmittanceColor: tuple[float, float, float] | None
    AdaptativeEmissive_ExposureOffset: float | None
    AdaptativeEmissive_FinalExposureMin: float | None
    AdaptativeEmissive_FinalExposureMax: float | None
    Glowmap: bool | None
    EffectPbrSpecular: bool | None

    def write(self, bw: io.BufferedWriter) -> None:
        from .base import write_color3
        self.header.write(bw)
        write_string(bw, self.BaseTexture)
        write_string(bw, self.GrayscaleTexture)
        write_string(bw, self.EnvmapTexture)
        write_string(bw, self.NormalTexture)
        write_string(bw, self.EnvmapMaskTexture)
        if self.header.version >= 11:
            write_string(bw, self.SpecularTexture)
            write_string(bw, self.LightingTexture)
            write_string(bw, self.GlowTexture)
        if self.header.version >= 21:
            write_string(bw, self.GlassRoughnessScratch)
            write_string(bw, self.GlassDirtOverlay)
            write_bool(bw, bool(self.GlassEnabled))
            if self.GlassEnabled:
                write_color3(bw, self.GlassFresnelColor or (1.0, 1.0, 1.0))
                # Note: order matches C# with FIXME
                write_f32(bw, float(self.GlassBlurScaleBase or 0.0))
                if self.header.version >= 22:
                    write_f32(bw, float(self.GlassBlurScaleFactor or 0.0))
                write_f32(bw, float(self.GlassRefractionScaleBase or 0.0))
        if self.header.version >= 10:
            write_bool(bw, bool(self.EnvironmentMapping))
            write_f32(bw, float(self.EnvironmentMappingMaskScale or 0.0))
        write_bool(bw, self.BloodEnabled)
        write_bool(bw, self.EffectLightingEnabled)
        write_bool(bw, self.FalloffEnabled)
        write_bool(bw, self.FalloffColorEnabled)
        write_bool(bw, self.GrayscaleToPaletteAlpha)
        write_bool(bw, self.SoftEnabled)
        write_color3(bw, self.BaseColor)
        write_f32(bw, self.BaseColorScale)
        write_f32(bw, self.FalloffStartAngle)
        write_f32(bw, self.FalloffStopAngle)
        write_f32(bw, self.FalloffStartOpacity)
        write_f32(bw, self.FalloffStopOpacity)
        write_f32(bw, self.LightingInfluence)
        from .base import write_u8
        write_u8(bw, int(self.EnvmapMinLOD))
        write_f32(bw, self.SoftDepth)
        if self.header.version >= 11:
            from .base import write_color3
            write_color3(bw, self.EmittanceColor or (1.0, 1.0, 1.0))
        if self.header.version >= 15:
            write_f32(bw, float(self.AdaptativeEmissive_ExposureOffset or 0.0))
            write_f32(bw, float(self.AdaptativeEmissive_FinalExposureMin or 0.0))
            write_f32(bw, float(self.AdaptativeEmissive_FinalExposureMax or 0.0))
        if self.header.version >= 16:
            write_bool(bw, bool(self.Glowmap))
        if self.header.version >= 20:
            write_bool(bw, bool(self.EffectPbrSpecular))


def read_bgem(br: io.BufferedReader) -> BGEMData:
    from .base import read_color3, read_u8
    header = BaseHeader.read(br, BGEM_SIGNATURE)
    BaseTexture = read_string(br)
    GrayscaleTexture = read_string(br)
    EnvmapTexture = read_string(br)
    NormalTexture = read_string(br)
    EnvmapMaskTexture = read_string(br)

    SpecularTexture = LightingTexture = GlowTexture = None
    if header.version >= 11:
        SpecularTexture = read_string(br)
        LightingTexture = read_string(br)
        GlowTexture = read_string(br)

    GlassRoughnessScratch = GlassDirtOverlay = None
    GlassEnabled = None
    GlassFresnelColor = None
    GlassBlurScaleBase = GlassBlurScaleFactor = GlassRefractionScaleBase = None
    if header.version >= 21:
        GlassRoughnessScratch = read_string(br)
        GlassDirtOverlay = read_string(br)
        GlassEnabled = read_bool(br)
        if GlassEnabled:
            GlassFresnelColor = read_color3(br)
            GlassBlurScaleBase = read_f32(br)
            if header.version >= 22:
                GlassBlurScaleFactor = read_f32(br)
            GlassRefractionScaleBase = read_f32(br)

    EnvironmentMapping = EnvironmentMappingMaskScale = None
    if header.version >= 10:
        EnvironmentMapping = read_bool(br)
        EnvironmentMappingMaskScale = read_f32(br)

    BloodEnabled = read_bool(br)
    EffectLightingEnabled = read_bool(br)
    FalloffEnabled = read_bool(br)
    FalloffColorEnabled = read_bool(br)
    GrayscaleToPaletteAlpha = read_bool(br)
    SoftEnabled = read_bool(br)

    BaseColor = read_color3(br)
    BaseColorScale = read_f32(br)
    FalloffStartAngle = read_f32(br)
    FalloffStopAngle = read_f32(br)
    FalloffStartOpacity = read_f32(br)
    FalloffStopOpacity = read_f32(br)

    LightingInfluence = read_f32(br)
    EnvmapMinLOD = read_u8(br)
    SoftDepth = read_f32(br)

    EmittanceColor = None
    if header.version >= 11:
        EmittanceColor = read_color3(br)

    AdaptativeEmissive_ExposureOffset = AdaptativeEmissive_FinalExposureMin = AdaptativeEmissive_FinalExposureMax = None
    if header.version >= 15:
        AdaptativeEmissive_ExposureOffset = read_f32(br)
        AdaptativeEmissive_FinalExposureMin = read_f32(br)
        AdaptativeEmissive_FinalExposureMax = read_f32(br)

    Glowmap = None
    if header.version >= 16:
        Glowmap = read_bool(br)

    EffectPbrSpecular = None
    if header.version >= 20:
        EffectPbrSpecular = read_bool(br)

    return BGEMData(
        header,
        BaseTexture,
        GrayscaleTexture,
        EnvmapTexture,
        NormalTexture,
        EnvmapMaskTexture,
        SpecularTexture,
        LightingTexture,
        GlowTexture,
        GlassRoughnessScratch,
        GlassDirtOverlay,
        GlassEnabled,
        GlassFresnelColor,
        GlassBlurScaleBase,
        GlassBlurScaleFactor,
        GlassRefractionScaleBase,
        EnvironmentMapping,
        EnvironmentMappingMaskScale,
        BloodEnabled,
        EffectLightingEnabled,
        FalloffEnabled,
        FalloffColorEnabled,
        GrayscaleToPaletteAlpha,
        SoftEnabled,
        BaseColor,
        BaseColorScale,
        FalloffStartAngle,
        FalloffStopAngle,
        FalloffStartOpacity,
        FalloffStopOpacity,
        LightingInfluence,
        EnvmapMinLOD,
        SoftDepth,
        EmittanceColor,
        AdaptativeEmissive_ExposureOffset,
        AdaptativeEmissive_FinalExposureMin,
        AdaptativeEmissive_FinalExposureMax,
        Glowmap,
        EffectPbrSpecular,
    )
