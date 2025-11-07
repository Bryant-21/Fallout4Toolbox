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

BGSM_SIGNATURE = 0x4D534742  # 'BGSM'


@dataclass
class BGSMData:
    header: BaseHeader
    # strings
    DiffuseTexture: str
    NormalTexture: str
    SmoothSpecTexture: str
    GreyscaleTexture: str
    EnvmapTexture: str | None
    GlowTexture: str | None
    InnerLayerTexture: str | None
    WrinklesTexture: str | None
    DisplacementTexture: str | None
    SpecularTexture: str | None
    LightingTexture: str | None
    FlowTexture: str | None
    DistanceFieldAlphaTexture: str | None
    # rest of fields (subset sufficient for round-trip)
    EnableEditorAlphaRef: bool
    # v>=8 block or v<8 block
    RimLighting: bool | None
    RimPower: float | None
    BackLightPower: float | None
    SubsurfaceLighting: bool | None
    SubsurfaceLightingRolloff: float | None
    Translucency: bool | None
    TranslucencyThickObject: bool | None
    TranslucencyMixAlbedoWithSubsurfaceColor: bool | None
    TranslucencySubsurfaceColor: tuple[float, float, float] | None
    TranslucencyTransmissiveScale: float | None
    TranslucencyTurbulence: float | None
    # spec / smoothness etc.
    SpecularEnabled: bool
    SpecularColor: tuple[float, float, float]
    SpecularMult: float
    Smoothness: float
    FresnelPower: float
    WetnessControlSpecScale: float
    WetnessControlSpecPowerScale: float
    WetnessControlSpecMinvar: float
    WetnessControlEnvMapScale: float | None
    WetnessControlFresnelPower: float
    WetnessControlMetalness: float
    # PBR and porosity
    PBR: bool | None
    CustomPorosity: bool | None
    PorosityValue: float | None
    RootMaterialPath: str
    AnisoLighting: bool
    EmitEnabled: bool
    EmittanceColor: tuple[float, float, float] | None
    EmittanceMult: float
    ModelSpaceNormals: bool
    ExternalEmittance: bool
    LumEmittance: float | None
    UseAdaptativeEmissive: bool | None
    AdaptativeEmissive_ExposureOffset: float | None
    AdaptativeEmissive_FinalExposureMin: float | None
    AdaptativeEmissive_FinalExposureMax: float | None
    BackLighting: bool | None
    ReceiveShadows: bool
    HideSecret: bool
    CastShadows: bool
    DissolveFade: bool
    AssumeShadowmask: bool
    Glowmap: bool
    EnvironmentMappingWindow: bool | None
    EnvironmentMappingEye: bool | None
    Hair: bool
    HairTintColor: tuple[float, float, float]
    Tree: bool
    Facegen: bool
    SkinTint: bool
    Tessellate: bool
    DisplacementTextureBias: float | None
    DisplacementTextureScale: float | None
    TessellationPnScale: float | None
    TessellationBaseFactor: float | None
    TessellationFadeDistance: float | None
    GrayscaleToPaletteScale: float
    SkewSpecularAlpha: bool | None
    Terrain: bool | None
    UnkInt1: int | None
    TerrainThresholdFalloff: float | None
    TerrainTilingDistance: float | None
    TerrainRotationAngle: float | None

    def write(self, bw: io.BufferedWriter) -> None:
        self.header.write(bw)
        write_string(bw, self.DiffuseTexture)
        write_string(bw, self.NormalTexture)
        write_string(bw, self.SmoothSpecTexture)
        write_string(bw, self.GreyscaleTexture)
        if self.header.version > 2:
            write_string(bw, self.GlowTexture)
            write_string(bw, self.WrinklesTexture)
            write_string(bw, self.SpecularTexture)
            write_string(bw, self.LightingTexture)
            write_string(bw, self.FlowTexture)
            if self.header.version >= 17:
                write_string(bw, self.DistanceFieldAlphaTexture)
        else:
            write_string(bw, self.EnvmapTexture)
            write_string(bw, self.GlowTexture)
            write_string(bw, self.InnerLayerTexture)
            write_string(bw, self.WrinklesTexture)
            write_string(bw, self.DisplacementTexture)

        write_bool(bw, self.EnableEditorAlphaRef)
        if self.header.version >= 8:
            write_bool(bw, bool(self.Translucency))
            write_bool(bw, bool(self.TranslucencyThickObject))
            write_bool(bw, bool(self.TranslucencyMixAlbedoWithSubsurfaceColor))
            # color
            from .base import write_color3
            write_color3(bw, self.TranslucencySubsurfaceColor or (1.0, 1.0, 1.0))
            write_f32(bw, float(self.TranslucencyTransmissiveScale or 0.0))
            write_f32(bw, float(self.TranslucencyTurbulence or 0.0))
        else:
            write_bool(bw, bool(self.RimLighting))
            write_f32(bw, float(self.RimPower or 0.0))
            write_f32(bw, float(self.BackLightPower or 0.0))
            write_bool(bw, bool(self.SubsurfaceLighting))
            write_f32(bw, float(self.SubsurfaceLightingRolloff or 0.0))

        write_bool(bw, self.SpecularEnabled)
        from .base import write_color3
        write_color3(bw, self.SpecularColor)
        write_f32(bw, self.SpecularMult)
        write_f32(bw, self.Smoothness)

        write_f32(bw, self.FresnelPower)
        write_f32(bw, self.WetnessControlSpecScale)
        write_f32(bw, self.WetnessControlSpecPowerScale)
        write_f32(bw, self.WetnessControlSpecMinvar)
        if self.header.version < 10:
            write_f32(bw, float(self.WetnessControlEnvMapScale or 0.0))
        write_f32(bw, self.WetnessControlFresnelPower)
        write_f32(bw, self.WetnessControlMetalness)
        if self.header.version > 2:
            write_bool(bw, bool(self.PBR))
            if self.header.version >= 9:
                write_bool(bw, bool(self.CustomPorosity))
                write_f32(bw, float(self.PorosityValue or 0.0))
        write_string(bw, self.RootMaterialPath)
        write_bool(bw, self.AnisoLighting)
        write_bool(bw, self.EmitEnabled)
        if self.EmitEnabled:
            write_color3(bw, self.EmittanceColor or (1.0, 1.0, 1.0))
        write_f32(bw, self.EmittanceMult)
        write_bool(bw, self.ModelSpaceNormals)
        write_bool(bw, self.ExternalEmittance)
        if self.header.version >= 12:
            write_f32(bw, float(self.LumEmittance or 0.0))
        if self.header.version >= 13:
            write_bool(bw, bool(self.UseAdaptativeEmissive))
            write_f32(bw, float(self.AdaptativeEmissive_ExposureOffset or 0.0))
            write_f32(bw, float(self.AdaptativeEmissive_FinalExposureMin or 0.0))
            write_f32(bw, float(self.AdaptativeEmissive_FinalExposureMax or 0.0))
        if self.header.version < 8:
            write_bool(bw, bool(self.BackLighting))
        write_bool(bw, self.ReceiveShadows)
        write_bool(bw, self.HideSecret)
        write_bool(bw, self.CastShadows)
        write_bool(bw, self.DissolveFade)
        write_bool(bw, self.AssumeShadowmask)
        write_bool(bw, self.Glowmap)
        if self.header.version < 7:
            write_bool(bw, bool(self.EnvironmentMappingWindow))
            write_bool(bw, bool(self.EnvironmentMappingEye))
        write_bool(bw, self.Hair)
        write_color3(bw, self.HairTintColor)
        write_bool(bw, self.Tree)
        write_bool(bw, self.Facegen)
        write_bool(bw, self.SkinTint)
        write_bool(bw, self.Tessellate)
        if self.header.version < 3:
            write_f32(bw, float(self.DisplacementTextureBias or 0.0))
            write_f32(bw, float(self.DisplacementTextureScale or 0.0))
            write_f32(bw, float(self.TessellationPnScale or 0.0))
            write_f32(bw, float(self.TessellationBaseFactor or 0.0))
            write_f32(bw, float(self.TessellationFadeDistance or 0.0))
        write_f32(bw, self.GrayscaleToPaletteScale)
        if self.header.version >= 1:
            write_bool(bw, bool(self.SkewSpecularAlpha))
        if self.header.version >= 3:
            write_bool(bw, bool(self.Terrain))
            if self.Terrain:
                if self.header.version == 3:
                    from .base import write_u32
                    write_u32(bw, int(self.UnkInt1 or 0))
                write_f32(bw, float(self.TerrainThresholdFalloff or 0.0))
                write_f32(bw, float(self.TerrainTilingDistance or 0.0))
                write_f32(bw, float(self.TerrainRotationAngle or 0.0))


def read_bgsm(br: io.BufferedReader) -> BGSMData:
    from .base import read_color3, read_u32
    header = BaseHeader.read(br, BGSM_SIGNATURE)
    DiffuseTexture = read_string(br)
    NormalTexture = read_string(br)
    SmoothSpecTexture = read_string(br)
    GreyscaleTexture = read_string(br)
    EnvmapTexture = GlowTexture = InnerLayerTexture = WrinklesTexture = DisplacementTexture = None
    SpecularTexture = LightingTexture = FlowTexture = DistanceFieldAlphaTexture = None
    if header.version > 2:
        GlowTexture = read_string(br)
        WrinklesTexture = read_string(br)
        SpecularTexture = read_string(br)
        LightingTexture = read_string(br)
        FlowTexture = read_string(br)
        if header.version >= 17:
            DistanceFieldAlphaTexture = read_string(br)
    else:
        EnvmapTexture = read_string(br)
        GlowTexture = read_string(br)
        InnerLayerTexture = read_string(br)
        WrinklesTexture = read_string(br)
        DisplacementTexture = read_string(br)

    EnableEditorAlphaRef = read_bool(br)

    Translucency = TranslucencyThickObject = TranslucencyMixAlbedoWithSubsurfaceColor = None
    TranslucencySubsurfaceColor = None
    TranslucencyTransmissiveScale = TranslucencyTurbulence = None
    RimLighting = RimPower = BackLightPower = SubsurfaceLighting = SubsurfaceLightingRolloff = None
    if header.version >= 8:
        Translucency = read_bool(br)
        TranslucencyThickObject = read_bool(br)
        TranslucencyMixAlbedoWithSubsurfaceColor = read_bool(br)
        TranslucencySubsurfaceColor = read_color3(br)
        TranslucencyTransmissiveScale = read_f32(br)
        TranslucencyTurbulence = read_f32(br)
    else:
        RimLighting = read_bool(br)
        RimPower = read_f32(br)
        BackLightPower = read_f32(br)
        SubsurfaceLighting = read_bool(br)
        SubsurfaceLightingRolloff = read_f32(br)

    SpecularEnabled = read_bool(br)
    SpecularColor = read_color3(br)
    SpecularMult = read_f32(br)
    Smoothness = read_f32(br)

    FresnelPower = read_f32(br)
    WetnessControlSpecScale = read_f32(br)
    WetnessControlSpecPowerScale = read_f32(br)
    WetnessControlSpecMinvar = read_f32(br)
    WetnessControlEnvMapScale = None
    if header.version < 10:
        WetnessControlEnvMapScale = read_f32(br)
    WetnessControlFresnelPower = read_f32(br)
    WetnessControlMetalness = read_f32(br)

    PBR = CustomPorosity = None
    PorosityValue = None
    if header.version > 2:
        PBR = read_bool(br)
        if header.version >= 9:
            CustomPorosity = read_bool(br)
            PorosityValue = read_f32(br)

    RootMaterialPath = read_string(br)
    AnisoLighting = read_bool(br)
    EmitEnabled = read_bool(br)
    EmittanceColor = None
    if EmitEnabled:
        EmittanceColor = read_color3(br)
    EmittanceMult = read_f32(br)
    ModelSpaceNormals = read_bool(br)
    ExternalEmittance = read_bool(br)

    LumEmittance = None
    if header.version >= 12:
        LumEmittance = read_f32(br)

    UseAdaptativeEmissive = None
    AdaptativeEmissive_ExposureOffset = None
    AdaptativeEmissive_FinalExposureMin = None
    AdaptativeEmissive_FinalExposureMax = None
    if header.version >= 13:
        UseAdaptativeEmissive = read_bool(br)
        AdaptativeEmissive_ExposureOffset = read_f32(br)
        AdaptativeEmissive_FinalExposureMin = read_f32(br)
        AdaptativeEmissive_FinalExposureMax = read_f32(br)

    BackLighting = None
    if header.version < 8:
        BackLighting = read_bool(br)

    ReceiveShadows = read_bool(br)
    HideSecret = read_bool(br)
    CastShadows = read_bool(br)
    DissolveFade = read_bool(br)
    AssumeShadowmask = read_bool(br)
    Glowmap = read_bool(br)

    EnvironmentMappingWindow = EnvironmentMappingEye = None
    if header.version < 7:
        EnvironmentMappingWindow = read_bool(br)
        EnvironmentMappingEye = read_bool(br)

    Hair = read_bool(br)
    HairTintColor = read_color3(br)

    Tree = read_bool(br)
    Facegen = read_bool(br)
    SkinTint = read_bool(br)
    Tessellate = read_bool(br)

    DisplacementTextureBias = DisplacementTextureScale = TessellationPnScale = TessellationBaseFactor = TessellationFadeDistance = None
    if header.version < 3:
        DisplacementTextureBias = read_f32(br)
        DisplacementTextureScale = read_f32(br)
        TessellationPnScale = read_f32(br)
        TessellationBaseFactor = read_f32(br)
        TessellationFadeDistance = read_f32(br)

    GrayscaleToPaletteScale = read_f32(br)

    SkewSpecularAlpha = None
    if header.version >= 1:
        SkewSpecularAlpha = read_bool(br)

    Terrain = None
    UnkInt1 = None
    TerrainThresholdFalloff = TerrainTilingDistance = TerrainRotationAngle = None
    if header.version >= 3:
        Terrain = read_bool(br)
        if Terrain:
            if header.version == 3:
                UnkInt1 = read_u32(br)
            TerrainThresholdFalloff = read_f32(br)
            TerrainTilingDistance = read_f32(br)
            TerrainRotationAngle = read_f32(br)

    return BGSMData(
        header,
        DiffuseTexture,
        NormalTexture,
        SmoothSpecTexture,
        GreyscaleTexture,
        EnvmapTexture,
        GlowTexture,
        InnerLayerTexture,
        WrinklesTexture,
        DisplacementTexture,
        SpecularTexture,
        LightingTexture,
        FlowTexture,
        DistanceFieldAlphaTexture,
        EnableEditorAlphaRef,
        RimLighting,
        RimPower,
        BackLightPower,
        SubsurfaceLighting,
        SubsurfaceLightingRolloff,
        Translucency,
        TranslucencyThickObject,
        TranslucencyMixAlbedoWithSubsurfaceColor,
        TranslucencySubsurfaceColor,
        TranslucencyTransmissiveScale,
        TranslucencyTurbulence,
        SpecularEnabled,
        SpecularColor,
        SpecularMult,
        Smoothness,
        FresnelPower,
        WetnessControlSpecScale,
        WetnessControlSpecPowerScale,
        WetnessControlSpecMinvar,
        WetnessControlEnvMapScale,
        WetnessControlFresnelPower,
        WetnessControlMetalness,
        PBR,
        CustomPorosity,
        PorosityValue,
        RootMaterialPath,
        AnisoLighting,
        EmitEnabled,
        EmittanceColor,
        EmittanceMult,
        ModelSpaceNormals,
        ExternalEmittance,
        LumEmittance,
        UseAdaptativeEmissive,
        AdaptativeEmissive_ExposureOffset,
        AdaptativeEmissive_FinalExposureMin,
        AdaptativeEmissive_FinalExposureMax,
        BackLighting,
        ReceiveShadows,
        HideSecret,
        CastShadows,
        DissolveFade,
        AssumeShadowmask,
        Glowmap,
        EnvironmentMappingWindow,
        EnvironmentMappingEye,
        Hair,
        HairTintColor,
        Tree,
        Facegen,
        SkinTint,
        Tessellate,
        DisplacementTextureBias,
        DisplacementTextureScale,
        TessellationPnScale,
        TessellationBaseFactor,
        TessellationFadeDistance,
        GrayscaleToPaletteScale,
        SkewSpecularAlpha,
        Terrain,
        UnkInt1,
        TerrainThresholdFalloff,
        TerrainTilingDistance,
        TerrainRotationAngle,
    )
