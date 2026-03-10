import concurrent.futures
import fnmatch
import os
from typing import List

import numpy as np
from PIL import Image, ImageOps
from PySide6 import QtCore, QtWidgets
from PySide6.QtWidgets import QDialog, QVBoxLayout as QVBoxLayoutQt, QLabel, QDialogButtonBox
from qfluentwidgets import (
    FluentIcon as FIF,
    PrimaryPushButton,
    PushSettingCard,
    ConfigItem,
    PushButton,
    InfoBar,
    SwitchSettingCard,
)

from src.utils import dds_utils
from src.utils.cards import TextSettingCard
from src.utils.helpers import BaseWidget
from src.utils.logging_utils import logger


class ExportWorker(QtCore.QThread):
    progress = QtCore.Signal(int, int, str)  # processed, total, message
    finished = QtCore.Signal(str)
    error = QtCore.Signal(str)

    def __init__(
        self,
        src_dir: str,
        out_root: str,
        ignore_patterns: List[str] | None = None,
        fallout76_mode: bool = False,
        fo4_metallic_grayscale: bool = False,
        fo76_diffuse_ao_divide: bool = False,
        fo76_diffuse_r_composite: bool = False,
        fo76_diffuse_debleach_r: bool = False,
        parent=None,
    ):
        super().__init__(parent)
        self.src_dir = src_dir
        self.out_root = out_root
        self.ignore_patterns = [
            p.strip().replace("\\", "/").lstrip("./") for p in (ignore_patterns or []) if p.strip()
        ]
        self.fallout76_mode = bool(fallout76_mode)
        self.fo4_metallic_grayscale = bool(fo4_metallic_grayscale)
        self.fo76_diffuse_ao_divide = bool(fo76_diffuse_ao_divide)
        self.fo76_diffuse_r_composite = bool(fo76_diffuse_r_composite)
        self.fo76_diffuse_debleach_r = bool(fo76_diffuse_debleach_r)
        # Adjusted diffuse is active if any sub-option is enabled
        self.fo76_adjust_diffuse = self.fo76_diffuse_ao_divide or self.fo76_diffuse_r_composite or self.fo76_diffuse_debleach_r
        self._abort = False

    def abort(self):
        self._abort = True

    def _match_ignored(self, rel_dir: str) -> bool:
        if not self.ignore_patterns:
            return False
        norm = (rel_dir or "").replace("\\", "/").strip("/")
        base = os.path.basename(norm)
        for pat in self.ignore_patterns:
            try:
                if fnmatch.fnmatchcase(norm, pat) or fnmatch.fnmatchcase(base, pat):
                    return True
            except Exception:
                pass
        return False

    def _save_png(self, img: Image.Image, out_path: str) -> None:
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        img.save(out_path)

    @staticmethod
    def _sanitize_stem(name: str) -> str:
        """Strip a trailing .png (case-insensitive) from a base name to avoid .png.png outputs."""
        if name.lower().endswith(".png"):
            return name[:-4]
        return name

    def _export_fo4_spec_variants(self, img: Image.Image, base_out_dir: str, stem_no_suffix: str) -> list[str]:
        messages: list[str] = []
        try:
            rgb = img.convert("RGB")
            r, g, _ = rgb.split()

            # Spec/Gloss workflow
            specular = r  # R channel
            glossiness = g  # G channel (do NOT invert)

            # Metal/Rough workflow
            roughness = ImageOps.invert(g)  # invert G
            # Metallic: threshold (default FO4) or grayscale from R based on toggle
            if self.fo4_metallic_grayscale:
                metallic = r  # full-range metallic from R
            else:
                metallic = r.point(lambda p: 255 if p >= 128 else 0)

            out_spec = os.path.join(base_out_dir, f"{stem_no_suffix}_specular.png")
            out_gloss = os.path.join(base_out_dir, f"{stem_no_suffix}_glossiness.png")
            out_rough = os.path.join(base_out_dir, f"{stem_no_suffix}_roughness.png")
            out_metal = os.path.join(base_out_dir, f"{stem_no_suffix}_metallic.png")

            self._save_png(specular, out_spec)
            self._save_png(glossiness, out_gloss)
            self._save_png(roughness, out_rough)
            self._save_png(metallic, out_metal)

            messages.extend(
                [
                    f"Saved {os.path.relpath(out_spec, self.out_root)}",
                    f"Saved {os.path.relpath(out_gloss, self.out_root)}",
                    f"Saved {os.path.relpath(out_rough, self.out_root)}",
                    f"Saved {os.path.relpath(out_metal, self.out_root)}",
                ]
            )
        except Exception as e:
            logger.error(f"Failed creating FO4 spec variants: {e}")
            messages.append(f"ERROR spec variants: {e}")
        return messages

    def _export_fo76_l_variants(self, img: Image.Image, base_out_dir: str, stem_no_suffix: str) -> list[str]:
        """FO76 _l.dds channel breakdown:
          R = smoothness (high = glossy)
          G = ambient occlusion
          B = subsurface scattering (often black, save anyway)
          A = emissive mask (optional, may be absent/white)

        Outputs:
          _glossiness.png  — R channel (smoothness, raw)
          _roughness.png   — R channel inverted (for Metal/Rough workflows)
          _ao.png          — G channel
          _subsurface.png  — B channel
          _emissive.png    — A channel (only saved if it contains non-white data)
        """
        messages: list[str] = []
        try:
            rgba = img.convert("RGBA")
            r, g, b, a = rgba.split()

            glossiness = r
            roughness = ImageOps.invert(r)
            ao = g
            subsurface = b
            emissive = a

            out_gloss = os.path.join(base_out_dir, f"{stem_no_suffix}_glossiness.png")
            out_rough = os.path.join(base_out_dir, f"{stem_no_suffix}_roughness.png")
            out_ao = os.path.join(base_out_dir, f"{stem_no_suffix}_ao.png")
            out_sss = os.path.join(base_out_dir, f"{stem_no_suffix}_subsurface.png")
            out_emis = os.path.join(base_out_dir, f"{stem_no_suffix}_emissive.png")

            self._save_png(glossiness, out_gloss)
            self._save_png(roughness, out_rough)
            self._save_png(ao, out_ao)
            self._save_png(subsurface, out_sss)

            # Only save emissive if it contains actual data (not all-white or all-black)
            a_min, a_max = emissive.getextrema()
            if not (a_min == 255 and a_max == 255) and not (a_min == 0 and a_max == 0):
                self._save_png(emissive, out_emis)
                messages.append(f"Saved {os.path.relpath(out_emis, self.out_root)}")

            messages.extend(
                [
                    f"Saved {os.path.relpath(out_gloss, self.out_root)}",
                    f"Saved {os.path.relpath(out_rough, self.out_root)}",
                    f"Saved {os.path.relpath(out_ao, self.out_root)}",
                    f"Saved {os.path.relpath(out_sss, self.out_root)}",
                ]
            )
        except Exception as e:
            logger.error(f"Failed creating FO76 lighting derivatives: {e}")
            messages.append(f"ERROR FO76 L variants: {e}")
        return messages

    def _export_fo76_spec_from_r(self, img: Image.Image, base_out_dir: str, stem_no_suffix: str) -> list[str]:
        """FO76 _r.dds → specular color PNG.

        The _r texture stores reflectance at normal incidence. It may be a full sRGB color
        texture (BC1_UNORM_SRGB) or a grayscale value (BC4_UNORM). Either way we preserve
        all available RGB data rather than forcing to grayscale, since the specular color
        information is meaningful for the Spec/Gloss workflow.
        """
        messages: list[str] = []
        try:
            # Preserve RGB — do not convert to L, in case it contains colored specular data
            spec_rgb = img.convert("RGB")
            out_spec = os.path.join(base_out_dir, f"{stem_no_suffix}_specular.png")
            self._save_png(spec_rgb, out_spec)
            messages.append(f"Saved {os.path.relpath(out_spec, self.out_root)}")
        except Exception as e:
            logger.error(f"Failed creating FO76 specular from _r: {e}")
            messages.append(f"ERROR FO76 _r → specular: {e}")
        return messages

    @staticmethod
    def _find_sibling_dds(src_path: str, suffix: str) -> str | None:
        """Find a sibling DDS file in the same directory with a different suffix.

        e.g. _find_sibling_dds('/path/to/foo_d.dds', '_l') -> '/path/to/foo_l.dds'
        Case-insensitive scan as a fallback.
        """
        src_dir = os.path.dirname(src_path)
        base = os.path.basename(src_path)
        lower_base = base.lower()

        # Strip the existing two-char suffix (_d, _r, etc.) and .dds
        if len(lower_base) >= 6 and lower_base[-4:] == ".dds" and lower_base[-6] == "_":
            stem = base[:-6]  # preserve original casing of the stem
            target = stem + suffix + ".dds"
        else:
            return None

        # Fast exact and case-variant attempts
        for candidate in (target, target.lower(), target.upper()):
            full = os.path.join(src_dir, candidate)
            if os.path.isfile(full):
                return full

        # Full case-insensitive directory scan as fallback
        target_lower = target.lower()
        try:
            for fn in os.listdir(src_dir):
                if fn.lower() == target_lower:
                    return os.path.join(src_dir, fn)
        except OSError:
            pass
        return None

    @staticmethod
    def _auto_contrast(arr: np.ndarray) -> np.ndarray:
        """Apply auto-contrast (stretch min-max to 0-255) per channel, float32 in/out."""
        out = np.empty_like(arr)
        for c in range(arr.shape[2]):
            ch = arr[:, :, c]
            lo, hi = ch.min(), ch.max()
            if hi > lo:
                out[:, :, c] = (ch - lo) / (hi - lo) * 255.0
            else:
                out[:, :, c] = ch
        return out

    def _export_fo76_adjusted_diffuse(
        self, d_img: Image.Image, src_path: str, base_out_dir: str, stem_no_suffix: str
    ) -> list[str]:
        """Produce an adjusted diffuse from a FO76 _d texture using any combination of:

        1. AO Divide  (fo76_diffuse_ao_divide)
           Divides out the baked AO from the diffuse. FO76 multiplies the _l green channel
           (AO) into the diffuse at export. Reversing it: output = diffuse / AO.
           Requires a matching _l.dds in the same folder.

        2. _r Composite  (fo76_diffuse_r_composite)
           Blends the de-bleached _r texture onto the diffuse using Exclusion blend at 25%
           opacity, masked to highlight areas only (dark pixels erased). This restores
           metallic detail and lightens crushed black metal areas — following the community
           guide approach. Requires a matching _r.dds in the same folder.

        3. De-bleach _r  (fo76_diffuse_debleach_r)
           Exports an auto-contrasted copy of the _r texture as _r_debleached.png.
           This is also used internally by the _r composite step.
           Requires a matching _r.dds in the same folder.

        Steps are applied in order: AO Divide → _r Composite.
        The de-bleached _r is always saved as a separate file when either _r option is on.
        All outputs are saved as _d_adjusted.png (diffuse) and _r_debleached.png (_r).
        """
        messages: list[str] = []
        try:
            base_filename = os.path.basename(src_path)
            # Start with the diffuse as a float32 RGB array for compositing
            result_arr = np.asarray(d_img.convert("RGB"), dtype=np.float32)

            # ── Step 1: AO Divide ────────────────────────────────────────────────────────
            if self.fo76_diffuse_ao_divide:
                l_path = self._find_sibling_dds(src_path, "_l")
                if l_path is None:
                    messages.append(f"SKIP AO divide: no matching _l.dds found for {base_filename}")
                else:
                    l_img = None
                    try:
                        l_img = dds_utils.load_image(l_path, f="RGBA")
                        ao_channel = l_img.split()[1]  # G channel = AO
                        ao_arr = np.asarray(ao_channel, dtype=np.float32)
                        ao_norm = ao_arr[:, :, np.newaxis] / 255.0  # (H, W, 1)

                        with np.errstate(divide="ignore", invalid="ignore"):
                            divided = np.where(ao_norm > 0.0, result_arr / ao_norm, result_arr)
                        result_arr = np.clip(divided, 0.0, 255.0)
                        messages.append("AO divide applied")
                    except Exception as e:
                        logger.error(f"AO divide failed: {e}")
                        messages.append(f"ERROR AO divide: {e}")
                    finally:
                        if l_img is not None:
                            try:
                                l_img.close()
                            except Exception:
                                pass

            # ── Step 2: De-bleach _r + optional composite ────────────────────────────────
            if self.fo76_diffuse_r_composite or self.fo76_diffuse_debleach_r:
                r_path = self._find_sibling_dds(src_path, "_r")
                if r_path is None:
                    messages.append(f"SKIP _r steps: no matching _r.dds found for {base_filename}")
                else:
                    r_img = None
                    try:
                        r_img = dds_utils.load_image(r_path, f="RGB")
                        r_arr = np.asarray(r_img.convert("RGB"), dtype=np.float32)

                        # Auto-contrast the _r (de-bleach): stretch per-channel to full range
                        r_debleached = self._auto_contrast(r_arr)

                        # Always save the de-bleached _r when either _r option is on
                        r_debleach_img = Image.fromarray(
                            np.clip(r_debleached, 0.0, 255.0).astype(np.uint8), mode="RGB"
                        )
                        out_r_debleach = os.path.join(base_out_dir, f"{stem_no_suffix}_r_debleached.png")
                        self._save_png(r_debleach_img, out_r_debleach)
                        messages.append(f"Saved {os.path.relpath(out_r_debleach, self.out_root)}")

                        # ── _r Composite ─────────────────────────────────────────────────
                        if self.fo76_diffuse_r_composite:
                            # Build a highlight mask from the de-bleached _r luminance.
                            # The guide selects bright pixels via Ctrl+click on RGB composite,
                            # then inverts to delete darks. We replicate this as a smooth
                            # luminance mask: mask = lum / 255  (0 = fully transparent, 1 = opaque).
                            # This avoids hard edges while preserving the intent of keeping
                            # only lighter/metallic areas.
                            r_lum = (
                                0.299 * r_debleached[:, :, 0]
                                + 0.587 * r_debleached[:, :, 1]
                                + 0.114 * r_debleached[:, :, 2]
                            )  # (H, W)
                            highlight_mask = (r_lum / 255.0)[:, :, np.newaxis]  # (H, W, 1)

                            # Exclusion blend: A + B - 2*A*B  (values normalised 0..1)
                            base_norm = result_arr / 255.0
                            blend_norm = r_debleached / 255.0
                            exclusion = base_norm + blend_norm - 2.0 * base_norm * blend_norm

                            # Apply at 25% fill, masked to highlights only
                            FILL = 0.25
                            composite = result_arr + (exclusion * 255.0 - result_arr) * highlight_mask * FILL
                            result_arr = np.clip(composite, 0.0, 255.0)
                            messages.append("_r Exclusion composite applied (25% fill, highlight mask)")

                    except Exception as e:
                        logger.error(f"_r steps failed: {e}")
                        messages.append(f"ERROR _r steps: {e}")
                    finally:
                        if r_img is not None:
                            try:
                                r_img.close()
                            except Exception:
                                pass

            # ── Save adjusted diffuse if any step ran ────────────────────────────────────
            if self.fo76_diffuse_ao_divide or self.fo76_diffuse_r_composite:
                adjusted = Image.fromarray(result_arr.astype(np.uint8), mode="RGB")
                out_adj = os.path.join(base_out_dir, f"{stem_no_suffix}_d_adjusted.png")
                self._save_png(adjusted, out_adj)
                messages.append(f"Saved {os.path.relpath(out_adj, self.out_root)}")

        except Exception as e:
            logger.error(f"Failed creating adjusted diffuse: {e}")
            messages.append(f"ERROR adjusted diffuse: {e}")

        return messages

    def run(self):
        try:
            dds_files: list[str] = []
            for root, dirs, files in os.walk(self.src_dir):
                rel_root = os.path.relpath(root, self.src_dir)
                if self._match_ignored(rel_root):
                    dirs[:] = []
                    continue
                dirs[:] = [d for d in dirs if not self._match_ignored(os.path.join(rel_root, d))]
                for fn in files:
                    if fn.lower().endswith(".dds"):
                        dds_files.append(os.path.join(root, fn))

            total = len(dds_files)
            if total == 0:
                self.finished.emit("No DDS files found.")
                return

            processed = 0
            processed_lock = QtCore.QMutex()

            def task(src_path: str) -> str:
                if self._abort:
                    return "aborted"

                rel_path = os.path.relpath(src_path, self.src_dir)
                rel_dir = os.path.dirname(rel_path)
                base_name = os.path.splitext(os.path.basename(src_path))[0]
                out_dir = os.path.join(self.out_root, rel_dir)
                lower = os.path.basename(src_path).lower()
                msg_local = ""

                # ── Base PNG export ──────────────────────────────────────────────────────────
                try:
                    img = dds_utils.load_image(src_path, f="RGBA")

                    safe_base = self._sanitize_stem(base_name)

                    # Normal map blue-channel reconstruction
                    if lower.endswith("_n.dds"):
                        try:
                            r, g, b, a = img.split()
                            b_min, b_max = b.getextrema()
                            # Entirely black B → reconstruct to white (flat normal)
                            if b_min == 0 and b_max == 0:
                                white_b = Image.new("L", b.size, 255)
                                img = Image.merge("RGBA", (r, g, white_b, a))
                            # FO76 BC5_SNORM normals often decode with a uniform grey B channel
                            elif self.fallout76_mode and b_min == b_max and b_max < 255:
                                white_b = Image.new("L", b.size, 255)
                                img = Image.merge("RGBA", (r, g, white_b, a))
                        except Exception:
                            pass  # Non-fatal; fall back to original img

                    out_png = os.path.join(out_dir, safe_base + ".png")
                    self._save_png(img, out_png)
                    msg_local = f"Saved {os.path.relpath(out_png, self.out_root)}"
                except Exception as e:
                    logger.error(f"Failed converting '{src_path}' to PNG: {e}")
                    return f"ERROR {rel_path}: {e}"
                finally:
                    try:
                        img.close()
                    except Exception:
                        pass

                # ── FO4 _s derivatives ───────────────────────────────────────────────────────
                if lower.endswith("_s.dds"):
                    img2 = None
                    try:
                        img2 = dds_utils.load_image(src_path, f="RGBA")
                        core = base_name[:-2] if base_name.lower().endswith("_s") else base_name
                        stem = self._sanitize_stem(core)
                        messages = self._export_fo4_spec_variants(img2, out_dir, stem)
                        if messages:
                            msg_local += " | " + " | ".join(messages)
                    except Exception as e:
                        logger.error(f"Failed FO4 spec variants for '{src_path}': {e}")
                        msg_local += f" | ERROR spec variants: {e}"
                    finally:
                        if img2 is not None:
                            try:
                                img2.close()
                            except Exception:
                                pass

                # ── FO76 derivatives ─────────────────────────────────────────────────────────
                if self.fallout76_mode:
                    if lower.endswith("_l.dds"):
                        img3 = None
                        try:
                            img3 = dds_utils.load_image(src_path, f="RGBA")
                            core = base_name[:-2] if base_name.lower().endswith("_l") else base_name
                            stem = self._sanitize_stem(core)
                            messages = self._export_fo76_l_variants(img3, out_dir, stem)
                            if messages:
                                msg_local += " | " + " | ".join(messages)
                        except Exception as e:
                            logger.error(f"Failed FO76 _l derivatives for '{src_path}': {e}")
                            msg_local += f" | ERROR FO76 _l: {e}"
                        finally:
                            if img3 is not None:
                                try:
                                    img3.close()
                                except Exception:
                                    pass

                    elif lower.endswith("_r.dds"):
                        img4 = None
                        try:
                            # Load as RGB — preserve colour data, do not force grayscale
                            img4 = dds_utils.load_image(src_path, f="RGB")
                            core = base_name[:-2] if base_name.lower().endswith("_r") else base_name
                            stem = self._sanitize_stem(core)
                            messages = self._export_fo76_spec_from_r(img4, out_dir, stem)
                            if messages:
                                msg_local += " | " + " | ".join(messages)
                        except Exception as e:
                            logger.error(f"Failed FO76 _r derivatives for '{src_path}': {e}")
                            msg_local += f" | ERROR FO76 _r: {e}"
                        finally:
                            if img4 is not None:
                                try:
                                    img4.close()
                                except Exception:
                                    pass

                    elif lower.endswith("_d.dds") and self.fo76_adjust_diffuse:
                        img5 = None
                        try:
                            img5 = dds_utils.load_image(src_path, f="RGBA")
                            core = base_name[:-2] if base_name.lower().endswith("_d") else base_name
                            stem = self._sanitize_stem(core)
                            messages = self._export_fo76_adjusted_diffuse(img5, src_path, out_dir, stem)
                            if messages:
                                msg_local += " | " + " | ".join(messages)
                        except Exception as e:
                            logger.error(f"Failed FO76 adjusted diffuse for '{src_path}': {e}")
                            msg_local += f" | ERROR FO76 adj diffuse: {e}"
                        finally:
                            if img5 is not None:
                                try:
                                    img5.close()
                                except Exception:
                                    pass

                return msg_local

            futures = []
            max_workers = max(1, os.cpu_count() or 1)
            with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as ex:
                for src_path in dds_files:
                    futures.append(ex.submit(task, src_path))
                for fut in concurrent.futures.as_completed(futures):
                    if self._abort:
                        break
                    try:
                        message = fut.result()
                    except Exception as e:
                        message = f"ERROR: {e}"
                    processed_lock.lock()
                    processed += 1
                    p = processed
                    processed_lock.unlock()
                    if message == "aborted":
                        self.progress.emit(p, total, "Aborted.")
                    else:
                        self.progress.emit(p, total, message)

            if self._abort:
                self.finished.emit("Aborted by user.")
                return

            self.finished.emit("Done.")
        except Exception as e:
            self.error.emit(str(e))


class DDSPNGExporterWindow(BaseWidget):
    def __init__(self, parent=None, text=None):
        super().__init__(parent=parent, text=text, vertical=True)

        # Persistent settings
        self.src_cfg = ConfigItem("dds_png_exporter", "src", "")
        self.out_cfg = ConfigItem("dds_png_exporter", "out", "")
        self.fo76_cfg = ConfigItem("dds_png_exporter", "fallout76", False)
        self.fo4_metal_gray_cfg = ConfigItem("dds_png_exporter", "fo4_metallic_grayscale", False)
        self.fo76_diffuse_ao_divide_cfg = ConfigItem("dds_png_exporter", "fo76_diffuse_ao_divide", False)
        self.fo76_diffuse_r_composite_cfg = ConfigItem("dds_png_exporter", "fo76_diffuse_r_composite", False)
        self.fo76_diffuse_debleach_r_cfg = ConfigItem("dds_png_exporter", "fo76_diffuse_debleach_r", False)

        # UI cards
        self.src_card = PushSettingCard(
            self.tr("Source folder"),
            FIF.FOLDER,
            self.tr("DDS files Location"),
            self.src_cfg.value or "",
        )
        self.out_card = PushSettingCard(
            self.tr("Output folder"),
            FIF.FOLDER,
            self.tr("PNG Output Location"),
            self.out_cfg.value or "",
        )
        self.fo76_card = SwitchSettingCard(
            icon=FIF.GAME,
            title=self.tr("Fallout 76 mode"),
            configItem=self.fo76_cfg,
            content=self.tr(
                "Use Fallout 76 texture rules (derive roughness/glossiness/AO/emissive from _l, specular from _r)"
            ),
        )
        self.fo4_metal_gray_card = SwitchSettingCard(
            icon=FIF.PALETTE,
            title=self.tr("Fallout 4: Grayscale Metallic"),
            configItem=self.fo4_metal_gray_cfg,
            content=self.tr(
                "When processing _s.dds: use full-range metallic from spec luminance instead of binary threshold"
            ),
        )
        # ── FO76 Diffuse adjustment sub-options ──────────────────────────────────────────
        self.fo76_diffuse_ao_divide_card = SwitchSettingCard(
            icon=FIF.BRIGHTNESS,
            title=self.tr("FO76 Diffuse: AO Divide"),
            configItem=self.fo76_diffuse_ao_divide_cfg,
            content=self.tr(
                "Divide out the baked AO from the _d texture to recover a cleaner base diffuse. "
                "Requires a matching _l.dds. Outputs _d_adjusted.png."
            ),
        )
        self.fo76_diffuse_r_composite_card = SwitchSettingCard(
            icon=FIF.BRUSH,
            title=self.tr("FO76 Diffuse: _r Composite"),
            configItem=self.fo76_diffuse_r_composite_cfg,
            content=self.tr(
                "Blend the de-bleached _r onto the diffuse using Exclusion at 25% fill, masked to highlights only. "
                "Restores metallic detail to crushed black areas. "
                "Requires a matching _r.dds. Outputs _d_adjusted.png and _r_debleached.png."
            ),
        )
        self.fo76_diffuse_debleach_r_card = SwitchSettingCard(
            icon=FIF.SCROLL,
            title=self.tr("FO76 Diffuse: De-bleach _r"),
            configItem=self.fo76_diffuse_debleach_r_cfg,
            content=self.tr(
                "Export an auto-contrasted copy of the _r texture as _r_debleached.png. "
                "Compresses the tonal range so it doesn't make the entire surface uniformly metallic when used as a spec map. "
                "Requires a matching _r.dds."
            ),
        )
        self.ignore_cfg = ConfigItem("dds_png_exporter", "ignore", "")
        self.ignore_card = TextSettingCard(
            self.ignore_cfg,
            FIF.FILTER,
            self.tr("Ignore subfolders (comma separated, supports wildcards)"),
            self.tr("e.g., temp, *_bak, */Generated/*"),
        )

        self.addToFrame(self.src_card)
        self.addToFrame(self.out_card)
        self.addToFrame(self.fo76_card)
        self.addToFrame(self.fo76_diffuse_ao_divide_card)
        self.addToFrame(self.fo76_diffuse_r_composite_card)
        self.addToFrame(self.fo76_diffuse_debleach_r_card)
        self.addToFrame(self.fo4_metal_gray_card)
        self.addToFrame(self.ignore_card)
        self.boxLayout.addStretch(1)

        self.run_button = PrimaryPushButton(icon=FIF.RIGHT_ARROW, text=self.tr("Run"))
        self.stop_button = PushButton(icon=FIF.CANCEL, text=self.tr("Stop"))
        self.stop_button.setEnabled(False)

        self.src_card.clicked.connect(self.on_input_dir_card)
        self.out_card.clicked.connect(self.on_output_root_card)
        self.run_button.clicked.connect(self.on_run)
        self.stop_button.clicked.connect(self.stop)

        self.buttons_layout.addWidget(self.stop_button, stretch=1)
        self.addButtonBarToBottom(self.run_button)

        # Disable FO76 sub-options when FO76 mode is off
        self.fo76_cfg.valueChanged.connect(self._on_fo76_toggle)
        self._on_fo76_toggle(self.fo76_cfg.value)

    def _on_fo76_toggle(self, enabled: bool):
        """Enable/disable FO76-specific options based on the FO76 mode switch."""
        for card in (
            self.fo76_diffuse_ao_divide_card,
            self.fo76_diffuse_r_composite_card,
            self.fo76_diffuse_debleach_r_card,
        ):
            card.setEnabled(bool(enabled))

    def on_input_dir_card(self):
        directory = QtWidgets.QFileDialog.getExistingDirectory(
            self, self.tr("Select source folder"), self.src_cfg.value or os.getcwd()
        )
        if directory:
            self.src_cfg.value = directory
            self.src_card.setContent(directory)
            if not (self.out_cfg.value or "").strip():
                base = os.path.basename(os.path.normpath(directory))
                suggested = os.path.join(os.path.dirname(directory), base + "_png")
                self.out_cfg.value = suggested
                self.out_card.setContent(suggested)

    def on_output_root_card(self):
        directory = QtWidgets.QFileDialog.getExistingDirectory(
            self, self.tr("Select output folder"), self.out_cfg.value or os.getcwd()
        )
        if directory:
            self.out_cfg.value = directory
            self.out_card.setContent(directory)

    def set_running(self, running: bool):
        self.run_button.setEnabled(not running)
        self.stop_button.setEnabled(running)
        self.src_card.setEnabled(not running)
        self.out_card.setEnabled(not running)
        self.ignore_card.setEnabled(not running)
        self.fo76_card.setEnabled(not running)
        self.fo4_metal_gray_card.setEnabled(not running)
        self.fo76_diffuse_ao_divide_card.setEnabled(not running)
        self.fo76_diffuse_r_composite_card.setEnabled(not running)
        self.fo76_diffuse_debleach_r_card.setEnabled(not running)

    def on_run(self):
        src = (self.src_cfg.value or "").strip()
        out = (self.out_cfg.value or "").strip()
        ignore_csv = (self.ignore_cfg.value or "").strip()
        fo76 = bool(self.fo76_cfg.value)
        fo4_metal_gray = bool(self.fo4_metal_gray_cfg.value)
        fo76_ao_divide = bool(self.fo76_diffuse_ao_divide_cfg.value)
        fo76_r_composite = bool(self.fo76_diffuse_r_composite_cfg.value)
        fo76_debleach_r = bool(self.fo76_diffuse_debleach_r_cfg.value)

        if not src or not os.path.isdir(src):
            InfoBar.warning(
                title=self.tr("Validation"),
                content=self.tr("Please select a valid source folder."),
                duration=3000,
                parent=self,
            )
            return
        if not out:
            InfoBar.warning(
                title=self.tr("Validation"),
                content=self.tr("Please choose an output folder."),
                duration=3000,
                parent=self,
            )
            return
        try:
            os.makedirs(out, exist_ok=True)
        except Exception:
            pass

        ignore_patterns = [p.strip() for p in ignore_csv.split(",") if p.strip()] if ignore_csv else []

        p = getattr(self, "parent", None)
        if p and hasattr(p, "show_progress"):
            try:
                p.show_progress()
            except Exception:
                pass

        self.set_running(True)
        self.worker = ExportWorker(
            src,
            out,
            ignore_patterns,
            fallout76_mode=fo76,
            fo4_metallic_grayscale=fo4_metal_gray,
            fo76_diffuse_ao_divide=fo76_ao_divide,
            fo76_diffuse_r_composite=fo76_r_composite,
            fo76_diffuse_debleach_r=fo76_debleach_r,
        )
        self.worker.progress.connect(self.on_progress)
        self.worker.finished.connect(self.on_finished)
        self.worker.error.connect(self.on_error)
        self.worker.start()

    def stop(self):
        if hasattr(self, "worker") and self.worker.isRunning():
            self.worker.abort()

    @QtCore.Slot(int, int, str)
    def on_progress(self, processed: int, total: int, message: str):
        p = getattr(self, "parent", None)
        if p and hasattr(p, "update_progress") and total:
            try:
                percent = int(max(0, min(100, round((processed / total) * 100))))
                p.update_progress(percent)
            except Exception:
                pass
        logger.debug(message)

    @QtCore.Slot(str)
    def on_finished(self, message: str):
        logger.debug(message)
        self.set_running(False)
        p = getattr(self, "parent", None)
        if p and hasattr(p, "complete_loader"):
            try:
                p.complete_loader()
            except Exception:
                pass
        dlg = QDialog(self)
        dlg.setWindowTitle(self.tr("Export Completed"))
        layout = QVBoxLayoutQt(dlg)
        lbl = QLabel(message)
        lbl.setWordWrap(True)
        layout.addWidget(lbl)
        buttons = QDialogButtonBox(QDialogButtonBox.Ok)
        layout.addWidget(buttons)
        buttons.accepted.connect(dlg.accept)
        dlg.exec()

    @QtCore.Slot(str)
    def on_error(self, message: str):
        logger.debug("ERROR: " + message)
        self.set_running(False)
        p = getattr(self, "parent", None)
        if p and hasattr(p, "complete_loader"):
            try:
                p.complete_loader()
            except Exception:
                pass
        dlg = QDialog(self)
        dlg.setWindowTitle(self.tr("Export Error"))
        layout = QVBoxLayoutQt(dlg)
        lbl = QLabel(self.tr("An error occurred: ") + message)
        lbl.setWordWrap(True)
        layout.addWidget(lbl)
        buttons = QDialogButtonBox(QDialogButtonBox.Ok)
        layout.addWidget(buttons)
        buttons.accepted.connect(dlg.accept)
        dlg.exec()