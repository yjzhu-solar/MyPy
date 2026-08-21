import numpy as np
import matplotlib.text as mtext


class CurvedText(mtext.Text):
    """
    Text following an arbitrary 2-D curve.

    Parameters
    ----------
    x, y : array-like
        Coordinates describing the curve in data coordinates.

    text : str
        Text to draw along the curve.

    axes : matplotlib.axes.Axes
        Axes on which the text is drawn.

    start_offset : float, default: 0
        Distance from the beginning of the curve to the start of the
        text, in points.

    offset : float, default: 0
        Perpendicular displacement of the text from the curve,
        in points. Positive values move to the left-hand side of the
        curve direction.

    character_spacing : float, default: 0
        Extra spacing between characters, in points.

    smooth_rotation : float, default: 0
        Length scale used to estimate the local curve tangent, in points.
        If zero, a value based on the character width is used.

    upright : bool, default: False
        If True, prevent characters from appearing upside down by
        rotating them by 180 degrees where necessary.

    **kwargs
        Standard matplotlib.text.Text properties.

    Notes
    -----
    Character placement and rotation are calculated in display
    coordinates. This automatically handles unequal x/y scales,
    logarithmic axes, resized figures, etc.
    """

    def __init__(
        self,
        x,
        y,
        text,
        axes,
        *,
        start_offset=0.0,
        offset=0.0,
        character_spacing=0.0,
        smooth_rotation=0.0,
        upright=False,
        **kwargs,
    ):
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)

        if x.ndim != 1 or y.ndim != 1:
            raise ValueError("x and y must be one-dimensional.")
        if len(x) != len(y):
            raise ValueError("x and y must have the same length.")
        if len(x) < 2:
            raise ValueError("The curve must contain at least two points.")
        if not (np.all(np.isfinite(x)) and np.all(np.isfinite(y))):
            raise ValueError("x and y must contain only finite values.")

        self._curve_x = x
        self._curve_y = y

        self.start_offset = start_offset
        self.offset = offset
        self.character_spacing = character_spacing
        self.smooth_rotation = smooth_rotation
        self.upright = upright

        self._characters = []

        # The parent Text object is only a controller.
        # Its own text is never actually drawn.
        super().__init__(x[0], y[0], text, **kwargs)

        axes.add_artist(self)
        self._create_characters()

    # ------------------------------------------------------------------
    # Character management
    # ------------------------------------------------------------------

    def _create_characters(self):
        """Create one Text artist for each character."""
        self._remove_characters()

        for char in self.get_text():
            t = mtext.Text(
                0.0,
                0.0,
                char,
                horizontalalignment="center",
                verticalalignment="center",
                rotation_mode="anchor",
            )

            self.axes.add_artist(t)
            self._characters.append(t)

        self._sync_character_properties()

    def _remove_characters(self):
        for t in getattr(self, "_characters", []):
            try:
                t.remove()
            except ValueError:
                pass

        self._characters = []

    def _sync_character_properties(self):
        """Copy relevant Text properties from controller to characters."""
        if not hasattr(self, "_characters"):
            return

        for t in self._characters:
            t.set_fontproperties(self.get_fontproperties().copy())
            t.set_color(self.get_color())
            t.set_alpha(self.get_alpha())
            t.set_path_effects(self.get_path_effects())

            t.set_clip_on(self.get_clip_on())
            t.set_visible(self.get_visible())

            # Slightly larger zorder ensures that update_positions()
            # runs before the characters themselves are drawn.
            t.set_zorder(self.get_zorder() + 1e-3)

    # ------------------------------------------------------------------
    # Text API
    # ------------------------------------------------------------------

    def set_text(self, s):
        changed = str(s) != getattr(self, "_text", "")

        result = super().set_text(s)

        if changed and hasattr(self, "_characters") and self.axes is not None:
            self._create_characters()

        return result

    def set_zorder(self, zorder):
        super().set_zorder(zorder)

        if hasattr(self, "_characters"):
            for t in self._characters:
                t.set_zorder(zorder + 1e-3)

    def set_visible(self, visible):
        super().set_visible(visible)

        if hasattr(self, "_characters"):
            for t in self._characters:
                t.set_visible(visible)

    def remove(self):
        self._remove_characters()
        return super().remove()

    # ------------------------------------------------------------------
    # Geometry
    # ------------------------------------------------------------------

    @staticmethod
    def _interp_curve(s, xy, positions):
        """
        Interpolate positions along a polyline parameterized by
        cumulative arc length.
        """
        x = np.interp(positions, s, xy[:, 0])
        y = np.interp(positions, s, xy[:, 1])
        return np.column_stack((x, y))

    def update_positions(self, renderer):
        """
        Update character positions and rotations.
        """

        self._sync_character_properties()

        if not self.get_visible() or not self._characters:
            return

        ax = self.axes

        # --------------------------------------------------------------
        # Transform curve from data coordinates -> display coordinates.
        # Everything below is therefore measured in pixels.
        # --------------------------------------------------------------

        curve_data = np.column_stack(
            (self._curve_x, self._curve_y)
        )

        curve_disp = ax.transData.transform(curve_data)

        dxy = np.diff(curve_disp, axis=0)
        ds = np.hypot(dxy[:, 0], dxy[:, 1])

        # Remove duplicate / zero-length curve segments.
        keep = np.r_[True, ds > 1e-12]

        curve_disp = curve_disp[keep]

        if len(curve_disp) < 2:
            for t in self._characters:
                t.set_visible(False)
            return

        dxy = np.diff(curve_disp, axis=0)
        ds = np.hypot(dxy[:, 0], dxy[:, 1])

        s = np.r_[0.0, np.cumsum(ds)]
        total_length = s[-1]

        # Points -> pixels.
        points_to_pixels = self.figure.dpi / 72.0

        cursor = self.start_offset * points_to_pixels
        normal_offset = self.offset * points_to_pixels
        extra_spacing = self.character_spacing * points_to_pixels

        # --------------------------------------------------------------
        # Character placement.
        # --------------------------------------------------------------

        for t in self._characters:

            # Measure the actual character.
            #
            # Matplotlib correctly gives a width for whitespace too,
            # so there is no need for the old invisible-'a' trick.
            t.set_rotation(0.0)

            bbox = t.get_window_extent(renderer=renderer)
            width = bbox.width

            center_s = cursor + width / 2.0

            # Character doesn't fit.
            if center_s + width / 2.0 > total_length:
                t.set_visible(False)
                cursor += width + extra_spacing
                continue

            t.set_visible(self.get_visible())

            # ----------------------------------------------------------
            # Position on the curve.
            # ----------------------------------------------------------

            p = self._interp_curve(
                s,
                curve_disp,
                np.array([center_s]),
            )[0]

            # ----------------------------------------------------------
            # Estimate tangent using points to either side.
            #
            # This gives smoother rotations than simply taking the
            # direction of one polyline segment.
            # ----------------------------------------------------------

            if self.smooth_rotation > 0:
                delta = self.smooth_rotation * points_to_pixels
            else:
                delta = max(1.0, width * 0.25)

            s0 = max(0.0, center_s - delta)
            s1 = min(total_length, center_s + delta)

            p0, p1 = self._interp_curve(
                s,
                curve_disp,
                np.array([s0, s1]),
            )

            tangent = p1 - p0

            norm = np.hypot(tangent[0], tangent[1])

            if norm == 0:
                cursor += width + extra_spacing
                continue

            tangent /= norm

            angle = np.degrees(
                np.arctan2(tangent[1], tangent[0])
            )

            # ----------------------------------------------------------
            # Optional perpendicular displacement.
            # ----------------------------------------------------------

            normal = np.array([
                -tangent[1],
                tangent[0],
            ])

            p = p + normal_offset * normal

            # ----------------------------------------------------------
            # Optionally keep characters upright.
            # ----------------------------------------------------------

            if self.upright:
                if angle > 90:
                    angle -= 180
                elif angle < -90:
                    angle += 180

            # Convert display position back into data coordinates.
            xy_data = ax.transData.inverted().transform(p)

            t.set_position(xy_data)
            t.set_rotation(angle)

            # Alignment controls which side of the path the letters use.
            t.set_horizontalalignment("center")
            t.set_verticalalignment(self.get_verticalalignment())
            t.set_rotation_mode("anchor")

            cursor += width + extra_spacing

    # ------------------------------------------------------------------

    def draw(self, renderer):
        """
        Update character geometry.

        The parent Text object itself is intentionally not drawn.
        """
        self.update_positions(renderer)
        self.stale = False