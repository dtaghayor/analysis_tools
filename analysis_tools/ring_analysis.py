"""
ring_analysis.py
================

Standardised, class-based tools for the *tank-side* (mPMT) part of a pion /
single-ring analysis: turn raw hit information (mPMT slot, PMT position, charge,
calibrated time) into per-hit angles relative to the beam, an optional time
residual (hit time minus time-of-flight from the beam entry point), and finally
a split of the deposited charge into the part **inside** the Cherenkov cone and
the part **outside** it.

The design mirrors the rest of ``analysis_tools``:

* :class:`RingGeometry` is the geometry / timing. The per-channel angle ``theta`` and time-of-flight
  offset ``tof`` depend only on the PMT position, the beam entry point and the
  beam direction -- by default all fixed -- so they are **precomputed once** per channel in
  the constructor and looked up per hit, but per event beam entry point and the
  beam direction can be provided too which makes the anaylsis much slower but comaptible with T5 information
  and secondary rings.
* :class:`CherenkovRingSelection` is the hit-level analogue of
  :class:`analysis_tools.beam_selection.BeamSelection`: it holds the cut values
  and produces per-event quantities. Single events and whole batches go through
  the *same* vectorised core (:meth:`CherenkovRingSelection._summarise`), so
  there is only one place to change the physics.
* :class:`RingResults` is a small container for the per-event arrays (and,
  optionally, the per-event jagged ``theta`` / ``delta_t``), with a
  ``concatenate`` helper so you can accumulate results across loader batches.
* :func:`classify_charge_topology` and :func:`plot_inside_vs_outside` are the
  common plotting / categorisation helpers used after the event loop.

Optional time cut
-----------------
The prompt-time selection is optional (``apply_time_cut``). When **on** (default)
each event's charge is restricted to hits in a prompt window ``|delta_t - mu| < K``,
where ``mu`` is found per event from the peak of the time-residual histogram.
When **off**, no coarse time window, no time-of-flight subtraction and no prompt
fit are applied -- the charge is split into inside/outside the cone using all
hits passed in. Turn it off when the timing has already been handled upstream
(e.g. by a T5 beam-monitor entry-time tool) and you pass already time-selected
hits.

Geometry source
---------------
PMT positions can come from any source that yields an ``(n_mpmt, n_pmt, 3)``
array of absolute positions (NaN for missing channels):

* Dean's ``Geometry`` package (design or survey placements) --
  :meth:`RingGeometry.from_geometry_package` / :func:`load_geometry_package_positions`,
* a raw positions array -- ``RingGeometry(positions=arr)``,
* the in-repo :class:`DetectorGeometry` -- ``RingGeometry(geometry=DetectorGeometry())``
  or simply ``RingGeometry()`` (used as a fallback when nothing else is given).

Cone apex / origin
------------------
``theta`` and the time-of-flight are measured from a cone apex (the beam entry
point). It defaults to the fixed ``origin`` and is precomputed per channel for
speed. A per-event apex can be supplied at call time via ``origins`` (e.g. the
particle entry position from T5); that path recomputes angles per hit (still
vectorised) instead of using the per-channel tables.
"""

from dataclasses import dataclass, field
from typing import Optional, Sequence, Tuple, Dict, Any

import numpy as np

try:
    import matplotlib.pyplot as plt
    _MPL_AVAILABLE = True
except ImportError:
    _MPL_AVAILABLE = False

try:
    from scipy.optimize import curve_fit
    _SCIPY_AVAILABLE = True
except ImportError:
    _SCIPY_AVAILABLE = False

try:
    from .detector_geometry import DetectorGeometry
    _HAS_DETECTOR_GEOMETRY = True
except Exception:
    DetectorGeometry = None
    _HAS_DETECTOR_GEOMETRY = False


# Group velocity of light in water, mm/ns.
# This is the value used in the original pion-scattering notebook; it is slightly
# smaller than DetectorGeometry.calc_tof's c/n (which uses n = 1.33) because it is
# a group velocity rather than a phase velocity. Override via RingGeometry(vg=...).
DEFAULT_VG = 2.20027795333758801e8 * 1000 / 1e9   # ~220.03 mm/ns

# Default beam entry point in WCTE tank coordinates (mm) and beam direction.
# These MUST be checked against the geometry frame you are using

DEFAULT_ORIGIN = (0.0, 0.0, -1520.0 + 188.0)   # = (0, 0, -1332)
DEFAULT_BEAM_DIRECTION = (0.0, 0.0, 1.0)
DEFAULT_PHI_REFERENCE = (1.0, 0.0, 0.0)

# Per-event status codes
_OK, _NO_HITS, _NO_GEOM, _FEW_HITS, _NO_TIME = 0, 1, 2, 3, 4
_STATUS_NAME = {_OK: "ok", _NO_HITS: "no_hits", _NO_GEOM: "no_geom",
                _FEW_HITS: "few_hits", _NO_TIME: "no_time"}

# Refractive index of water for the Cherenkov angle (phase index, ~1.33).
DEFAULT_N_INDEX = 1.33


def _unit(vec):
    """Normalise a (3,) vector or each row of an (N,3) array."""
    vec = np.asarray(vec, dtype=float)
    if vec.ndim == 1:
        return vec / np.linalg.norm(vec)
    return vec / np.linalg.norm(vec, axis=1, keepdims=True)


# ----------------------------------------------------------------------------
# Cherenkov angle from particle momentum
# ----------------------------------------------------------------------------
# Local particle masses in MeV/c^2; the table from beam_selection is reused when
# importable so there is a single source of truth.
_LOCAL_MASSES_MEV = {
    "electron": 0.511, "muon": 105.66, "pion": 139.57, "kaon": 493.68,
    "proton": 938.27, "deuteron": 1876.54, "helium3": 2808.39,
}
_PARTICLE_ALIASES = {
    "e": "electron", "e+": "electron", "e-": "electron", "positron": "electron",
    "mu": "muon", "mu+": "muon", "mu-": "muon", "muon": "muon",
    "pi": "pion", "pi+": "pion", "pi-": "pion", "pion": "pion",
    "k": "kaon", "k+": "kaon", "k-": "kaon", "kaon": "kaon",
    "p": "proton", "proton": "proton",
    "d": "deuteron", "deuteron": "deuteron",
    "he3": "helium3", "helium3": "helium3",
}


def _particle_masses_mev():
    """Particle masses (MeV); reuses beam_selection's table when available."""
    masses = dict(_LOCAL_MASSES_MEV)
    try:
        from .beam_selection import _PARTICLE_MASSES
        masses.update(_PARTICLE_MASSES)
    except Exception:
        pass
    return masses


def particle_mass_mev(particle):
    """Resolve a particle name (with common aliases) to a mass in MeV/c^2."""
    key = _PARTICLE_ALIASES.get(str(particle).lower(), str(particle).lower())
    masses = _particle_masses_mev()
    if key not in masses:
        raise KeyError(f"Unknown particle {particle!r}; known: {sorted(masses)}")
    return masses[key]


def cherenkov_angle_deg(momentum, particle="pion", n=DEFAULT_N_INDEX, mass=None):
    """
    Cherenkov angle (degrees) for a particle of given momentum in a medium of
    refractive index ``n``: ``cos(theta_c) = 1 / (n * beta)`` with
    ``beta = p / sqrt(p^2 + m^2)``.

    Parameters
    ----------
    momentum : float or array
        Momentum in MeV/c (scalar nominal value, or per-event array).
    particle : str
        Particle name/alias (e.g. "pion", "pi+", "muon", "proton"); ignored if
        ``mass`` is given.
    n : float
        Refractive index of the medium (water ~1.33).
    mass : float, optional
        Particle mass in MeV/c^2; overrides ``particle``.

    Returns
    -------
    float or ndarray
        Cherenkov angle in degrees. NaN where the particle is below Cherenkov
        threshold (``n * beta <= 1``), i.e. no ring is produced.
    """
    scalar = np.ndim(momentum) == 0
    p = np.asarray(momentum, dtype=float)
    if mass is None:
        mass = particle_mass_mev(particle)
    E = np.sqrt(p ** 2 + mass ** 2)
    beta = np.divide(p, E, out=np.zeros_like(E), where=E > 0)
    nbeta = n * beta
    with np.errstate(invalid="ignore", divide="ignore"):
        cos_c = np.where(nbeta > 0, 1.0 / nbeta, np.inf)
    theta = np.where(cos_c < 1.0, np.degrees(np.arccos(np.clip(cos_c, -1.0, 1.0))), np.nan)
    return float(theta) if scalar else theta


def cherenkov_cone_halfangle(momentum, particle="pion", n=DEFAULT_N_INDEX,
                             margin_deg=0.0, mass=None):
    """
    Convenience: Cherenkov angle plus a margin, for use as the cone half-angle
    (``angle_cut_deg``). Returns NaN where below threshold. ``margin_deg`` widens
    the cone to catch resolution/scattering (e.g. a few degrees).
    """
    return cherenkov_angle_deg(momentum, particle=particle, n=n, mass=mass) + margin_deg


# ----------------------------------------------------------------------------
# Beam configuration
# ----------------------------------------------------------------------------
@dataclass
class BeamGeometry:
    """
    Bundled beam (or secondary-track) configuration, kept separate from the
    detector geometry: where the Cherenkov cone apex sits, which way it points,
    and optionally the particle/momentum used to derive the cone angle.

    Pass it to :class:`RingGeometry` (``beam=...``) to set the apex/axis, and use
    :meth:`cone_halfangle` to get the ``angle_cut_deg`` for
    :class:`CherenkovRingSelection`.

    Attributes
    ----------
    origin : (3,) cone apex / entry point (mm), in the geometry frame.
    direction : (3,) cone axis (need not be normalised).
    momentum : float, optional
        Momentum in MeV/c (e.g. the estimated momentum at the tank).
    particle : str, optional
        Particle name/alias (e.g. "pion") used for the mass.
    n : float
        Refractive index for the Cherenkov angle (water ~1.33).
    """
    origin: Any = DEFAULT_ORIGIN
    direction: Any = DEFAULT_BEAM_DIRECTION
    momentum: Optional[float] = None
    particle: Optional[str] = None
    n: float = DEFAULT_N_INDEX

    def __post_init__(self):
        self.origin = np.asarray(self.origin, dtype=float)
        self.direction = np.asarray(self.direction, dtype=float)

    @property
    def unit_direction(self):
        """The cone axis, normalised."""
        return self.direction / np.linalg.norm(self.direction)

    def cherenkov_angle_deg(self, momentum=None, particle=None, n=None):
        """
        Cherenkov angle (degrees) for this configuration; NaN below threshold.
        Falls back to the stored ``momentum`` / ``particle`` / ``n`` when the
        arguments are not supplied.
        """
        p = self.momentum if momentum is None else momentum
        part = self.particle if particle is None else particle
        nn = self.n if n is None else n
        if p is None or part is None:
            raise ValueError(
                "BeamGeometry needs momentum and particle to compute the "
                "Cherenkov angle (set them on the BeamGeometry or pass them in).")
        return cherenkov_angle_deg(p, particle=part, n=nn)

    def cone_halfangle(self, margin_deg=0.0, momentum=None, particle=None, n=None):
        """Cherenkov angle + ``margin_deg`` (degrees), for use as angle_cut_deg."""
        return self.cherenkov_angle_deg(momentum=momentum, particle=particle, n=n) + margin_deg


# ----------------------------------------------------------------------------
# Geometry / timing backbone
# ----------------------------------------------------------------------------
def load_geometry_package_positions(geo_file, wcd_index=0, place_info="est",
                                    n_pos_per_slot=19, geo_package_path=None):
    """
    Build an ``(n_mpmt, n_pmt, 3)`` array of absolute PMT positions from the
    external ``Geometry`` package (the one used in the original notebook).

    Parameters
    ----------
    geo_file : str
        Path to the ``.geo`` file, e.g. ``.../examples/wcte_bldg157.geo``.
    wcd_index : int
        Which water Cherenkov detector in the hall to use (0 for WCTE).
    place_info : str
        Placement set passed to ``pmt.get_placement(place_info=...)``. The
        original notebook used ``"est"`` (survey/estimated). Pass the value your
        Geometry package exposes for design positions (e.g. ``"design"``) to use
        those instead.
    n_pos_per_slot : int
        Number of PMT positions per mPMT (19 for WCTE).
    geo_package_path : str, optional
        Directory that *contains* the ``Geometry`` package folder. If given it
        is prepended to ``sys.path`` before importing, so you can point at a
        checkout (e.g. ``.../SWAN_projects/Geometry``) without installing it or
        editing this module. This keeps the personal path in your notebook call
        rather than hardcoded in the library.

    Returns
    -------
    positions : (n_mpmt, n_pmt, 3) ndarray
        Absolute positions in mm; NaN where a placement is missing.

    Notes
    -----
    Imports ``Geometry`` lazily, so this module does not require that package
    unless this function is called.
    """
    if geo_package_path is not None:
        import sys
        if geo_package_path not in sys.path:
            sys.path.insert(0, geo_package_path)

    try:
        from Geometry.Device import Device  # lazy import; external package
    except ModuleNotFoundError as e:
        raise ModuleNotFoundError(
            "Could not import the 'Geometry' package. Make sure it is importable "
            "in this kernel: pass geo_package_path=<dir containing the Geometry/ "
            "folder>, or add it to PYTHONPATH / install it, then restart the "
            "kernel. (`import Geometry` should succeed on its own first.)"
        ) from e

    hall = Device.open_file(geo_file)
    wcd = hall.wcds[wcd_index]
    n_slots = len(wcd.mpmts)

    positions = np.full((n_slots, n_pos_per_slot, 3), np.nan)
    for slot in range(n_slots):
        for pos in range(n_pos_per_slot):
            try:
                placement = wcd.mpmts[slot].pmts[pos].get_placement(place_info=place_info)
                if placement is not None and "location" in placement:
                    positions[slot, pos] = placement["location"]
            except (KeyError, ValueError, AttributeError, IndexError):
                continue
    return positions


class RingGeometry:
    """
    Per-channel angles relative to the beam and time-of-flight offsets.

    For a fixed cone apex (``origin``) the per-channel ``theta``, ``phi``,
    distance ``r`` and time-of-flight ``tof`` are precomputed once in the
    constructor and looked up per hit. A per-event apex can instead be passed to
    :meth:`lookup` / :meth:`compute` at call time.

    Parameters
    ----------
    positions : (n_mpmt, n_pmt, 3) or (n_mpmt*n_pmt, 3) array, optional
        Absolute PMT positions (mm), NaN for missing channels. This is the
        preferred input -- e.g. from :func:`load_geometry_package_positions`.
    geometry : object, optional
        Any object exposing ``mpmts_pos`` of shape ``(n_mpmt, n_pmt, 3)``
        (e.g. :class:`DetectorGeometry`). Used if ``positions`` is not given.
    beam : BeamGeometry, optional
        Bundled beam configuration. If given, its ``origin`` and ``direction``
        set the cone apex/axis (taking precedence over the ``origin`` /
        ``beam_direction`` arguments), and it is kept on ``self.beam``.
    origin : sequence of 3 floats
        Default cone apex / beam entry point, in the same frame as ``positions``.
    beam_direction, phi_reference : sequence of 3 floats
        Beam axis and transverse reference for ``theta`` / ``phi``.
    vg : float
        Group velocity of light (mm/ns) for the time-of-flight offset.
    tol : float
        Numerical tolerance for normalising near-zero vectors.
    n_pos_per_slot : int
        Used only to reshape a flat ``(n_mpmt*n_pmt, 3)`` positions array.

    If neither ``positions`` nor ``geometry`` is given, falls back to
    :class:`DetectorGeometry` (when available).
    """

    def __init__(self,
                 positions=None,
                 *,
                 geometry=None,
                 beam: "Optional[BeamGeometry]" = None,
                 origin: Sequence[float] = DEFAULT_ORIGIN,
                 beam_direction: Sequence[float] = DEFAULT_BEAM_DIRECTION,
                 phi_reference: Sequence[float] = DEFAULT_PHI_REFERENCE,
                 vg: float = DEFAULT_VG,
                 tol: float = 1e-8,
                 n_pos_per_slot: int = 19):
        # Resolve the positions table from whichever source was provided.
        if positions is not None:
            pos = np.asarray(positions, dtype=float)
            if pos.ndim == 2:
                pos = pos.reshape(-1, n_pos_per_slot, 3)
            self.channel_pos = pos
        elif geometry is not None:
            self.channel_pos = np.asarray(geometry.mpmts_pos, dtype=float)
        else:
            if not _HAS_DETECTOR_GEOMETRY:
                raise ValueError(
                    "No positions/geometry given and DetectorGeometry is not "
                    "available. Pass positions=... (e.g. from "
                    "load_geometry_package_positions) or geometry=...")
            self.channel_pos = np.asarray(DetectorGeometry().mpmts_pos, dtype=float)

        self.n_mpmt, self.n_pmt = self.channel_pos.shape[0], self.channel_pos.shape[1]
        self.channel_valid = np.isfinite(self.channel_pos).all(axis=2)

        # A BeamGeometry, if given, supplies the cone apex and axis.
        self.beam = beam
        if beam is not None:
            origin = beam.origin
            beam_direction = beam.direction

        self.origin = np.asarray(origin, dtype=float)
        bd = np.asarray(beam_direction, dtype=float)
        self.beam_direction = bd / np.linalg.norm(bd)
        phi_ref = np.asarray(phi_reference, dtype=float)
        self.phi_reference = phi_ref / np.linalg.norm(phi_ref)
        self.vg = float(vg)
        self.tol = float(tol)

        self._build_channel_tables()

    # -- construction from the external Geometry package -------------------
    @classmethod
    def from_geometry_package(cls, geo_file, *, wcd_index=0, place_info="est",
                              n_pos_per_slot=19, geo_package_path=None, **kwargs):
        """
        Build a RingGeometry from a ``.geo`` file via the external Geometry
        package. ``place_info`` selects survey ("est", the original default) vs
        design placements. ``geo_package_path`` (dir containing the ``Geometry``
        package) is prepended to ``sys.path`` if given. Extra kwargs (origin,
        beam_direction, vg, ...) are forwarded to ``__init__``.
        """
        positions = load_geometry_package_positions(
            geo_file, wcd_index=wcd_index, place_info=place_info,
            n_pos_per_slot=n_pos_per_slot, geo_package_path=geo_package_path)
        return cls(positions=positions, n_pos_per_slot=n_pos_per_slot, **kwargs)

    # -- one-time per-channel precompute (fixed origin) --------------------
    def _build_channel_tables(self):
        """Compute theta/phi/r/tof for every channel once, for the fixed origin."""
        flat = self.channel_pos.reshape(-1, 3)
        theta, phi, r = self._angles_core(flat, self.origin)
        tof = r / self.vg
        shape = (self.n_mpmt, self.n_pmt)
        self.channel_theta = theta.reshape(shape)
        self.channel_phi = phi.reshape(shape)
        self.channel_r = r.reshape(shape)
        self.channel_tof = tof.reshape(shape)

    def _angles_core(self, positions, origin, beam_direction=None, phi_reference=None):
        """
        Vectorised theta/phi/r for (N,3) positions about ``origin`` with cone
        axis ``beam_direction``. ``origin`` and ``beam_direction`` may each be
        (3,) (one value for all hits) or (N,3) (per-hit). ``theta`` is the angle
        from the cone axis; only ``theta`` is used by the inside/outside cut, so
        ``phi`` (measured from ``phi_reference``) is informational.
        """
        bd = self.beam_direction if beam_direction is None else _unit(beam_direction)
        pr = self.phi_reference if phi_reference is None else _unit(phi_reference)

        v = positions - origin
        r = np.linalg.norm(v, axis=1)
        inv_r = np.where(r > self.tol, 1.0 / np.where(r > self.tol, r, 1.0), 0.0)
        u = v * inv_r[:, None]

        def _dot(a, b):                       # a is (N,3); b is (3,) or (N,3)
            return a @ b if np.ndim(b) == 1 else np.einsum("ij,ij->i", a, b)

        cos_theta = np.clip(_dot(u, bd), -1.0, 1.0)
        theta = np.arccos(cos_theta)

        sin_theta = np.sin(theta)
        cos_alpha = np.clip(_dot(u, pr), -1.0, 1.0)
        cos_phi = np.clip(cos_alpha / np.where(sin_theta > self.tol, sin_theta, 1.0),
                          -1.0, 1.0)
        phi = np.arccos(cos_phi)
        phi[v[:, 1] < 0] = 2.0 * np.pi - phi[v[:, 1] < 0]
        return theta, phi, r

    # -- per-hit lookup -----------------------------------------------------
    def lookup(self, slot_ids, pos_ids, origin=None, beam_direction=None):
        """
        Look up per-hit theta/phi/r/tof for arrays of hit ids.

        If both ``origin`` and ``beam_direction`` are None, the precomputed
        fixed-apex tables are used (fast). If either is given, the cone apex
        (``origin``) and/or axis (``beam_direction``) override the defaults and
        the angles are recomputed (still vectorised). Each override may be a
        single (3,) value or a per-hit (N,3) array -- e.g. a secondary-ring
        vertex and direction. The time-of-flight depends only on the apex.

        Out-of-range or missing-geometry channels are flagged invalid (NaN
        results) rather than raising.

        Returns ``theta, phi, r, tof, valid`` (NaN where invalid).
        """
        slot_ids = np.asarray(slot_ids).astype(int).ravel()
        pos_ids = np.asarray(pos_ids).astype(int).ravel()

        in_range = (
            (slot_ids >= 0) & (slot_ids < self.n_mpmt) &
            (pos_ids >= 0) & (pos_ids < self.n_pmt)
        )
        s = np.where(in_range, slot_ids, 0)
        p = np.where(in_range, pos_ids, 0)
        valid = in_range & self.channel_valid[s, p]

        if origin is None and beam_direction is None:
            theta = self.channel_theta[s, p]
            phi = self.channel_phi[s, p]
            r = self.channel_r[s, p]
            tof = self.channel_tof[s, p]
        else:
            positions = self.channel_pos[s, p]
            o = self.origin if origin is None else np.asarray(origin, dtype=float)
            theta, phi, r = self._angles_core(positions, o, beam_direction=beam_direction)
            tof = r / self.vg

        theta = np.where(valid, theta, np.nan)
        phi = np.where(valid, phi, np.nan)
        r = np.where(valid, r, np.nan)
        tof = np.where(valid, tof, np.nan)
        return theta, phi, r, tof, valid

    # -- convenience (single set of hits) ----------------------------------
    def compute(self, slot_ids, pos_ids, hit_times=None, origin=None,
                beam_direction=None) -> Dict[str, np.ndarray]:
        """
        Convenience wrapper around :meth:`lookup`. Returns a dict with
        ``theta``, ``phi``, ``r``, ``tof``, ``valid`` and, if ``hit_times`` is
        given, ``delta_t`` (= hit_time - tof). ``origin`` / ``beam_direction``
        override the cone apex / axis.
        """
        theta, phi, r, tof, valid = self.lookup(slot_ids, pos_ids, origin=origin,
                                                beam_direction=beam_direction)
        out = dict(theta=theta, phi=phi, r=r, tof=tof, valid=valid)
        if hit_times is not None:
            hit_times = np.asarray(hit_times, dtype=float).ravel()
            out["delta_t"] = np.where(valid, hit_times - tof, np.nan)
        return out


# ----------------------------------------------------------------------------
# Gaussian helpers (used only for the single-event diagnostic plot)
# ----------------------------------------------------------------------------
def gauss(x, A, mu, sigma):
    """A Gaussian, safe against sigma == 0."""
    return A * np.exp(-0.5 * ((x - mu) / np.where(sigma > 0, sigma, 1e-6)) ** 2)


def fit_time_residual_gaussian(delta_t, t_min, t_max, n_bins=100, do_fit=True) -> Dict[str, Any]:
    """
    Estimate the prompt-peak location of the time-residual (delta_t) histogram
    in [t_min, t_max]. ``mu`` is the histogram peak; when ``do_fit`` a Gaussian
    is additionally fit (cosmetic, for the diagnostic plot). Never raises.
    """
    delta_t = np.asarray(delta_t, dtype=float).ravel()
    delta_t = delta_t[np.isfinite(delta_t)]

    edges = np.linspace(t_min, t_max, n_bins + 1)
    counts, _ = np.histogram(delta_t, bins=edges)
    centers = 0.5 * (edges[:-1] + edges[1:])

    A_fit = counts.max() if counts.size else 10.0
    mu_fit = centers[int(np.argmax(counts))] if counts.size and counts.max() > 0 else 0.0
    sigma_fit = 1.0

    if do_fit and _SCIPY_AVAILABLE and counts.size and counts.max() > 0:
        try:
            popt, _ = curve_fit(gauss, centers, counts, p0=[A_fit, mu_fit, 1.0])
            A_fit, mu_fit, sigma_fit = popt
            sigma_fit = max(abs(sigma_fit), 1e-6)
        except Exception:
            pass

    return dict(A=A_fit, mu=mu_fit, sigma=sigma_fit,
                centers=centers, edges=edges, counts=counts)


# ----------------------------------------------------------------------------
# Per-event result container
# ----------------------------------------------------------------------------
@dataclass
class RingResults:
    """
    Per-event quantities produced by :meth:`CherenkovRingSelection.process_events`.

    Attributes
    ----------
    q_total   : total charge per event (geometry-valid hits; after the coarse
                time window too when the time cut is on)
    q_time    : charge after the prompt-time window (== q_total when the time
                cut is off)
    q_inside  : charge inside the Cherenkov cone (theta < angle_cut)
    q_outside : charge outside the cone, == |q_time - q_inside|
    event_id  : event id for each entry (input order preserved, failures kept)
    fail_ids  : dict of lists of event ids rejected at each stage
    theta     : optional per-event jagged array (rad) of the geometry-valid hits
    delta_t   : optional per-event jagged array (ns); None if not computed
    """
    q_total:   np.ndarray = field(default_factory=lambda: np.empty(0))
    q_time:    np.ndarray = field(default_factory=lambda: np.empty(0))
    q_inside:  np.ndarray = field(default_factory=lambda: np.empty(0))
    q_outside: np.ndarray = field(default_factory=lambda: np.empty(0))
    event_id:  np.ndarray = field(default_factory=lambda: np.empty(0, dtype=int))
    fail_ids:  Dict[str, list] = field(default_factory=dict)
    theta:     Any = None
    delta_t:   Any = None

    def __len__(self):
        return len(self.q_total)

    @property
    def ring_fraction(self):
        """q_inside / q_total per event (0 where q_total == 0)."""
        qt = self.q_total
        return np.divide(self.q_inside, qt, out=np.zeros_like(qt), where=qt > 0)

    @property
    def outside_fraction(self):
        """q_outside / q_total per event (0 where q_total == 0)."""
        qt = self.q_total
        return np.divide(self.q_outside, qt, out=np.zeros_like(qt), where=qt > 0)

    @classmethod
    def concatenate(cls, results: Sequence["RingResults"]) -> "RingResults":
        """Concatenate a sequence of RingResults (e.g. one per loader batch)."""
        if not results:
            return cls()
        merged_fail: Dict[str, list] = {}
        for r in results:
            for key, ids in r.fail_ids.items():
                merged_fail.setdefault(key, []).extend(ids)

        def cat_jagged(attr):
            parts = [getattr(r, attr) for r in results]
            if any(p is None for p in parts):
                return None
            import awkward as ak
            return ak.concatenate(parts)

        return cls(
            q_total=np.concatenate([r.q_total for r in results]),
            q_time=np.concatenate([r.q_time for r in results]),
            q_inside=np.concatenate([r.q_inside for r in results]),
            q_outside=np.concatenate([r.q_outside for r in results]),
            event_id=np.concatenate([r.event_id for r in results]),
            fail_ids=merged_fail,
            theta=cat_jagged("theta"),
            delta_t=cat_jagged("delta_t"),
        )


# ----------------------------------------------------------------------------
# Hit-level Cherenkov-ring selection
# ----------------------------------------------------------------------------
class CherenkovRingSelection:
    """
    Hit-level selection: optional prompt-time window + Cherenkov-cone angular cut.

    Parameters
    ----------
    ring_geometry : RingGeometry
        Configured geometry/timing backbone (holds the precomputed channel tables).
    angle_cut_deg : float or (n_events,) array
        Cherenkov cone half-angle in degrees; hits with theta below this are
        "inside the ring". May be a single value (e.g. 44) or a per-event array.
        Compute it from the particle momentum with
        :func:`cherenkov_cone_halfangle` (Cherenkov angle + margin). Can also be
        overridden per call in :meth:`process_events` / :meth:`event_charges`.
    apply_time_cut : bool
        If True (default) apply the prompt-time window (coarse window + per-event
        ``|delta_t - mu| < K``). If False, skip all timing: no coarse window, no
        TOF subtraction, no prompt fit -- charge is split inside/outside the cone
        using every hit passed in. Use False when timing is handled upstream.
    time_window_K : float
        Half-width (ns) of the prompt-time window.
    fit_t_min, fit_t_max : float
        Range (ns) over which the per-event prompt peak ``mu`` is found; only
        needs to bracket the prompt peak. Used only when ``apply_time_cut``.
    n_bins : int
        Number of bins used to locate the per-event prompt peak.
    min_hits : int
        Minimum number of geometry-valid hits required to process an event.
    coarse_time_window : (float, float), optional
        Optional pre-cut on calibrated hit times, applied only when
        ``apply_time_cut`` is True.
    use_gaussian_fit : bool
        Only affects the single-event diagnostic plot: if True the drawn curve is
        a Gaussian fit. The cut always uses the histogram-peak ``mu`` so that the
        single-event and batch paths are identical.
    """

    def __init__(self,
                 ring_geometry: RingGeometry,
                 angle_cut_deg: float = 44.0,
                 apply_time_cut: bool = True,
                 time_window_K: float = 3.0,
                 fit_t_min: float = 1670.0,
                 fit_t_max: float = 1770.0,
                 n_bins: int = 100,
                 min_hits: int = 10,
                 coarse_time_window: Optional[Tuple[float, float]] = None,
                 use_gaussian_fit: bool = False):
        self.ring = ring_geometry
        self.angle_cut_deg = float(angle_cut_deg)
        self.apply_time_cut = bool(apply_time_cut)
        self.time_window_K = float(time_window_K)
        self.fit_t_min = float(fit_t_min)
        self.fit_t_max = float(fit_t_max)
        self.n_bins = int(n_bins)
        self.min_hits = int(min_hits)
        self.coarse_time_window = coarse_time_window
        self.use_gaussian_fit = bool(use_gaussian_fit)

    # -- describe -----------------------------------------------------------
    def describe(self):
        print("CherenkovRingSelection")
        if np.ndim(self.angle_cut_deg) == 0:
            print(f"  cone half-angle      : theta < {self.angle_cut_deg:g} deg")
        else:
            a = np.asarray(self.angle_cut_deg, dtype=float)
            print(f"  cone half-angle      : per-event array, "
                  f"theta < [{np.nanmin(a):g}..{np.nanmax(a):g}] deg")
        print(f"  time cut             : {'on' if self.apply_time_cut else 'OFF (timing handled upstream)'}")
        if self.apply_time_cut:
            print(f"  prompt-time window   : |dt - mu| < {self.time_window_K:g} ns "
                  f"(peak found in [{self.fit_t_min:g}, {self.fit_t_max:g}] ns, {self.n_bins} bins)")
            if self.coarse_time_window is not None:
                lo, hi = self.coarse_time_window
                print(f"  coarse time pre-cut  : {lo:g} < t < {hi:g} ns")
        print(f"  min valid hits       : {self.min_hits}")

    # -- flatten any event source into flat hit arrays + event boundaries ---
    def _flatten(self, events, fields, max_events):
        slot_f, pos_f, charge_f, time_f = fields
        try:
            import awkward as ak
            is_awkward = isinstance(events, ak.Array)
        except ImportError:
            ak = None
            is_awkward = False

        if is_awkward:
            if max_events is not None:
                events = events[:max_events]
            n_events = len(events)
            counts = ak.to_numpy(ak.num(events[slot_f], axis=1)).astype(np.int64)
            slot = ak.to_numpy(ak.flatten(events[slot_f])).astype(np.int64)
            pos = ak.to_numpy(ak.flatten(events[pos_f])).astype(np.int64)
            charge = ak.to_numpy(ak.flatten(events[charge_f])).astype(float)
            time = ak.to_numpy(ak.flatten(events[time_f])).astype(float)
        else:
            slot_l, pos_l, charge_l, time_l, counts_l = [], [], [], [], []
            for i, ev in enumerate(events):
                if max_events is not None and i >= max_events:
                    break
                s = np.asarray(ev[slot_f]).astype(np.int64).ravel()
                slot_l.append(s)
                pos_l.append(np.asarray(ev[pos_f]).astype(np.int64).ravel())
                charge_l.append(np.asarray(ev[charge_f]).astype(float).ravel())
                time_l.append(np.asarray(ev[time_f]).astype(float).ravel())
                counts_l.append(s.size)
            n_events = len(counts_l)
            counts = np.asarray(counts_l, dtype=np.int64)
            cat = lambda L: np.concatenate(L) if L else np.empty(0)
            slot, pos, charge, time = cat(slot_l), cat(pos_l), cat(charge_l), cat(time_l)

        ev_idx = np.repeat(np.arange(n_events), counts) if n_events else np.empty(0, dtype=np.int64)
        return slot, pos, charge, time, ev_idx, n_events
    
    # -- get dt and angle -----------------------------------------------------------
    def get_hit_quantities( self, event=None, *, slot_ids=None, pos_ids=None, charges=None, hit_times=None, origin=None,
        beam_direction=None, slot_field="hit_mpmt_slot_ids", pos_field="hit_pmt_position_ids", charge_field="hit_pmt_charges",
        time_field="hit_pmt_calibrated_times",):
        """
        Return hit-level quantities BEFORE any cuts.
        This is intended for debugging and visualisation.
        Parameters
        ----------
        event : awkward record, optional
            Single event containing hit branches.
        slot_ids, pos_ids, charges, hit_times : array-like, optional
            Can be provided directly instead of `event`.
        origin : (3,), optional
            Event-specific cone apex (e.g. T5 entry position).
        beam_direction : (3,), optional
            Event-specific beam direction.
        Returns
        -------
        dict containing:
            slot, pos, charge, time, theta (rad), theta_deg, phi (rad), r (mm), tof (ns), delta_t (ns), valid
        """
        # Extract hit arrays
        if event is not None:
            slot = np.asarray(event[slot_field], dtype=np.int64).ravel()
            pos = np.asarray(event[pos_field], dtype=np.int64).ravel()
            charge = np.asarray(event[charge_field], dtype=float).ravel()
            time = np.asarray(event[time_field], dtype=float).ravel()
        else:
            slot = np.asarray(slot_ids, dtype=np.int64).ravel()
            pos = np.asarray(pos_ids, dtype=np.int64).ravel()
            charge = np.asarray(charges, dtype=float).ravel()
            time = np.asarray(hit_times, dtype=float).ravel()

        # Geometry lookup
        theta, phi, r, tof, valid = self.ring.lookup(
            slot,
            pos,
            origin=origin,
            beam_direction=beam_direction,)
        delta_t = time - tof

        return dict(
            slot=slot,
            pos=pos,
            charge=charge,
            time=time,
            theta=theta,
            theta_deg=np.degrees(theta),
            phi=phi,
            r=r,
            tof=tof,
            delta_t=delta_t,
            valid=valid,)

    
    # -- the single shared core for process_event_charge and process_events functions--------------------------------------------
    def _summarise(self, slot, pos, charge, time, ev_idx, n_events,
                   apply_time_cut, return_hits, origins=None, beam_directions=None,
                   angle_cut_deg=None):
        """
        Vectorised per-event summary. Both event_charges() and process_events()
        call this -- there is only one implementation of the physics.

        ``origins`` / ``beam_directions`` optionally override the cone apex and
        axis: each a single (3,) value or a per-event (n_events, 3) array (e.g.
        T5 entry positions, or a secondary-ring vertex/direction). When given,
        angles/tof are recomputed per hit rather than read from the tables.

        ``angle_cut_deg`` overrides the cone half-angle: a scalar, or a per-event
        (n_events,) array (e.g. a per-event Cherenkov angle from the momentum).

        Returns a dict of per-event arrays plus, when requested, per-event jagged
        theta / delta_t and the per-event mu (for the single-event plot).
        """
        # --- hit-level keep: the coarse time window, only when the time cut is
        #     on. Channel/data-quality cuts are assumed already applied upstream
        #     (e.g. by DataLoader); trigger mainboards and bad geometry are
        #     dropped anyway by the geometry-bounds check in RingGeometry.lookup.
        if apply_time_cut and self.coarse_time_window is not None:
            lo, hi = self.coarse_time_window
            keep = (time > lo) & (time < hi)
            slot, pos, charge, time, ev_idx = (slot[keep], pos[keep], charge[keep],
                                               time[keep], ev_idx[keep])

        n_hits = np.bincount(ev_idx, minlength=n_events)

        # --- resolve cone apex / axis (fixed tables, or recomputed per hit) ---
        def _per_hit(arr):
            if arr is None:
                return None
            arr = np.asarray(arr, dtype=float)
            return arr if arr.ndim == 1 else arr[ev_idx]

        origin_arg = _per_hit(origins)
        beamdir_arg = _per_hit(beam_directions)

        # --- geometry lookup (precomputed tables, or recomputed per hit) ---
        theta, _phi, _r, tof, valid = self.ring.lookup(
            slot, pos, origin=origin_arg, beam_direction=beamdir_arg)
        ev_v = ev_idx[valid]
        n_valid = np.bincount(ev_v, minlength=n_events)

        # --- time residual (only if needed) ---
        need_dt = apply_time_cut or return_hits
        delta_t = (time - tof) if need_dt else None

        # --- per-event prompt peak mu (vectorised histogram + argmax) ---
        mu = np.full(n_events, np.nan)
        if apply_time_cut:
            width = (self.fit_t_max - self.fit_t_min) / self.n_bins
            inwin = valid & np.isfinite(delta_t) & \
                    (delta_t >= self.fit_t_min) & (delta_t < self.fit_t_max)
            bin_idx = np.clip(((delta_t[inwin] - self.fit_t_min) / width).astype(int),
                              0, self.n_bins - 1)
            counts2d = np.zeros((n_events, self.n_bins), dtype=np.int64)
            np.add.at(counts2d, (ev_idx[inwin], bin_idx), 1)
            has_peak = counts2d.max(axis=1) > 0
            centers = self.fit_t_min + (np.arange(self.n_bins) + 0.5) * width
            mu = np.where(has_peak, centers[counts2d.argmax(axis=1)], np.nan)

            mu_hit = mu[ev_idx]
            time_pass = valid & np.isfinite(mu_hit) & \
                        (np.abs(delta_t - mu_hit) < self.time_window_K)
        else:
            time_pass = valid

        n_time = np.bincount(ev_idx[time_pass], minlength=n_events)

        # --- cone half-angle: scalar or per-event array ---
        cut = self.angle_cut_deg if angle_cut_deg is None else angle_cut_deg
        if np.ndim(cut) == 0:
            thr_hit = cut
        else:
            thr_hit = np.asarray(cut, dtype=float)[ev_idx]
        # NaN threshold (e.g. below Cherenkov threshold) -> nothing inside
        inside = valid & (np.degrees(theta) < thr_hit)

        # --- per-event charge sums ---
        q_total = np.bincount(ev_idx[valid], weights=charge[valid], minlength=n_events)
        q_time = np.bincount(ev_idx[time_pass], weights=charge[time_pass], minlength=n_events)
        in_and_time = time_pass & inside
        q_inside = np.bincount(ev_idx[in_and_time], weights=charge[in_and_time], minlength=n_events)
        q_outside = np.abs(q_time - q_inside)

        # --- per-event status (precedence: no_hits > no_geom > few_hits > no_time) ---
        status = np.full(n_events, _OK, dtype=np.int8)
        if apply_time_cut:
            status[n_time == 0] = _NO_TIME
        status[n_valid < self.min_hits] = _FEW_HITS
        status[(n_hits > 0) & (n_valid == 0)] = _NO_GEOM
        status[n_hits == 0] = _NO_HITS

        # Events that never reach the charge step contribute zero charge
        # (no_hits / no_geom / few_hits). no_time keeps q_total but has q_time =
        # q_inside = q_outside = 0 by construction.
        no_charge = (status == _NO_HITS) | (status == _NO_GEOM) | (status == _FEW_HITS)
        q_total[no_charge] = 0.0
        q_time[no_charge] = 0.0
        q_inside[no_charge] = 0.0
        q_outside[no_charge] = 0.0

        out = dict(q_total=q_total, q_time=q_time, q_inside=q_inside,
                   q_outside=q_outside, status=status, mu=mu,
                   n_hits=n_hits, n_valid=n_valid, n_time=n_time)

        if return_hits:
            import awkward as ak
            theta_jag = ak.unflatten(theta[valid], n_valid)
            out["theta"] = theta_jag
            out["delta_t"] = ak.unflatten(delta_t[valid], n_valid) if need_dt else None
        return out

    # -- single event -------------------------------------------------------
    def process_event_charge(self, event=None, *, slot_ids=None, pos_ids=None,
                      charges=None, hit_times=None, event_id=None,
                      apply_time_cut=None, origin=None, beam_direction=None,
                      angle_cut_deg=None, do_plot=False,
                      slot_field="hit_mpmt_slot_ids", pos_field="hit_pmt_position_ids",
                      charge_field="hit_pmt_charges", time_field="hit_pmt_calibrated_times"):
        """
        Compute (q_total, q_time, q_inside, q_outside) for one event, via the
        same core as the batch path.

        Pass an awkward ``event`` record (hit branches read from the standard
        field names) or the four hit arrays directly. ``origin`` / ``beam_direction``
        optionally set this event's cone apex / axis (each a (3,) vector, e.g.
        its T5 entry position and the particle direction). ``angle_cut_deg``
        overrides the cone half-angle for this event (e.g. its Cherenkov angle).
        Returns a dict with keys ``q_total``, ``q_time``, ``q_inside``,
        ``q_outside``, ``status`` and the per-hit ``theta`` / ``delta_t``.
        """
        atc = self.apply_time_cut if apply_time_cut is None else bool(apply_time_cut)

        if event is not None:
            slot = np.asarray(event[slot_field]).astype(np.int64).ravel()
            pos = np.asarray(event[pos_field]).astype(np.int64).ravel()
            charge = np.asarray(event[charge_field]).astype(float).ravel()
            time = np.asarray(event[time_field]).astype(float).ravel()
        else:
            slot = np.asarray(slot_ids).astype(np.int64).ravel()
            pos = np.asarray(pos_ids).astype(np.int64).ravel()
            charge = np.asarray(charges, dtype=float).ravel()
            time = np.asarray(hit_times, dtype=float).ravel()

        ev_idx = np.zeros(slot.size, dtype=np.int64)
        s = self._summarise(slot, pos, charge, time, ev_idx, 1,
                            apply_time_cut=atc, return_hits=True,
                            origins=(None if origin is None else np.asarray(origin, float)),
                            beam_directions=(None if beam_direction is None
                                             else np.asarray(beam_direction, float)),
                            angle_cut_deg=angle_cut_deg)

        import awkward as ak
        theta = ak.to_numpy(s["theta"][0]) if len(s["theta"]) else np.empty(0)
        delta_t = (ak.to_numpy(s["delta_t"][0]) if (s["delta_t"] is not None and len(s["delta_t"]))
                   else np.empty(0))
        res = dict(q_total=float(s["q_total"][0]), q_time=float(s["q_time"][0]),
                   q_inside=float(s["q_inside"][0]), q_outside=float(s["q_outside"][0]),
                   theta=theta, delta_t=delta_t, status=_STATUS_NAME[int(s["status"][0])])

        if do_plot and _MPL_AVAILABLE:
            cut0 = self.angle_cut_deg if angle_cut_deg is None else angle_cut_deg
            cut0 = float(cut0) if np.ndim(cut0) == 0 else float(np.asarray(cut0).ravel()[0])
            self._plot_time_window(theta, delta_t, float(s["mu"][0]), event_id, atc,
                                   angle_cut_deg=cut0)
        return res

    # -- many events --------------------------------------------------------
    def process_events(self, events, max_events=None, start_index=0,
                       apply_time_cut=None, origins=None, beam_directions=None,
                       angle_cut_deg=None, return_hits=False,
                       verbose=False, do_plot=False,
                       slot_field="hit_mpmt_slot_ids", pos_field="hit_pmt_position_ids",
                       charge_field="hit_pmt_charges", time_field="hit_pmt_calibrated_times"
                       ) -> RingResults:
        """
        Process an awkward batch (from ``DataLoader.iterate``) or any iterable of
        event records and return a :class:`RingResults`. Fully vectorised: the
        whole batch goes through one pass of :meth:`_summarise`.

        Parameters
        ----------
        return_hits : bool
            If True, also return per-event jagged ``theta`` / ``delta_t`` on the
            result (aligned to the geometry-valid hits of each event). Off by
            default -- materialising every hit is memory-heavy on a full run.
        apply_time_cut : bool, optional
            Override the selection's default for this call.
        origins, beam_directions : (3,) or (n_events, 3) array, optional
            Cone apex / axis override. A single value applies to every event; a
            per-event array (in input event order and length, before
            ``max_events``) gives each event its own apex / axis. Use for T5
            entry positions or secondary-ring vertices and directions.
        angle_cut_deg : float or (n_events,) array, optional
            Cone half-angle override (e.g. a per-event Cherenkov angle from the
            momentum, via :func:`cherenkov_cone_halfangle`).
        start_index : int
            Added to the within-batch event index to keep ids unique across
            successive batches.
        """
        atc = self.apply_time_cut if apply_time_cut is None else bool(apply_time_cut)
        if do_plot:
            print("note: do_plot is ignored by process_events; use event_charges for plots.")

        fields = (slot_field, pos_field, charge_field, time_field)
        slot, pos, charge, time, ev_idx, n_events = self._flatten(events, fields, max_events)

        # Trim per-event overrides to the events actually processed.
        # A single override is (3,) (1-D); a per-event override is (n_events, 3)
        # (2-D) or (n_events,) for angle_cut -- only those need trimming.
        def _trim(arr):
            if arr is None:
                return None
            arr = np.asarray(arr, dtype=float)
            if arr.ndim == 2 and max_events is not None:
                arr = arr[:n_events]
            return arr

        origins = _trim(origins)
        beam_directions = _trim(beam_directions)
        if angle_cut_deg is not None and np.ndim(angle_cut_deg) > 0 and max_events is not None:
            angle_cut_deg = np.asarray(angle_cut_deg, dtype=float)[:n_events]

        s = self._summarise(slot, pos, charge, time, ev_idx, n_events,
                            apply_time_cut=atc, return_hits=return_hits,
                            origins=origins, beam_directions=beam_directions,
                            angle_cut_deg=angle_cut_deg)

        event_id = np.arange(n_events) + start_index
        status = s["status"]
        fail_ids = {name: event_id[status == code].tolist()
                    for code, name in _STATUS_NAME.items() if code != _OK}

        if verbose:
            n_ok = int(np.sum(status == _OK))
            print(f"  {n_events} events, {n_ok} ok; "
                  f"rejected: { {k: len(v) for k, v in fail_ids.items()} }")

        return RingResults(
            q_total=s["q_total"], q_time=s["q_time"],
            q_inside=s["q_inside"], q_outside=s["q_outside"],
            event_id=event_id, fail_ids=fail_ids,
            theta=s.get("theta"), delta_t=s.get("delta_t"),
        )

    # -- single-event diagnostic plot --------------------------------------
    def _plot_time_window(self, theta, delta_t, mu, event_id, apply_time_cut,
                          angle_cut_deg=None):
        cut = self.angle_cut_deg if angle_cut_deg is None else angle_cut_deg
        cut = float(cut) if np.ndim(cut) == 0 else float(np.asarray(cut).ravel()[0])
        theta_deg = np.degrees(theta)
        if apply_time_cut and theta.size and np.isfinite(mu):
            K = self.time_window_K
            mask = np.abs(delta_t - mu) < K
            fig, ax = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)

            fit = fit_time_residual_gaussian(delta_t, self.fit_t_min, self.fit_t_max,
                                             self.n_bins, do_fit=self.use_gaussian_fit)
            ax[0].bar(fit["centers"], fit["counts"], width=np.diff(fit["edges"]),
                      align="edge", color="0.85", edgecolor="k", label="dt data")
            ax[0].axvspan(mu - K, mu + K, color="C4", alpha=0.18, label=f"mu +/- {K} ns")
            if self.use_gaussian_fit:
                ax[0].plot(fit["centers"], gauss(fit["centers"], fit["A"], fit["mu"], fit["sigma"]),
                           "C3-", lw=2, label="gauss fit")
            ax[0].axvline(mu, color="C3", ls="--", lw=1)
            ax[0].set_xlabel("dt (ns)"); ax[0].set_ylabel("Counts")
            ax[0].set_title(f"Event {event_id} : dt histogram" if event_id is not None else "dt histogram")
            ax[0].legend(fontsize="small")

            ax[1].scatter(theta_deg, delta_t, s=1, c="0.80", label="all hits")
            if mask.any():
                ax[1].scatter(theta_deg[mask], delta_t[mask], s=3, c="C0",
                              label=f"in window ({int(mask.sum())})")
            ax[1].axhspan(mu - K, mu + K, color="C4", alpha=0.10)
            ax[1].axvline(cut, color="k", ls=":", lw=1, label=f"cone {cut:g} deg")
            ax[1].set_xlim(0, 180)
            ax[1].set_xlabel("theta (deg)"); ax[1].set_ylabel("dt (ns)")
            ax[1].set_title("theta vs dt (prompt window)")
            ax[1].legend(loc="upper right", fontsize="small")
            plt.show()
        else:
            # time cut off: just show the angular distribution of the charge
            fig, ax = plt.subplots(figsize=(6, 4))
            ax.hist(theta_deg, bins=60, color="0.7", edgecolor="k")
            ax.axvline(cut, color="C3", ls="--", label=f"cone {cut:g} deg")
            ax.set_xlabel("theta (deg)"); ax.set_ylabel("hits")
            ax.set_title(f"Event {event_id} : theta distribution" if event_id is not None else "theta distribution")
            ax.legend(fontsize="small")
            plt.show()


# ----------------------------------------------------------------------------
# Topology classification + plotting helpers
# ----------------------------------------------------------------------------
def classify_charge_topology(q_inside, q_outside,
                             outside_track_max=3800.0,
                             outside_shower_min=4000.0,
                             track_percentiles=(1, 23, 23.1, 96, 96.1, 99.99),
                             shower_percentiles=(0.1, 13, 28, 99.9999)) -> Dict[str, np.ndarray]:
    """
    Split events into track-like and shower-like topologies using the charge
    deposited outside the Cherenkov cone, then bin each topology into low / mid /
    high bands by the inside-cone charge percentiles. Returns a dict of boolean
    masks (``track_low/mid/high``, ``shower_low/mid/high``) plus ``track_cuts`` and
    ``shower_cuts``. Adjust thresholds and percentiles for your run.
    """
    q_inside = np.asarray(q_inside, dtype=float)
    q_outside = np.asarray(q_outside, dtype=float)

    track = q_outside < outside_track_max
    shower = q_outside > outside_shower_min
    out: Dict[str, np.ndarray] = {}

    vals = q_inside[track]; vals = vals[vals > 0]
    if vals.size:
        c = np.percentile(vals, track_percentiles)
        out["track_low"] = track & (q_inside > c[0]) & (q_inside <= c[1])
        out["track_mid"] = track & (q_inside > c[2]) & (q_inside <= c[3])
        out["track_high"] = track & (q_inside > c[4]) & (q_inside <= c[5])
        out["track_cuts"] = c
    else:
        for k in ("track_low", "track_mid", "track_high"):
            out[k] = np.zeros_like(track)
        out["track_cuts"] = np.array([])

    vals = q_inside[shower]; vals = vals[vals > 0]
    if vals.size:
        ce = np.percentile(vals, shower_percentiles)
        out["shower_low"] = shower & (q_inside > ce[0]) & (q_inside <= ce[1])
        out["shower_mid"] = shower & (q_inside > ce[1]) & (q_inside <= ce[2])
        out["shower_high"] = shower & (q_inside > ce[2]) & (q_inside <= ce[3])
        out["shower_cuts"] = ce
    else:
        for k in ("shower_low", "shower_mid", "shower_high"):
            out[k] = np.zeros_like(shower)
        out["shower_cuts"] = np.array([])
    return out

def plot_inside_vs_outside(q_inside, q_outside, topology=None,
                           outside_track_max=None, outside_shower_min=None,
                           xlim=None, ylim=None, bins=300, ax=None,
                           style="regions", cmap=None):
    """
    2D histogram of charge inside vs outside the Cherenkov cone, with the
    track/shower split drawn on top.

    Parameters
    ----------
    topology : dict, optional
        Output of :func:`classify_charge_topology`. If given, its percentile
        bands are drawn as rectangular regions, and its thresholds supply the
        track/shower split lines.
    outside_track_max, outside_shower_min : float, optional
        Track/shower split thresholds on the outside-cone charge. Pass these to
        draw the split lines and zone labels **without** computing a full
        topology. They override the values found in ``topology``.
    bins : int
        Number of bins per axis (default 300).
    style : {"regions", "boundaries", "scatter"}
        How to overlay the topology categories (only used when ``topology`` is
        given): shaded rectangles (default), outlines only, or the old scatter.
    cmap : str, optional
        Colormap for the density. Defaults to "viridis" (visible on sparse
        data); when ``topology`` regions are drawn it defaults to "Greys" so the
        coloured regions stand out. Pass any name to override.

    Returns the matplotlib Axes.
    """
    if not _MPL_AVAILABLE:
        raise RuntimeError("matplotlib is required for plot_inside_vs_outside")
    from matplotlib.colors import LogNorm
    from matplotlib.patches import Rectangle, Patch

    q_inside = np.asarray(q_inside, dtype=float)
    q_outside = np.asarray(q_outside, dtype=float)
    if ax is None:
        _, ax = plt.subplots(figsize=(7, 6))

    draw_regions = (topology is not None and style in ("regions", "boundaries"))
    if cmap is None:
        cmap = "Greys" if draw_regions else "viridis"

    h = ax.hist2d(q_inside, q_outside, bins=(bins, bins), norm=LogNorm(), cmap=cmap)
    plt.colorbar(h[3], ax=ax, label="counts (log)")

    # Fix the axis limits before drawing regions / labels so they fill the frame.
    if xlim is not None:
        ax.set_xlim(xlim)
    if ylim is not None:
        ax.set_ylim(ylim)
    x0, x1 = ax.get_xlim()
    y0, y1 = ax.get_ylim()

    # Resolve the track/shower thresholds: explicit args win, else topology.
    otm = outside_track_max
    osm = outside_shower_min
    if topology is not None:
        if otm is None:
            otm = topology.get("outside_track_max")
        if osm is None:
            osm = topology.get("outside_shower_min")

    if topology is not None and style == "scatter":
        overlays = [("track_low", "blue", "track low"), ("track_mid", "green", "track mid"),
                    ("track_high", "orange", "track high"), ("shower_low", "purple", "shower low"),
                    ("shower_mid", "pink", "shower mid"), ("shower_high", "magenta", "shower high")]
        for key, color, label in overlays:
            m = topology.get(key)
            if m is not None and np.any(m):
                ax.scatter(q_inside[m], q_outside[m], s=0.2, c=color, alpha=0.3, label=label)
        ax.legend(fontsize=7, loc="best")

    elif draw_regions:
        tcuts = np.asarray(topology.get("track_cuts", []), dtype=float)
        scuts = np.asarray(topology.get("shower_cuts", []), dtype=float)
        track_bands = [(0, 1, "#1f77b4", "track low"),
                       (2, 3, "#2ca02c", "track mid"),
                       (4, 5, "#ff7f0e", "track high")]
        shower_bands = [(0, 1, "#9467bd", "shower low"),
                        (1, 2, "#e377c2", "shower mid"),
                        (2, 3, "#d62728", "shower high")]
        filled = (style == "regions")
        handles = []

        def add_band(cuts, lo_i, hi_i, ylo, yhi, color, label):
            if hi_i >= len(cuts):
                return
            xlo, xhi = cuts[lo_i], cuts[hi_i]
            if not (np.isfinite(xlo) and np.isfinite(xhi)) or xhi <= xlo or yhi <= ylo:
                return
            ax.add_patch(Rectangle(
                (xlo, ylo), xhi - xlo, yhi - ylo,
                facecolor=(color if filled else "none"),
                edgecolor=color, lw=1.4, alpha=(0.16 if filled else 1.0), zorder=3))
            handles.append(Patch(facecolor=color, edgecolor=color, alpha=0.5, label=label))

        if tcuts.size and otm is not None:
            for lo_i, hi_i, color, label in track_bands:
                add_band(tcuts, lo_i, hi_i, max(y0, 0.0), otm, color, label)
        if scuts.size and osm is not None:
            for lo_i, hi_i, color, label in shower_bands:
                add_band(scuts, lo_i, hi_i, osm, y1, color, label)
        if handles:
            ax.legend(handles=handles, fontsize=8, loc="upper right",
                      framealpha=0.9, title="topology")

    # The track/shower split lines + zone labels (whenever thresholds are known).
    if otm is not None:
        ax.axhline(otm, color="crimson", ls="--", lw=1.3, zorder=4)
        ax.text(x1, otm, " track-like \u2193", color="crimson", fontsize=8,
                ha="right", va="top", zorder=4)
    if osm is not None and osm != otm:
        ax.axhline(osm, color="crimson", ls="--", lw=1.3, zorder=4)
    if osm is not None:
        ax.text(x1, osm, " shower-like \u2191", color="crimson", fontsize=8,
                ha="right", va="bottom", zorder=4)

    ax.set_xlabel("total charge inside Cherenkov cone")
    ax.set_ylabel("total charge outside Cherenkov cone")
    ax.set_title("Charge deposited in the tank: inside vs outside the ring")
    return ax

