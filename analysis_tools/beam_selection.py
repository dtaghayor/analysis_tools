import awkward as ak
import numpy as np
import math
import matplotlib.pyplot as plt

from typing import Callable, Any, Optional, Union, Tuple

from dataclasses import dataclass

try:
    from IPython.display import clear_output as _ipython_clear_output
    _IPYTHON_AVAILABLE = True
except ImportError:
    _IPYTHON_AVAILABLE = False


@dataclass
class Cut:
    """
    Generic cut object.
    """

    name:      str
    variable:  str
    func:      Callable[[Any], Any]
    threshold: Optional[Union[float, Tuple[float, float]]] = None
    direction: Optional[str] = None   # "above", "below", "between", or None for pass-all

    def apply(self, batch):
        """
        Returns boolean mask.
        """
        values = batch[self.variable]
        return self.func(values)


    @staticmethod
    def greater_than(threshold):
        return lambda x: x > threshold

    @staticmethod
    def less_than(threshold):
        return lambda x: x < threshold

    @staticmethod
    def between(low, high):
        return lambda x: (x > low) & (x < high)
    
    @staticmethod
    def equal_to(value):
        return lambda x: x == value
    
    @staticmethod
    def not_equal_to(value):
        return lambda x: x != value

    @staticmethod
    def true():
        return lambda x: x == x

    @staticmethod
    def false():
        return lambda x: x != x

    def _give_name(self):
        return self.name

    def _give_variable(self):
        return self.variable

    def _give_function(self):
        return self.func



class BeamSelection:

    def __init__(self, name):
        self.name   = name
        self.cuts   = []
        self._write_parquet  = False
        self._parquet_path   = None
        self._parquet_writer = None

    def add(self, cut: Cut):
        self.cuts.append(cut)

    def mask(self, batch):
        """
        Combined AND of all cuts.
        """
        mask = self.cuts[0].apply(batch)
        for cut in self.cuts[1:]:
            mask = cut.apply(batch) & mask
        return mask

    def describe(self):
        """Print the cuts applied by this selection with threshold values and directions."""
        print(f"Selection : {self.name}")
        for cut in self.cuts:
            unit = _VARIABLE_UNITS.get(cut.variable, "")
            unit_str = f" {unit}" if unit else ""
            if cut.threshold is None:
                value_str = "pass all (no TOF separation available for this run)"
            elif cut.direction == "above":
                value_str = f"> {cut.threshold:.3g}{unit_str}  [above threshold]"
            elif cut.direction == "below":
                value_str = f"< {cut.threshold:.3g}{unit_str}  [below threshold]"
            elif cut.direction == "between":
                lo, hi = cut.threshold
                value_str = f"between {lo:.3g} and {hi:.3g}{unit_str}"
            else:
                value_str = str(cut.threshold)
            print(f"  {cut.variable:<22} {value_str}")

    def enable_parquet_output(self, path=None):
        """
        Enable writing selected events to a parquet file during iteration.
        Call this before the batch loop, then call close_parquet_writer() after.

        Parameters
        ----------
        path : str, optional
            Output file path. Defaults to "<selection_name>.parquet".
        """
        self._parquet_path   = path if path is not None else f"{self.name}.parquet"
        self._write_parquet  = True
        self._parquet_writer = None

    def _write_to_parquet(self, batch):
        if not self._write_parquet or len(batch) == 0:
            return
        import pyarrow.parquet as pq
        table = ak.to_arrow_table(batch)
        if self._parquet_writer is None:
            self._parquet_writer = pq.ParquetWriter(self._parquet_path, table.schema)
        self._parquet_writer.write_table(table)

    def close_parquet_writer(self):
        if self._parquet_writer is not None:
            self._parquet_writer.close()
            self._parquet_writer = None
            print(f"Wrote {self.name} events to: {self._parquet_path}")

    # ------------------------------------------------------------------
    # Named constructors — one per particle type.
    # Thresholds are explicit so users can see and modify them directly.
    # ------------------------------------------------------------------

    @classmethod
    def selection(cls, name, *cuts):
        """
        Custom selection: to be named by the user and defined by arbitrary cuts.
        Pass all cuts as [variable, operator, value] triplets.
        ---------------------------------------------------------------------------
        Electrons: fast particles above threshold in the upstream ACT (act_eveto).

        Typical cuts
        ------------
        ["act_eveto", ">", act_eveto_cut]
        ["tof",       "<", proton_tof_cut]   # omit if TOF info is unavailable
        ---------------------------------------------------------------------------
        Muons (or muons at higher momenta in kaon runs): fast particles, below threshold in the upstream ACT (act_eveto)
        and above threshold in the downstream ACT (act_tagger).
        
        Typical cuts
        ------------
        ["act_eveto",    "<", act_eveto_cut]
        ["act_tagger",   ">", act_tagger_cut]
        ["tof",          "<", proton_tof_cut]   # omit if TOF info is unavailable
        ["mu_tag_total", ">", mu_tag_cut]        # optional extra cut
        ---------------------------------------------------------------------------
        Pions: fast particles below threshold in both ACTs. 
        Typical cuts
        ------------
        ["act_eveto",  "<", act_eveto_cut]
        ["act_tagger", "<", act_tagger_cut]
        ["tof",        "<", proton_tof_cut]   # omit if TOF info is unavailable
        ---------------------------------------------------------------------------
        protons: slow particles identified by their TOF falling in a window above the fast/slow separation value.
        Typical cuts
        ------------
        ["tof", "between", [proton_tof_cut, proton_tof_cut + window]]

        """
        sel = cls(name)
        for spec in cuts:
            sel.add(_parse_cut_spec(spec))
        return sel

    

def _parse_cut_spec(spec):
    """
    Parse a [variable, operator, value] triplet into a Cut object.
    The variable name is resolved via _VARIABLE_ALIASES, or auto-prefixed
    with 'vme_' if not found in the alias table.
    Supported operators: '>', '<', '>=', '<=', 'between'
    For 'between', value must be [low, high]: e.g. ["tof", "between", [20, 50]]
    """
    if not (isinstance(spec, (list, tuple)) and len(spec) == 3):
        raise ValueError(f"Cut spec must be [variable, operator, value], got: {spec!r}")
    var, op, value = spec
    col = _VARIABLE_ALIASES.get(var, f"vme_{var}" if not var.startswith("vme_") else var)
    if op == "between":
        if not (isinstance(value, (list, tuple)) and len(value) == 2):
            raise ValueError(f"'between' requires [low, high], got: {value!r}")
        low, high = value
        return Cut(
            name      = f"{var}_between_threshold",
            variable  = col,
            func      = Cut.between(low, high),
            threshold = (low, high),
            direction = "between",
        )
    if op == ">":
        func, direction = Cut.greater_than(value), "above"
    elif op == "<":
        func, direction = Cut.less_than(value), "below"
    elif op == "==":
        func, direction = Cut.equal_to(value), "equal"
    elif op == "!=":
        func, direction = Cut.not_equal_to(value), "not_equal"
    elif op == ">=":
        func, direction = (lambda v: lambda x: x >= v)(value), "above"
    elif op == "<=":
        func, direction = (lambda v: lambda x: x <= v)(value), "below"
    else:
        raise ValueError(f"Unsupported operator {op!r}. Use '>', '<', '>=', '<=', or 'between'.")
    return Cut(
        name      = f"{var}_{direction}_threshold",
        variable  = col,
        func      = func,
        threshold = value,
        direction = direction,
    )


_VARIABLE_ALIASES = {
    "act_eveto":    "vme_act_eveto",
    "act_tagger":   "vme_act_tagger",
    "tof":          "vme_tof_corr",
    "mu_tag_total": "vme_mu_tag_total",
    "act_0_charge": "vme_act0_l_charge",
    "T5_particle_nr": "T5_particle_nr",

}




_VARIABLE_UNITS = {
    "vme_act_eveto":    "PE",
    "vme_act_tagger":   "PE",
    "vme_tof_corr":     "ns",
    "vme_mu_tag_total": "a.u.",
    "vme_act0_l_charge": "PE",
    'vme_t0_time': "ns",
    'vme_t1_time': "ns",
    'vme_t4_time': "ns",
    'vme_t0_time_second_hit': "ns",
    'vme_t1_time_second_hit': "ns",
    'vme_t4_time_second_hit': "ns",
    'vme_time_t0_0': "ns",
    'vme_time_t0_1': "ns",
    'vme_time_t0_2': "ns",
    'vme_time_t0_3': "ns",
    'vme_time_t1_0': "ns",
    'vme_time_t1_1': "ns",
    'vme_time_t1_2': "ns",
    'vme_time_t1_3': "ns",
    'vme_time_t4_0': "ns",
    'vme_time_t4_1': "ns",
    'vme_t5_time': "ns",
    'vme_t4_l_time': "ns",
    'vme_t4_r_time': "ns",
    'vme_t4_l_second_hit': "ns",
    'vme_t4_r_second_hit': "ns",
    'vme_act0_l_charge': "PE",
    'vme_act1_l_charge': "PE",
    'vme_act2_l_charge': "PE",
    'vme_act3_l_charge': "PE",
    'vme_act4_l_charge': "PE",
    'vme_act5_l_charge': "PE",
    'vme_act0_r_charge': "PE",
    'vme_act1_r_charge': "PE",
    'vme_act2_r_charge': "PE",
    'vme_act3_r_charge': "PE",
    'vme_act4_r_charge': "PE",
    'vme_act5_r_charge': "PE",
    'vme_act0_l_time': "ns",
    'vme_act0_r_time': "ns",
    'vme_tof_t0t1': "ns",
    'vme_tof_t0t4': "ns",
    'vme_tof_t4t1': "ns",
    'vme_tof_t0t5': "ns",
    'vme_tof_t1t5': "ns",
    'vme_mu_tag_l_charge': "a.u.",
    'vme_mu_tag_r_charge': "a.u.",
    'T5_particle_nr': " ",
}


class SelectionMonitor:
    """
    Accumulates per-variable histograms across batches and shows live-updating
    plots with cut lines taken from the BeamSelection objects.

    Each plot shows all events in gray with each particle selection overlaid
    in a distinct colour. Cut lines are labelled to show whether the selection
    requires events to be above or below that threshold.

    Parameters
    ----------
    selections   : list of BeamSelection
    update_every : int       — redraw the plot every N calls to update() (default 10)
    bins         : int       — number of histogram bins (default 100)
    vme_run_info : optional  — awkward record from loader.get_vme_analysis_run_info();
                               if provided, adds run number, momentum and ACT cut values
                               to the figure title
    """

    def __init__(self, selections, update_every=10, bins=100, vme_run_info=None):
        self._selections   = selections
        self._update_every = update_every
        self._bins         = bins
        self._call_count   = 0
        self._run_info     = vme_run_info

        seen, self._variables = set(), []
        for sel in selections:
            for cut in sel.cuts:
                if cut.variable not in seen:
                    seen.add(cut.variable)
                    self._variables.append(cut.variable)

        self._counts     = {v: None for v in self._variables}
        self._edges      = {v: None for v in self._variables}
        self._sel_counts = {sel.name: {v: None for v in self._variables}
                            for sel in selections}
        self._total      = 0
        self._sel_total  = {sel.name: 0 for sel in selections}

    def _get_cut_threshold(self, variable):
        """Return the first non-None threshold found for this variable across all selections."""
        for sel in self._selections:
            for cut in sel.cuts:
                if cut.variable == variable and cut.threshold is not None:
                    return cut.threshold
        return None

    def update(self, batch):
        masked = {sel.name: batch[sel.mask(batch)] for sel in self._selections}
        self._total += len(batch)
        for sel_name, sel_batch in masked.items():
            self._sel_total[sel_name] += len(sel_batch)
        self._accumulate(batch, masked)
        self._call_count += 1
        if self._call_count % self._update_every == 0:
            self.show()

    def _accumulate(self, batch, masked_batches):
        for var in self._variables:
            if var not in ak.fields(batch):
                continue
            values = ak.to_numpy(batch[var])
            values = values[np.isfinite(values)]
            if len(values) == 0:
                continue
            if self._edges[var] is None:
                if var == "vme_tof_corr":
                    # For TOF, use the cut threshold to set the histogram range,
                    # so we can see the separation between fast and slow particles.
                
                    lo, hi = float(10.0), float(min(values.max()*1.8, 45))
                else:
                    lo, hi = float(values.min() * 0.8), float(values.max() * 1.2)
                
                if lo == hi:
                    lo, hi = lo - 1.0, hi + 1.0
                self._edges[var]  = np.linspace(lo, hi, self._bins + 1)
                self._counts[var] = np.zeros(self._bins, dtype=np.float64)
                for sel_name in self._sel_counts:
                    self._sel_counts[sel_name][var] = np.zeros(self._bins, dtype=np.float64)
            counts, _ = np.histogram(values, bins=self._edges[var])
            self._counts[var] += counts
            for sel_name, sel_batch in masked_batches.items():
                if var not in ak.fields(sel_batch):
                    continue
                sel_vals = ak.to_numpy(sel_batch[var])
                sel_vals = sel_vals[np.isfinite(sel_vals)]
                sel_counts, _ = np.histogram(sel_vals, bins=self._edges[var])
                self._sel_counts[sel_name][var] += sel_counts

    def _build_suptitle(self):
        parts = []
        if self._run_info is not None:
            parts.append(f"Run {int(self._run_info['run_number'])}")
            parts.append(f"p = {int(self._run_info['run_momentum'])} MeV/c")
            parts.append(f"n(act_eveto) = {float(self._run_info['n_eveto']):.3f}")
            parts.append(f"n(act_tagger) = {float(self._run_info['n_tagger']):.3f}")
        return "   |   ".join(parts)

    def show(self):
        n = len(self._variables)
        if n == 0:
            return
        fig, axes = plt.subplots(1, n, figsize=(5 * n, 4))
        if n == 1:
            axes = [axes]
        colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

        for ax, var in zip(axes, self._variables):
            counts, edges = self._counts[var], self._edges[var]
            if counts is None:
                ax.set_title(var)
                ax.text(0.5, 0.5, "no data yet", transform=ax.transAxes,
                        ha="center", va="center")
                continue

            unit    = _VARIABLE_UNITS.get(var, "")
            unit_str = f" {unit}" if unit else ""
            centers = 0.5 * (edges[:-1] + edges[1:])
            safe    = np.maximum(counts, 0.5)

            # Gray background: all events
            ax.fill_between(centers, safe, step="mid",
                            color="gray", alpha=0.25, label=f"all events  ({self._total:,})")
            ax.step(centers, safe, where="mid", color="gray", alpha=0.6, linewidth=1)

            # Coloured overlays: per-selection events
            for i, sel in enumerate(self._selections):
                color      = colors[i % len(colors)]
                sel_counts = self._sel_counts[sel.name][var]
                if sel_counts is not None:
                    n_sel  = self._sel_total[sel.name]
                    ax.step(centers, np.maximum(sel_counts, 0.5), where="mid",
                            color=color, linewidth=1.8, label=f"{sel.name}  ({n_sel:,})")

            # Cut lines with informative labels
            for i, sel in enumerate(self._selections):
                color = colors[i % len(colors)]
                for cut in sel.cuts:
                    if cut.variable != var or cut.threshold is None:
                        continue
                    if cut.direction == "between":
                        lo, hi = cut.threshold
                        ax.axvline(lo, color=color, linestyle="--", linewidth=1.2,
                                   label=f"{sel.name} between {lo:.2g}–{hi:.2g}{unit_str}")
                        ax.axvline(hi, color=color, linestyle="--", linewidth=1.2)
                    elif cut.direction == "above":
                        ax.axvline(cut.threshold, color=color, linestyle="--", linewidth=1.2,
                                   label=f"{sel.name} above {cut.threshold:.2g}{unit_str}")
                    elif cut.direction == "below":
                        ax.axvline(cut.threshold, color=color, linestyle="--", linewidth=1.2,
                                   label=f"{sel.name} below {cut.threshold:.2g}{unit_str}")

            ax.set_yscale("log")
            ax.set_xlabel(f"{var} [{unit}]" if unit else var)
            ax.set_ylabel("Counts")
            ax.set_title(var)
            ax.legend(fontsize=7, loc="best")

        suptitle = self._build_suptitle()
        if suptitle:
            fig.suptitle(suptitle, fontsize=14, weight="bold")
            fig.tight_layout(rect=[0, 0, 1, 0.93])
        else:
            fig.tight_layout()
        if _IPYTHON_AVAILABLE:
            _ipython_clear_output(wait=True)
        plt.show()
        plt.close(fig)


# Particle masses in MeV/c²
_PARTICLE_MASSES = {
    "electron": 0.511,
    "muon":     105.66,
    "pion":     139.57,
    "kaon":     493.68,
    "proton":   938.27,
    "deuteron": 1876.54,
    "helium3":  2808.39,
}


def print_cherenkov_thresholds(vme_run_info):
    """
    Print which particles are above the Cherenkov threshold in each ACT for this run.

    A particle produces light in an ACT only if its momentum exceeds the threshold
    p_threshold = mass / sqrt(n^2 - 1), where n is the refractive index of the radiator.
    This determines whether the ACT-based cuts in your selection are meaningful.

    Parameters
    ----------
    vme_run_info : awkward record from loader.get_vme_analysis_run_info()
                   must contain: run_momentum [MeV/c], n_eveto, n_tagger
    """
    momentum = abs(float(vme_run_info["run_momentum"]))
    n_eveto  = float(vme_run_info["n_eveto"])
    n_tagger = float(vme_run_info["n_tagger"])

    def cherenkov_threshold(mass, n):
        return mass / math.sqrt(n ** 2 - 1)

    print(f"Run momentum : {momentum:.0f} MeV/c")
    print(f"n (act_eveto)  = {n_eveto:.4f}   n (act_tagger) = {n_tagger:.4f}")
    print()
    header = f"{'Particle':<10} {'Mass [MeV]':>10}   {'Thresh. ACT eveto [MeV]':>22}  {'Above?':>6}   {'Thresh. ACT tagger [MeV]':>23}  {'Above?':>6}"
    print(header)
    print("-" * len(header))
    for name, mass in _PARTICLE_MASSES.items():
        thr_eveto  = cherenkov_threshold(mass, n_eveto)
        thr_tagger = cherenkov_threshold(mass, n_tagger)
        above_eveto  = "yes" if momentum > thr_eveto  else "no"
        above_tagger = "yes" if momentum > thr_tagger else "no"
        print(f"{name:<10} {mass:>10.3f}   {thr_eveto:>22.1f}  {above_eveto:>6}   {thr_tagger:>23.1f}  {above_tagger:>6}")
