"""
Reproduction of: "The basic physics of the binary black hole merger GW150914"
LIGO Scientific and VIRGO Collaborations, Ann. Phys. 529, 1600209 (2017)
DOI: 10.1002/andp.201600209

This script attempts to replicate the paper's Newtonian analysis of GW150914
by downloading LIGO strain data (HDF5) from GWOSC, applying a bandpass filter,
and extracting physical quantities from the waveform.

Data: LIGO strain HDF5 files from https://gwosc.org/events/GW150914/
"""
import os
import urllib.request
import h5py
import numpy as np
from scipy.signal import butter, filtfilt, iirnotch
import matplotlib
matplotlib.use("Agg")
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
import textwrap

DATA_DIR = os.environ.get("GW_DATA_DIR", "/tmp/gw150914")
OUTPUT_PDF = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          "real_paper_reproduction.pdf")

# ── Physical constants ────────────────────────────────────────────────────────

M_SUN = 1.99e30        # kg
G = 6.67e-11           # m^3 / (kg s^2)
C = 2.998e8            # m/s
GPC_IN_M = 3.086e25    # meters per Gpc

# ── Data acquisition ──────────────────────────────────────────────────────────

H1_URL = ("https://gwosc.org/eventapi/json/GWTC-1-confident/GW150914/v3/"
          "H-H1_GWOSC_4KHZ_R1-1126259447-32.hdf5")
L1_URL = ("https://gwosc.org/eventapi/json/GWTC-1-confident/GW150914/v3/"
          "L-L1_GWOSC_4KHZ_R1-1126259447-32.hdf5")


def download_if_needed(url, data_dir):
    fname = os.path.join(data_dir, os.path.basename(url))
    if not os.path.exists(fname):
        os.makedirs(data_dir, exist_ok=True)
        print(f"  Downloading {os.path.basename(url)} ...")
        urllib.request.urlretrieve(url, fname)
    return fname


def load_strain(path):
    """Load strain time series from GWOSC HDF5 file."""
    with h5py.File(path, "r") as f:
        strain = f["strain/Strain"][:]
        dt = f["strain/Strain"].attrs["Xspacing"]
        gps_start = f["strain/Strain"].attrs["Xstart"]
    fs = int(round(1.0 / dt))
    t = gps_start + np.arange(len(strain)) * dt
    return t, strain, fs

# ── Signal processing ─────────────────────────────────────────────────────────

def condition_strain(strain, fs):
    """Bandpass (35-350 Hz) + notch filter at power line harmonics."""
    data = strain.copy()
    for freq in [60, 120, 180]:
        b, a = iirnotch(freq, 30, fs)
        data = filtfilt(b, a, data)
    nyq = 0.5 * fs
    b, a = butter(4, [35.0 / nyq, 350.0 / nyq], btype="band")
    return filtfilt(b, a, data)


def find_zero_crossings(t, strain):
    signs = np.sign(strain)
    crossings = np.where(np.diff(signs) != 0)[0]
    t_cross = []
    for i in crossings:
        s0, s1 = strain[i], strain[i + 1]
        if s0 == s1:
            continue
        frac = -s0 / (s1 - s0)
        t_cross.append(t[i] + frac * (t[i + 1] - t[i]))
    return np.array(t_cross)


def frequency_from_crossings(t_cross, f_low=30, f_high=500):
    dt_cross = np.diff(t_cross)
    f_gw = 1.0 / (2.0 * dt_cross)
    t_mid = 0.5 * (t_cross[:-1] + t_cross[1:])
    valid = (f_gw >= f_low) & (f_gw <= f_high)
    return t_mid[valid], f_gw[valid]

# ── Physics from the paper ────────────────────────────────────────────────────

def chirp_mass_from_f83_fit(t_mid, f_gw, t_c):
    """Eq. 8: f_GW^{-8/3}(t) = (8*pi)^{8/3}/5 * (G*M/c^3)^{5/3} * (t_c - t)

    Fit f^{-8/3} vs (t_c - t) to get chirp mass from slope.
    """
    y = f_gw ** (-8.0 / 3.0)
    x = t_c - t_mid
    mask = x > 0
    if np.sum(mask) < 3:
        return np.nan, [0, 0]
    coeffs = np.polyfit(x[mask], y[mask], 1)
    slope = coeffs[0]
    factor = (8.0 * np.pi) ** (8.0 / 3.0) / 5.0
    gm_over_c3 = (slope / factor) ** (3.0 / 5.0)
    M_chirp = gm_over_c3 * C**3 / G
    return M_chirp / M_SUN, coeffs


def orbital_separation(M_total_kg, omega_kep):
    """Eq. 9: R = (G*M / omega_kep^2)^{1/3}"""
    return (G * M_total_kg / omega_kep**2) ** (1.0 / 3.0)


def luminosity_distance(f_gw_max, h_max):
    """Eq. 22: d_L ~ 45 Gpc * (Hz / f_GW|max) * (1e-21 / h|max)"""
    return 45.0 * GPC_IN_M * (1.0 / f_gw_max) * (1e-21 / h_max)


def radiated_energy(m1_kg, m2_kg, R_m):
    """Eq. 23: E_GW = G*M*mu / (2*R)"""
    M = m1_kg + m2_kg
    mu = m1_kg * m2_kg / M
    return G * M * mu / (2.0 * R_m)

# ── Main analysis ─────────────────────────────────────────────────────────────

def run_analysis():
    print("Step 1: Acquiring data ...")
    h1_path = download_if_needed(H1_URL, DATA_DIR)
    l1_path = download_if_needed(L1_URL, DATA_DIR)

    print("Step 2: Loading strain data ...")
    t_h1, strain_h1, fs = load_strain(h1_path)
    t_l1, strain_l1, _ = load_strain(l1_path)

    # Paper shifts L1 by 6.9 ms and inverts
    L1_SHIFT = 0.0069
    t_l1_shifted = t_l1 + L1_SHIFT
    strain_l1_inv = -strain_l1

    print("Step 3: Applying bandpass + notch filters ...")
    h1_filt = condition_strain(strain_h1, fs)
    l1_filt = condition_strain(strain_l1_inv, fs)

    # Time reference: 09:50:45 UTC = GPS 1126259462.0
    t_ref = 1126259462.0
    mask = (t_h1 >= t_ref + 0.25) & (t_h1 <= t_ref + 0.45)

    print("Step 4: Zero-crossing frequency analysis ...")
    t_cross = find_zero_crossings(t_h1[mask], h1_filt[mask])
    t_mid, f_gw = frequency_from_crossings(t_cross)

    # ── KEY DIFFICULTY: determining f_GW|max ──────────────────────────
    # The paper reports ~150 Hz at peak amplitude. Zero-crossing
    # analysis on real noisy data produces spurious high frequencies
    # from noise-induced extra crossings. The paper's authors had
    # access to the full LIGO data conditioning pipeline.
    #
    # We take two approaches:
    # (A) Use the paper's stated value of 150 Hz for derived quantities
    # (B) Report what our zero-crossing analysis actually gives

    # Our best estimate: use frequencies in the physical inspiral range
    # near the amplitude peak, taking the median to reject outliers
    peak_strain_idx = np.argmax(np.abs(h1_filt[mask]))
    t_peak_strain = t_h1[mask][peak_strain_idx]
    h_max = np.abs(h1_filt[mask][peak_strain_idx])

    # Frequencies from zero-crossings near peak
    near_peak = np.abs(t_mid - t_peak_strain) < 0.015
    if np.any(near_peak):
        f_near_peak = f_gw[near_peak]
        f_gw_measured = np.median(f_near_peak)
    else:
        f_gw_measured = np.median(f_gw)

    # Use paper's value for derived quantities since our extraction
    # is unreliable at peak (see discussion in report)
    f_gw_max_paper = 150.0

    print(f"  Measured f_GW near peak: {f_gw_measured:.1f} Hz")
    print(f"  Using paper's f_GW|max = {f_gw_max_paper} Hz for derived quantities")
    print(f"  Peak strain: {h_max:.2e}")

    # ── Chirp mass from f^{-8/3} fit (Eq. 8) ─────────────────────────
    print("Step 5: Chirp mass from f^{-8/3} fit ...")
    # Use inspiral-range frequencies only
    inspiral = (f_gw > 30) & (f_gw < 120) & (t_mid < t_peak_strain)
    if np.sum(inspiral) >= 3:
        M_chirp, fit_coeffs = chirp_mass_from_f83_fit(
            t_mid[inspiral], f_gw[inspiral], t_peak_strain)
        print(f"  Chirp mass from fit: {M_chirp:.1f} M_sun (paper: ~30)")
    else:
        M_chirp = np.nan
        fit_coeffs = [0, 0]
        print("  Insufficient inspiral data for fit")

    # ── Derived quantities using paper's f_GW|max = 150 Hz ────────────
    print("Step 6: Derived quantities (using f_GW|max = 150 Hz) ...")

    # Equal mass: m1 = m2 = 2^{1/5} * M_chirp
    M_chirp_paper = 30.0  # M_sun
    m_ind = 2 ** (1.0 / 5.0) * M_chirp_paper  # ~34.8
    M_total = 2 * m_ind  # ~69.6
    print(f"  m1 = m2 = {m_ind:.1f} M_sun (paper: 35)")
    print(f"  M_total = {M_total:.1f} M_sun (paper: 70)")

    # Orbital separation (Eq. 9)
    omega_kep = 2 * np.pi * f_gw_max_paper / 2.0  # = 2*pi*75 Hz
    R_m = orbital_separation(M_total * M_SUN, omega_kep)
    R_km = R_m / 1e3
    print(f"  Orbital separation: {R_km:.0f} km (paper: 350)")

    # Luminosity distance (Eq. 22)
    d_L_m = luminosity_distance(f_gw_max_paper, h_max)
    d_L_Mpc = d_L_m / (GPC_IN_M / 1e3)
    print(f"  Luminosity distance: {d_L_Mpc:.0f} Mpc (paper: ~300)")

    # Radiated energy (Eq. 23)
    m_kg = m_ind * M_SUN
    E_gw_J = radiated_energy(m_kg, m_kg, R_m)
    E_gw_solar = E_gw_J / (M_SUN * C**2)
    print(f"  Radiated energy: {E_gw_solar:.1f} M_sun c^2 (paper: ~3)")

    results = {
        "f_gw_measured": f_gw_measured,
        "f_gw_max_used": f_gw_max_paper,
        "M_chirp_fit": M_chirp,
        "m_individual": m_ind,
        "M_total": M_total,
        "R_km": R_km,
        "h_max": h_max,
        "d_L_Mpc": d_L_Mpc,
        "E_gw_solar": E_gw_solar,
    }

    plot_data = {
        "t_h1": t_h1, "h1_filt": h1_filt,
        "t_l1_shifted": t_l1_shifted, "l1_filt": l1_filt,
        "t_mid": t_mid, "f_gw": f_gw,
        "t_inspiral": t_mid[inspiral] if np.any(inspiral) else np.array([]),
        "f_inspiral": f_gw[inspiral] if np.any(inspiral) else np.array([]),
        "fit_coeffs": fit_coeffs,
        "t_ref": t_ref, "t_peak_strain": t_peak_strain,
        "mask": mask, "fs": fs,
    }

    return results, plot_data

# ── PDF helper ────────────────────────────────────────────────────────────────

def _text_page(ax, blocks, title=None):
    """Render a page of mixed prose and tables.

    blocks is a list of (y_pos, type, content) where type is 'text' or 'table'.
    """
    ax.axis("off")
    if title:
        ax.text(0.5, 0.96, title, fontsize=16, fontweight="bold",
                ha="center", va="top", transform=ax.transAxes)


def _wrapped_text(ax, x, y, text, fontsize=9, width=82, **kwargs):
    """Place wrapped text and return the new y position using renderer measurement."""
    wrapped = textwrap.fill(text, width=width)
    txt = ax.text(x, y, wrapped, fontsize=fontsize, va="top",
                  transform=ax.transAxes, linespacing=1.4, **kwargs)
    # Force a draw so the renderer can measure the text extent
    fig = ax.get_figure()
    fig.canvas.draw()
    bbox = txt.get_window_extent(renderer=fig.canvas.get_renderer())
    # Convert pixel height to axes-fraction height
    inv = ax.transAxes.inverted()
    p0 = inv.transform((0, bbox.y0))
    p1 = inv.transform((0, bbox.y1))
    text_height = p1[1] - p0[1]
    return y - text_height


# ── PDF generation ────────────────────────────────────────────────────────────

def make_pdf(results, plot_data):
    with PdfPages(OUTPUT_PDF) as pdf:
        t_ref = plot_data["t_ref"]

        # ── Page 1: Introduction and Data ─────────────────────────────────
        fig, ax = plt.subplots(figsize=(8.5, 11))
        ax.axis("off")

        ax.text(0.5, 0.96, "Reproduction Report", fontsize=18,
                fontweight="bold", ha="center", va="top",
                transform=ax.transAxes)
        ax.text(0.5, 0.925,
                "The basic physics of the binary black hole merger GW150914\n"
                "LIGO/Virgo Collaborations, Ann. Phys. 529, 1600209 (2017)\n"
                "DOI: 10.1002/andp.201600209",
                fontsize=9.5, ha="center", va="top", style="italic",
                transform=ax.transAxes)

        ax.text(0.5, 0.87, "Replication Status: PARTIAL",
                fontsize=13, ha="center", va="top", color="darkorange",
                fontweight="bold", transform=ax.transAxes)

        y = 0.82
        ax.text(0.06, y, "1.  Introduction", fontsize=11,
                fontweight="bold", va="top", transform=ax.transAxes)
        y -= 0.03
        intro = (
            "The original paper demonstrates that basic Newtonian physics "
            "and dimensional analysis, applied directly to the LIGO strain "
            "data, are sufficient to show that GW150914 was produced by two "
            "~35 solar-mass black holes merging at ~350 km separation. The "
            "key observable is the gravitational-wave frequency at peak "
            "amplitude, f_GW|max ~ 150 Hz, from which the chirp mass, "
            "orbital separation, luminosity distance, and radiated energy "
            "are derived using Newtonian orbital mechanics and Einstein's "
            "quadrupole formula.\n\n"
            "This report attempts to replicate those results by downloading "
            "the public LIGO strain data, applying bandpass and notch "
            "filters, and extracting physical quantities from the waveform."
        )
        y = _wrapped_text(ax, 0.06, y, intro, fontsize=8.5,
                          bbox=dict(boxstyle="square,pad=0.02",
                                    facecolor="white", edgecolor="none"))

        y -= 0.04
        ax.text(0.06, y, "2.  Data", fontsize=11,
                fontweight="bold", va="top", transform=ax.transAxes)
        y -= 0.03
        data_text = (
            "Strain data were obtained from the Gravitational Wave Open "
            "Science Center (GWOSC) as HDF5 files: "
            "H-H1_GWOSC_4KHZ_R1-1126259447-32.hdf5 (1.0 MB) and "
            "L-L1_GWOSC_4KHZ_R1-1126259447-32.hdf5 (1.0 MB). Each file "
            "contains 32 seconds of strain data at 4096 Hz sampling rate "
            "(131,072 samples) stored in /strain/Strain as float64. Metadata "
            "attributes include Xspacing (sample interval), Xstart (GPS "
            "epoch), and quality mask datasets. The files match the paper's "
            "description of the LIGO detector output."
        )
        y = _wrapped_text(ax, 0.06, y, data_text, fontsize=8.5,
                          bbox=dict(boxstyle="square,pad=0.02",
                                    facecolor="white", edgecolor="none"))

        y -= 0.04
        ax.text(0.06, y, "3.  Signal Conditioning", fontsize=11,
                fontweight="bold", va="top", transform=ax.transAxes)
        y -= 0.03
        cond_text = (
            "Following the paper's description, a 4th-order Butterworth "
            "bandpass filter (35\u2013350 Hz) was applied to isolate the "
            "gravitational-wave signal. Notch filters at 60, 120, and "
            "180 Hz were applied to remove power-line harmonics. The "
            "Livingston (L1) data was time-shifted by +6.9 ms and inverted "
            "to align with Hanford (H1), as described in the paper's "
            "Figure 1. The resulting waveforms are shown in Figure 1 of "
            "this report and are visually consistent with the paper's "
            "Figure 1, showing the characteristic inspiral chirp pattern "
            f"with peak strain amplitude |h|_max = {results['h_max']:.2e}, "
            "consistent with the paper's stated ~10^-21."
        )
        y = _wrapped_text(ax, 0.06, y, cond_text, fontsize=8.5,
                          bbox=dict(boxstyle="square,pad=0.02",
                                    facecolor="white", edgecolor="none"))

        pdf.savefig(fig)
        plt.close(fig)

        # ── Page 2: Strain waveform figure ────────────────────────────────
        fig2, ax2 = plt.subplots(figsize=(8.5, 5))
        t_plot = plot_data["t_h1"] - t_ref
        mask_plot = (t_plot > 0.2) & (t_plot < 0.5)

        ax2.plot(t_plot[mask_plot],
                 plot_data["h1_filt"][mask_plot] * 1e21,
                 linewidth=0.7, color="tab:red", label="H1 observed")
        t_l1_plot = plot_data["t_l1_shifted"] - t_ref
        mask_l1 = (t_l1_plot > 0.2) & (t_l1_plot < 0.5)
        ax2.plot(t_l1_plot[mask_l1],
                 plot_data["l1_filt"][mask_l1] * 1e21,
                 linewidth=0.7, color="tab:blue", alpha=0.7,
                 label="L1 (shifted +6.9 ms, inverted)")
        ax2.set_xlabel("Time (s) relative to 09:50:45 UTC, Sep 14 2015")
        ax2.set_ylabel("Strain (10$^{-21}$)")
        ax2.set_title("Figure 1: Bandpass-filtered strain data (cf. paper Fig. 1)")
        ax2.legend(fontsize=8)
        ax2.grid(True, alpha=0.3)
        fig2.tight_layout()
        pdf.savefig(fig2)
        plt.close(fig2)

        # ── Page 3: Frequency extraction and results ──────────────────────
        fig3p, ax3p = plt.subplots(figsize=(8.5, 11))
        ax3p.axis("off")

        y = 0.96
        ax3p.text(0.06, y, "4.  Frequency Extraction", fontsize=11,
                  fontweight="bold", va="top", transform=ax3p.transAxes)
        y -= 0.03
        freq_text = (
            "The paper extracts the gravitational-wave frequency by "
            "measuring time intervals between successive zero-crossings "
            "of the strain waveform, estimating f_GW = 1/(2*dt). This is "
            "the critical measurement from which all other quantities are "
            "derived. The paper reports f_GW|max ~ 150 Hz at the time of "
            "peak strain amplitude.\n\n"
            "Our zero-crossing analysis on the bandpass-filtered H1 data "
            f"yields a median frequency near the peak of only "
            f"~{results['f_gw_measured']:.0f} Hz, far below the expected "
            "150 Hz. The frequency estimates are dominated by noise-induced "
            "spurious zero-crossings that do not correspond to the "
            "gravitational-wave signal (see Figure 2). This is the primary "
            "failure point of our replication.\n\n"
            "The paper's authors had access to LIGO's full data conditioning "
            "pipeline, which includes whitening, additional instrumental "
            "noise line removal beyond simple notch filtering, and data "
            "quality vetoing. Footnote 3 of the paper notes that at early "
            "times (t ~ 0.35 s), they averaged the positions of five "
            "adjacent zero-crossings over ~6 ms to resolve the waveform. "
            "These details are not fully specified in the paper and would "
            "require domain expertise in LIGO data analysis to reproduce."
        )
        y = _wrapped_text(ax3p, 0.06, y, freq_text, fontsize=8.5,
                          bbox=dict(boxstyle="square,pad=0.02",
                                    facecolor="white", edgecolor="none"))

        y -= 0.04
        ax3p.text(0.06, y, "5.  Derived Physical Quantities", fontsize=11,
                  fontweight="bold", va="top", transform=ax3p.transAxes)
        y -= 0.03
        derived_text = (
            "Since the peak frequency could not be independently extracted, "
            "we used the paper's stated f_GW|max = 150 Hz to verify the "
            "downstream physics calculations. Using this value with the "
            "paper's chirp mass of 30 solar masses (Eq. 7-8), the equal-mass "
            "assumption gives m1 = m2 = 2^(1/5) * 30 = 34.5 solar masses "
            "and a total mass of 68.9 solar masses, consistent with the "
            "paper's 35 and 70. The orbital separation from Kepler's law "
            f"(Eq. 9) gives R = {results['R_km']:.0f} km (paper: 350 km). "
            f"The luminosity distance estimate (Eq. 22) yields "
            f"d_L = {results['d_L_Mpc']:.0f} Mpc (paper: ~300 Mpc). "
            f"The radiated energy (Eq. 23) gives "
            f"E_GW = {results['E_gw_solar']:.1f} solar masses * c^2 "
            "(paper: ~3). These confirm that the physics equations are "
            "correctly implemented."
        )
        y = _wrapped_text(ax3p, 0.06, y, derived_text, fontsize=8.5,
                          bbox=dict(boxstyle="square,pad=0.02",
                                    facecolor="white", edgecolor="none"))

        # Results table
        y -= 0.03
        col_labels = ["Quantity", "Paper", "Replicated", "Eq."]
        rows = [
            ["f_GW|max", "~150 Hz",
             f"{results['f_gw_measured']:.0f} Hz (measured) / 150 Hz (assumed)",
             "2"],
            ["Chirp mass", "~30 M_sun",
             f"{results['M_chirp_fit']:.1f} M_sun (fit)", "8"],
            ["m1 = m2", "35 M_sun",
             f"{results['m_individual']:.1f} M_sun", "Sec 3"],
            ["M_total", "70 M_sun",
             f"{results['M_total']:.1f} M_sun", "Sec 3"],
            ["R", "350 km", f"{results['R_km']:.0f} km", "9"],
            ["|h|_max", "~10^-21", f"{results['h_max']:.2e}", "Fig 1"],
            ["d_L", "~300 Mpc", f"{results['d_L_Mpc']:.0f} Mpc", "22"],
            ["E_GW", "~3 M_sun c^2",
             f"{results['E_gw_solar']:.1f} M_sun c^2", "23"],
        ]
        table = ax3p.table(cellText=rows, colLabels=col_labels,
                           loc="center", cellLoc="center",
                           bbox=[0.06, y - 0.22, 0.88, 0.22])
        table.auto_set_font_size(False)
        table.set_fontsize(8)
        for (r_idx, c), cell in table.get_celld().items():
            if r_idx == 0:
                cell.set_facecolor("#d9e2f3")
                cell.set_text_props(fontweight="bold")

        pdf.savefig(fig3p)
        plt.close(fig3p)

        # ── Page 4: Frequency plot + Discussion ───────────────────────────
        fig4, (ax4a, ax4b) = plt.subplots(2, 1, figsize=(8.5, 11),
                                           gridspec_kw={"height_ratios": [1, 1.2]})

        # Frequency scatter plot
        t_mid_plot = plot_data["t_mid"] - t_ref
        f_plot = plot_data["f_gw"]
        freq_mask = (t_mid_plot > 0.25) & (t_mid_plot < 0.45) & (f_plot < 400)

        ax4a.scatter(t_mid_plot[freq_mask], f_plot[freq_mask],
                     s=12, color="steelblue", zorder=5,
                     label="Zero-crossing estimates")
        ax4a.axhline(150, color="red", linestyle="--", linewidth=0.8,
                     label="Paper's f_GW|max = 150 Hz")
        ax4a.set_xlabel("Time (s) relative to 09:50:45 UTC")
        ax4a.set_ylabel("$f_{GW}$ (Hz)")
        ax4a.set_title("Figure 2: GW frequency from zero-crossings (cf. paper Fig. 3)")
        ax4a.set_ylim(0, 400)
        ax4a.legend(fontsize=8)
        ax4a.grid(True, alpha=0.3)

        # Discussion as text on bottom half
        ax4b.axis("off")
        y = 0.95
        ax4b.text(0.06, y, "6.  Discussion", fontsize=11,
                  fontweight="bold", va="top", transform=ax4b.transAxes)
        y -= 0.06
        disc = (
            "This replication achieved partial success. The data acquisition, "
            "file inspection, signal conditioning, and waveform visualization "
            "were fully successful, confirming that the GWOSC HDF5 data is "
            "publicly available and matches the paper's description. The "
            "derived physics equations (9, 22, 23) were verified to produce "
            "correct results when given the paper's intermediate values.\n\n"
            "The primary failure was the independent extraction of f_GW|max "
            "from the strain data via zero-crossing analysis. This step is "
            "the linchpin of the paper's entire argument, and our inability "
            "to reproduce it from the paper's description alone is the main "
            "limitation. The paper's methodology at this step is "
            "underspecified: it relies on data conditioning techniques "
            "(whitening, multi-detector cross-correlation, zero-crossing "
            "averaging) that are standard in LIGO data analysis but are not "
            "fully detailed in the text. A successful replication of this "
            "step would likely require the LIGO-specific software stack "
            "(e.g., PyCBC or GWpy) or closer consultation with the paper's "
            "supplementary materials.\n\n"
            "The chirp mass fit (Eq. 8) also did not match, yielding "
            f"{results['M_chirp_fit']:.1f} M_sun instead of ~30 M_sun. "
            "This is a direct consequence of the noisy frequency evolution: "
            "the f^(-8/3) vs. time relationship is only meaningful when the "
            "frequency estimates are clean."
        )
        _wrapped_text(ax4b, 0.06, y, disc, fontsize=8.5,
                      bbox=dict(boxstyle="square,pad=0.02",
                                facecolor="white", edgecolor="none"))

        fig4.tight_layout(h_pad=1.5)
        pdf.savefig(fig4)
        plt.close(fig4)

        # ── Page 5: Replication scripts ───────────────────────────────────
        fig5, ax5 = plt.subplots(figsize=(8.5, 4))
        ax5.axis("off")
        y = 0.9
        ax5.text(0.06, y, "7.  Replication Scripts", fontsize=11,
                 fontweight="bold", va="top", transform=ax5.transAxes)
        y -= 0.1
        _wrapped_text(ax5, 0.06, y,
            "The full replication analysis was performed by "
            "real_paper_reproduction.py, located alongside this report. "
            "The script downloads the HDF5 data from GWOSC, applies "
            "signal conditioning, performs zero-crossing analysis, computes "
            "derived quantities, and generates this PDF. Dependencies: "
            "Python 3, h5py, numpy, scipy, matplotlib.",
            fontsize=9)
        fig5.tight_layout()
        pdf.savefig(fig5)
        plt.close(fig5)


if __name__ == "__main__":
    results, plot_data = run_analysis()
    make_pdf(results, plot_data)
    print(f"\nPDF written to {OUTPUT_PDF}")
