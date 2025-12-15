# Sgr A* Dynamics Evaluation Scripts Documentation

This document provides a detailed breakdown of the evaluation scripts used in the Sgr A* Dynamics project. It includes step-by-step mathematical formulations, processing logic, and the specific thresholds used for pass/fail evaluation.

---

## 1. `src/chisq.py`

### Goal
To assess the goodness-of-fit of the reconstruction against the observed visibility data using $\chi^2$ statistics for closure quantities and polarization ratios.

### Processing Steps
1.  **Data Preparation**:
    *   Load UVFITS observation data.
    *   **Flagging**: Automatically flags AA (ALMA) and AP (APEX) baselines (unless disabled) to avoid calibration errors dominating the metric.
    *   **Time Alignment**: Observation times $t_{obs}$ are matched to movie frames $t_{mov}$ (nearest neighbor interpolation).

2.  **Metric Calculation** (for each time $t$):
    *   **Closure Phase ($\chi^2_{cp}$)**:
        $$ \chi^2_{cp} = \frac{1}{N_{cp}} \sum_{i,j,k} \frac{(\Phi_{ijk}^{obs} - \Phi_{ijk}^{model})^2}{\sigma_{\Phi, ijk}^2} $$
    *   **Log Closure Amplitude ($\chi^2_{lca}$)**:
        $$ A_{ijkl} = \ln \left( \frac{|V_{ij} V_{kl}|}{|V_{ik} V_{jl}|} \right) $$
        $$ \chi^2_{lca} = \frac{1}{N_{lca}} \sum \frac{(A^{obs} - A^{model})^2}{\sigma_{A}^2} $$
    *   **Polarization Ratio ($\chi^2_{m}$)**:
        Uses the "m-breve" fraction $\breve{m} = \frac{\mathcal{Q} + i\mathcal{U}}{\mathcal{I}}$.
        $$ \chi^2_{m} = \frac{1}{N_{vis}} \sum \frac{|\breve{m}^{obs} - \breve{m}^{model}|^2}{\sigma_{\breve{m}}^2} $$
        *Note*: JC (JCMT) sites are flagged for polarization metrics due to known calibration issues.

3.  **Bayesian Statistics**:
    If multiple reconstructions (samples) are provided:
    *   Mean: $\mu(t) = \frac{1}{N_s} \sum_{s} \chi^2_s(t)$
    *   Std: $\sigma(t) = \sqrt{\frac{1}{N_s} \sum_{s} (\chi^2_s(t) - \mu(t))^2}$

### Thresholds
*   **Visual Benchmark**: A horizontal line at $\chi^2 = 1$ is plotted as the standard for a "good" fit (reduced $\chi^2 \approx 1$).
*   There is no binary Pass/Fail output for this script.

---

## 2. `src/hotspot.py`

### Goal
To extract and track the brightest dynamic feature ("hotspot") in the image sequence and compare its trajectory and properties with the ground truth.

### Processing Steps
1.  **Isolate Component**:
    $$ I_{dynamic}(\mathbf{x}, t) = \max(0, I(\mathbf{x}, t) - I_{median}(\mathbf{x})) $$
    Negative residuals are clipped to 0.

2.  **Gaussian Fitting**:
    For each frame, fit a 2D Gaussian $G(x,y)$ to the location of the maximum pixel intensity.
    $$ G(x,y) = A \cdot \exp\left(-\frac{(x-x_0)^2 + (y-y_0)^2}{2\sigma^2}\right) $$
    *(Simplified isotropic form for illustration; script uses astropy's Gaussian2D)*

3.  **Derived Quantities**:
    *   **Position**: $(x, y) = (x_0, y_0)$ [$\mu$as]
    *   **Distance**: $r = \sqrt{x^2 + y^2}$ [$\mu$as]
    *   **Angle**: $\theta = (\text{degrees}(\text{angle}(-x - iy)) + 90) \% 360 - 180$  (Converts image coords to Sky PA East of North).
    *   **Flux**: Volume under the Gaussian.

4.  **Bayesian Aggregation**:
    Compute Mean $\mu_{rec}$ and Std $\sigma_{rec}$ across all reconstruction samples.

### Thresholds & Pass Criteria
The reconstruction **PASSES** a metric if the reconstruction interval overlaps with the Truth tolerance interval.

**Condition**:
$$ [\mu_{rec} - \sigma_{rec}, \mu_{rec} + \sigma_{rec}] \cap [\text{Truth} - \delta, \text{Truth} + \delta] \neq \emptyset $$

**Defined Tolerances ($\delta$)**:
*   **Position ($x, y$)**: $\pm 5.0\, \mu\text{as}$
*   **Distance**: $\pm 5.0\, \mu\text{as}$
*   **FWHM**: $\pm 5.0\, \mu\text{as}$
*   **Angle (PA)**: $\pm 20.0^\circ$
*   **Flux**: $\pm 25\%$ (Relative fraction of Truth flux)

---

## 3. `src/nxcorr.py`

### Goal
To quantify image fidelity using Normalized Cross-Correlation (NXCORR) for scalar ($I, P_{mag}$) and vector ($P, X$) quantities, weighted by observation quality.

### Processing Steps
1.  **Regridding**: Reconstruction is regridded to match Truth ($200 \times 200$ pixels, $200 \mu\text{as}$ FOV).

2.  **Mode Handling**:
    *   **Static**: Use Median image of the set. Shift to align centers using `align_images`.
    *   **Dynamic**: Subtract median (static) from each frame. Shift total frame to align, then apply same shift to dynamic residuals.
    *   **Total**: Use raw frames. No shift (assume absolute astrometry matches) or shift if requested.

3.  **Metric Calculation**:
    *   **Scalar ($I, P_{mag}$)**: Standard NXCORR.
    *   **Vector ($P_{vec}$)** (Cosine Similarity):
        $$ \text{NXCORR}_{vec} = \text{Re}\left[ \frac{\sum P_{rec} P_{truth}^*}{\sqrt{\sum |P_{rec}|^2} \sqrt{\sum |P_{truth}|^2}} \right] $$
    *   **Cross-Pol ($X$)** (Centered Pearson):
        $$ P'_{rec} = P_{rec} - \langle P_{rec} \rangle $$
        $$ \text{NXCORR}_{X} = \text{Re}\left[ \frac{\sum P'_{rec} (P'_{truth})^*}{\sqrt{\sum |P'_{rec}|^2} \sqrt{\sum |P'_{truth}|^2}} \right] $$
        Note: The Real part penalizes EVPA misalignment.

4.  **Weighting ($w(t)$)**:
    Weights are calculated per time-step based on observation quality:
    *   **Isotropy ($\mathcal{S}_{iso}$)**: Measures (u,v) coverage uniformity (Ramesh metric).
    *   **SNR**: Mean Signal-to-Noise ratio of visibilities.
    *   **Weight**: $w(t) \propto \mathcal{S}_{iso}(t) \times \text{SNR}(t)$.

### Thresholds & Pass Criteria
*   **Scalar ($I, P_{mag}$)**:
    *   **Adaptive Threshold ($\tau$)**:
        Calculated by comparing the Truth image ($I_{truth}$) with a **blurred version of itself** ($I_{blur}$).
        $$ \tau = \text{NXCORR}(I_{truth}, I_{blur}) $$
        *Blur kernel*: Gaussian fit to the dirty beam of the observation at that time.
        *Concept*: The reconstruction cannot be expected to be better than the "resolution limit" of the observation.

*   **Vector (`Pvec`, `X`) Threshold**:
    For vector quantities, the correlation is sensitive to the EVPA alignment. The threshold represents the correlation drop expected from a specific EVPA rotation error limit ($\chi_{rot} = 20^\circ$).
    $$ \tau_{vec} = \cos(2 \cdot \chi_{rot}) $$
    For $\chi_{rot} = 20^\circ$, $\tau_{vec} = \cos(40^\circ) \approx 0.766$.

*   **Pass Condition**:
    $$ \text{NXCORR}(t) > \tau(t) $$
    *(For Bayesian: Pass if $(\mu_{nxcorr} + \sigma_{nxcorr}) > \tau$)*.

---

## 4. `src/rex.py` / `src/mean_image_extraction.py`

### Goal
To extract morphological (Ring) and polarimetric quantities from the images using the **REx** algorithm.

### Processing Steps
1.  **Center Finding**: Use `findCenter` to locate the ring centroid.
2.  **Polar Unwrapping**: Interpolate image onto a $(r, \phi)$ grid.
3.  **Ring Fitting**:
    *   Find radial peak $R_{peak}(\phi)$ for every angle.
    *   **Diameter**: $D = 2 \langle R_{peak} \rangle$.
    *   **Width**: $W = \langle R_{out} - R_{in} \rangle$ (FWHM of radial profile).
    *   **Asymmetry**: First Fourier mode amplitude $A$.
4.  **Polarization Integration**:
    *   Integrate $Q, U$ over the ring annulus.
    *   Compute $\beta_2$ (m=2 azimuthal mode of polarization).

### Thresholds & Pass Criteria
The script compares Reconstruction vs Truth. A metric passes if $|Rec - Truth| - \sigma_{rec} \le \text{Threshold}$.

**Absolute Thresholds**:
*   **Diameter ($D$)**: $5.0\, \mu\text{as}$
*   **Width ($W$)**: $5.0\, \mu\text{as}$
*   **Peak PA**: $26.0^\circ$
*   **$\beta_2$ Angle**: $26.0^\circ$

**Relative Thresholds**:
*   **$A, m_{net}, m_{avg}, |v|_{net}, \text{Fluxes}$**: $10\%$ of the Truth value.

**Dynamic Threshold ($PA_{ori}$)**:
The position angle threshold scales with asymmetry (harder to define PA for symmetric rings).
$$ \delta_{PA} = 26.0^\circ \times \left( \frac{0.718}{A_{rec}} \right) $$

---

## 5. `src/patternspeed.py`

### Goal
To determine the rotation speed of the accretion flow ($\Omega_p$) using spatiotemporal autocorrelation.

### Processing Steps
1.  **Cylinder Sampling**:
    Extract $I(\phi, t)$ at the ring radius $R$.
    Normalize: $q(\phi, t) = \frac{I - \langle I \rangle_\phi}{\langle I \rangle_\phi}$.

2.  **Autocorrelation (ACF)**:
    $$ \mathcal{C}(\Delta \phi, \Delta t) = \text{IFFT}( |\text{FFT}(q)|^2 ) $$

3.  **Thresholding**:
    Identify the significant correlation region:
    $$ \mathcal{C} > \xi_{crit} $$
    where $\xi_{crit} = F_{pipe} \times \sigma_{ACF}$.

4.  **Speed Calculation**:
    Calculate the slope of the centroid of the thresholded ACF region.
    $$ \Omega_P = \frac{\langle \Delta \phi \Delta t \rangle}{\langle \Delta t^2 \rangle} $$

### Thresholds ($F_{pipe}$)
The factor $F_{pipe}$ depends on the pipeline/method character:
*   **Truth (Hotspot)**: 2.0
*   **Truth (Turbulence)**: 3.0
*   **Modeling (Mean)**: 0.6
*   **Resolve (MEM)**: 0.2
*   **EHTIM**: 0.6
*   **DOGHIT**: 1.2
*   **NGMEM**: 0.4

### Pass Criteria
*   The script runs MCMC to generate a distribution of pattern speeds for the reconstruction.
*   **Pass**: If the Reconstruction credible interval overlaps with the Truth credible interval.

---

## 6. `src/visualize.py`

### Goal
To visualize the data quality through visibility variance metrics.

### Mathematical Formulation
For baselines in the $(u,v)$ plane:
1.  **Amplitude Variance**:
    $$ V_{amp} = \frac{\text{std}(|V|)}{\text{mean}(|V|)} $$
    *(Normalized standard deviation of visibility amplitude)*.

2.  **Phase Variance**:
    $$ R = \left| \frac{1}{N} \sum_{t} e^{i \Phi_t} \right| $$
    $$ V_{phase} = 1 - R $$
    Where $R$ is the mean resultant vector length. $R=1$ implies perfect coherence (0 variance), $R=0$ implies uniform randomness (1 variance).

These variances map the stability of the source structure over time.
