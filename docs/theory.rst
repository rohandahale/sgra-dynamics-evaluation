==================
Theory and Methods
==================

This document provides detailed mathematical formulations for all evaluation 
metrics used in the Sgr A* Dynamics project.

.. contents:: Table of Contents
   :local:
   :depth: 2

----

Chi-Squared Analysis (``chisq``)
--------------------------------

Goal
^^^^

Assess the goodness-of-fit of the reconstruction against observed visibility 
data using :math:`\chi^2` statistics for closure quantities and polarization ratios.

Processing Steps
^^^^^^^^^^^^^^^^

1. **Data Preparation**

   - Load UVFITS observation data
   - **Flagging**: Automatically flag AA (ALMA) and AP (APEX) baselines to avoid 
     calibration errors dominating the metric
   - **Time Alignment**: Match observation times :math:`t_{obs}` to movie frames 
     :math:`t_{mov}` using nearest neighbor interpolation

2. **Metric Calculation** (for each time :math:`t`)

   **Closure Phase** (:math:`\chi^2_{cp}`):

   .. math::

      \chi^2_{cp} = \frac{1}{N_{cp}} \sum_{i,j,k} 
      \frac{(\Phi_{ijk}^{obs} - \Phi_{ijk}^{model})^2}{\sigma_{\Phi, ijk}^2}

   **Log Closure Amplitude** (:math:`\chi^2_{lca}`):

   .. math::

      A_{ijkl} = \ln \left( \frac{|V_{ij} V_{kl}|}{|V_{ik} V_{jl}|} \right)

   .. math::

      \chi^2_{lca} = \frac{1}{N_{lca}} \sum 
      \frac{(A^{obs} - A^{model})^2}{\sigma_{A}^2}

   **Polarization Ratio** (:math:`\chi^2_{m}`):

   Uses the "m-breve" fraction :math:`\breve{m} = \frac{\mathcal{Q} + i\mathcal{U}}{\mathcal{I}}`:

   .. math::

      \chi^2_{m} = \frac{1}{N_{vis}} \sum 
      \frac{|\breve{m}^{obs} - \breve{m}^{model}|^2}{\sigma_{\breve{m}}^2}

   .. note::

      JC (JCMT) sites are flagged for polarization metrics due to known calibration issues.

Thresholds
^^^^^^^^^^

A horizontal line at :math:`\chi^2 = 1` is plotted as the benchmark for a "good" 
fit (reduced :math:`\chi^2 \approx 1`).

----

Hotspot Tracking (``hotspot``)
------------------------------

Goal
^^^^

Extract and track the brightest dynamic feature ("hotspot") in the image sequence 
and compare its trajectory with ground truth.

Processing Steps
^^^^^^^^^^^^^^^^

1. **Isolate Dynamic Component**

   .. math::

      I_{dynamic}(\mathbf{x}, t) = \max(0, I(\mathbf{x}, t) - I_{median}(\mathbf{x}))

   Negative residuals are clipped to zero.

2. **Gaussian Fitting**

   For each frame, fit a 2D Gaussian to the maximum intensity location:

   .. math::

      G(x,y) = A \cdot \exp\left(-\frac{(x-x_0)^2 + (y-y_0)^2}{2\sigma^2}\right)

3. **Derived Quantities**

   - **Position**: :math:`(x, y) = (x_0, y_0)` [μas]
   - **Distance**: :math:`r = \sqrt{x^2 + y^2}` [μas]
   - **Angle**: :math:`\theta` (Sky PA East of North)
   - **Flux**: Volume under the Gaussian

Pass Criteria
^^^^^^^^^^^^^

The reconstruction **passes** if reconstruction interval overlaps with truth tolerance:

.. math::

   [\mu_{rec} - \sigma_{rec}, \mu_{rec} + \sigma_{rec}] \cap 
   [\text{Truth} - \delta, \text{Truth} + \delta] \neq \emptyset

**Tolerances** (:math:`\delta`):

- **Position** :math:`(x, y)`: ±5.0 μas
- **Distance**: ±5.0 μas
- **FWHM**: ±5.0 μas
- **Angle (PA)**: ±20.0°
- **Flux**: ±25% (relative)

----

Normalized Cross-Correlation (``nxcorr``)
-----------------------------------------

Goal
^^^^

Quantify image fidelity using NXCORR for scalar (:math:`I`, :math:`P_{mag}`) 
and vector (:math:`P`, :math:`X`) quantities.

Metric Calculation
^^^^^^^^^^^^^^^^^^

**Scalar NXCORR** (:math:`I`, :math:`P_{mag}`):

Standard normalized cross-correlation.

**Vector NXCORR** (:math:`P_{vec}`) - Cosine Similarity:

.. math::

   \text{NXCORR}_{vec} = \text{Re}\left[ 
   \frac{\sum P_{rec} P_{truth}^*}
   {\sqrt{\sum |P_{rec}|^2} \sqrt{\sum |P_{truth}|^2}} 
   \right]

**Cross-Pol NXCORR** (:math:`X`) - Centered Pearson:

.. math::

   P'_{rec} = P_{rec} - \langle P_{rec} \rangle

.. math::

   \text{NXCORR}_{X} = \text{Re}\left[ 
   \frac{\sum P'_{rec} (P'_{truth})^*}
   {\sqrt{\sum |P'_{rec}|^2} \sqrt{\sum |P'_{truth}|^2}} 
   \right]

Adaptive Threshold
^^^^^^^^^^^^^^^^^^

The threshold :math:`\tau` is computed by comparing truth with a blurred version:

.. math::

   \tau = \text{NXCORR}(I_{truth}, I_{blur})

where the blur kernel is a Gaussian fit to the dirty beam.

**Vector Threshold**:

.. math::

   \tau_{vec} = \cos(2 \cdot \chi_{rot})

For :math:`\chi_{rot} = 20°`: :math:`\tau_{vec} = \cos(40°) \approx 0.766`

----

Ring Extraction (``rex``)
-------------------------

Goal
^^^^

Extract morphological and polarimetric quantities using the **REx** algorithm.

Processing Steps
^^^^^^^^^^^^^^^^

1. **Center Finding**: Locate ring centroid using ``findCenter``
2. **Polar Unwrapping**: Interpolate onto :math:`(r, \phi)` grid
3. **Ring Fitting**:

   - **Diameter**: :math:`D = 2 \langle R_{peak} \rangle`
   - **Width**: :math:`W = \langle R_{out} - R_{in} \rangle` (FWHM)
   - **Asymmetry**: First Fourier mode amplitude :math:`A`

4. **Polarization Integration**:

   - Integrate :math:`Q, U` over the ring annulus
   - Compute :math:`\beta_2` (m=2 azimuthal mode)

Thresholds
^^^^^^^^^^

**Absolute**:

- Diameter :math:`D`: 5.0 μas
- Width :math:`W`: 5.0 μas
- Peak PA: 26.0°
- :math:`\beta_2` Angle: 26.0°

**Relative** (10% of truth):

- :math:`A, m_{net}, m_{avg}, |v|_{net}`, Fluxes

----

Pattern Speed (``patternspeed``)
--------------------------------

Goal
^^^^

Determine rotation speed :math:`\Omega_p` using spatiotemporal autocorrelation.

Processing Steps
^^^^^^^^^^^^^^^^

1. **Cylinder Sampling**

   Extract :math:`I(\phi, t)` at ring radius :math:`R`. Normalize:

   .. math::

      q(\phi, t) = \frac{I - \langle I \rangle_\phi}{\langle I \rangle_\phi}

2. **Autocorrelation**

   .. math::

      \mathcal{C}(\Delta \phi, \Delta t) = \text{IFFT}( |\text{FFT}(q)|^2 )

3. **Thresholding**

   Identify significant correlation region where :math:`\mathcal{C} > \xi_{crit}`:

   .. math::

      \xi_{crit} = F_{pipe} \times \sigma_{ACF}

4. **Speed Calculation**

   .. math::

      \Omega_P = \frac{\langle \Delta \phi \cdot \Delta t \rangle}
      {\langle \Delta t^2 \rangle}

Pipeline Factors (:math:`F_{pipe}`)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. list-table::
   :header-rows: 1

   * - Pipeline
     - :math:`F_{pipe}`
   * - Truth (Hotspot)
     - 2.0
   * - Truth (Turbulence)
     - 3.0
   * - Modeling (Mean)
     - 0.6
   * - Resolve (MEM)
     - 0.2
   * - EHTIM
     - 0.6
   * - DOGHIT
     - 1.2
   * - NGMEM
     - 0.4

----

Visibility Variance (``visualize``)
-----------------------------------

Goal
^^^^

Visualize data quality through visibility variance metrics in the :math:`(u,v)` plane.

Metrics
^^^^^^^

**Amplitude Variance**:

.. math::

   V_{amp} = \frac{\text{std}(|V|)}{\text{mean}(|V|)}

**Phase Variance**:

.. math::

   R = \left| \frac{1}{N} \sum_{t} e^{i \Phi_t} \right|

.. math::

   V_{phase} = 1 - R

where :math:`R` is the mean resultant vector length (:math:`R=1` implies perfect 
coherence, :math:`R=0` implies uniform randomness).
