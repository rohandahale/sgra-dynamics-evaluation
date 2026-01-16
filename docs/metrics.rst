=======
Metrics
=======

This document describes the evaluation metrics used to compare reconstructed 
black hole movies against ground truth images and observational data.

----

Metrics Summary
---------------

.. list-table:: Evaluation Metrics Overview
   :header-rows: 1
   :widths: 25 30 25 20

   * - Metric
     - Script
     - Key Quantities
     - Threshold Type
   * - Chi-squared
     - ``chisq.py``
     - :math:`\chi^2_{cp}`, :math:`\chi^2_{lca}`, :math:`\chi^2_m`
     - Absolute (1.0)
   * - NXCORR
     - ``nxcorr.py``
     - :math:`\rho_I`, :math:`\rho_P`, :math:`\rho_X`
     - Adaptive
   * - Hotspot Tracking
     - ``hotspot.py``
     - Position, FWHM, Flux
     - Absolute/Relative
   * - REx Ring Extraction
     - ``rex.py``
     - D, W, :math:`D_{true}`, A, PA, :math:`\beta_2`
     - Mixed
   * - VIDA Polarimetric
     - ``vida_pol.py``
     - MRing fit, :math:`\beta_2`, :math:`m_{net}`
     - Mixed
   * - Pattern Speed
     - ``patternspeed.py``
     - :math:`\Omega_P`
     - N/A
   * - Visibility Variance
     - ``visualize.py``
     - Amplitude/Phase variance
     - N/A

----

Chi-Squared Analysis
--------------------

**Script**: ``chisq.py``

**Purpose**: Assess the goodness-of-fit of the reconstruction against observed 
visibility data using :math:`\chi^2` statistics.


Closure Phase
^^^^^^^^^^^^^

.. math::

   \chi^2_{cp} = \frac{1}{N_{cp}} \sum_{i,j,k} 
   \frac{\left( \Phi_{ijk}^{obs} - \Phi_{ijk}^{model} \right)^2}{\sigma_{\Phi,ijk}^2}

where :math:`\Phi_{ijk}` is the closure phase on triangle :math:`(i,j,k)`.


Log Closure Amplitude
^^^^^^^^^^^^^^^^^^^^^

.. math::

   A_{ijkl} = \ln \left( \frac{|V_{ij}||V_{kl}|}{|V_{ik}||V_{jl}|} \right)

.. math::

   \chi^2_{lca} = \frac{1}{N_{lca}} \sum_{i,j,k,l} 
   \frac{\left( A_{ijkl}^{obs} - A_{ijkl}^{model} \right)^2}{\sigma_{A,ijkl}^2}


Polarization Ratio (m-breve)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The fractional polarization in visibility space:

.. math::

   \breve{m} = \frac{\mathcal{V}_Q + i\mathcal{V}_U}{\mathcal{V}_I}

.. math::

   \chi^2_m = \frac{1}{N_{vis}} \sum_k 
   \frac{\left| \breve{m}_k^{obs} - \breve{m}_k^{model} \right|^2}{\sigma_{\breve{m},k}^2}


Threshold and Pass Criteria
^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. list-table::
   :header-rows: 1
   :widths: 30 30 40

   * - Metric
     - Threshold
     - Note
   * - :math:`\chi^2_{cp}`
     - 1.0
     - Reduced :math:`\chi^2 \approx 1` indicates good fit
   * - :math:`\chi^2_{lca}`
     - 1.0
     - Horizontal reference line at 1.0
   * - :math:`\chi^2_m`
     - 1.0
     - JC sites auto-flagged for polarization

.. note::

   AA (ALMA) and AP (APEX) baselines are automatically flagged 
   to avoid calibration artifacts dominating the metric.

----

Normalized Cross-Correlation (NXCORR)
-------------------------------------

**Script**: ``nxcorr.py``

**Purpose**: Quantify image fidelity between reconstruction and truth using 
normalized cross-correlation for scalar and vector quantities.


Scalar NXCORR (Stokes I, :math:`|P|`)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Standard Pearson correlation coefficient:

.. math::

   \rho = \frac{\sum_k (X_k - \bar{X})(Y_k - \bar{Y})}
   {\sqrt{\sum_k (X_k - \bar{X})^2} \sqrt{\sum_k (Y_k - \bar{Y})^2}}


Polarization Vector NXCORR (:math:`\rho_P`)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

FFT-based cross-correlation of complex polarization vectors 
:math:`P = Q + iU`:

.. math::

   \rho_P = \text{Re}\left[ 
   \frac{\mathcal{F}^{-1}\left\{ \mathcal{F}(P_{rec}) \cdot \mathcal{F}^*(P_{truth}) \right\}}
   {\sqrt{\sum |P_{rec}|^2} \sqrt{\sum |P_{truth}|^2}}
   \right]

where :math:`\mathcal{F}` denotes the 2D Fourier transform. The real part 
penalizes EVPA misalignment.


EVPA-Weighted NXCORR (:math:`\rho_X`)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Evaluates EVPA alignment weighted by truth polarization magnitude:

.. math::

   P'_{rec} = \frac{P_{rec}}{|P_{rec}|} \cdot |P_{truth}|

.. math::

   \rho_X = \text{Re}\left[ 
   \frac{\sum P'_{rec} \cdot P^*_{truth}}
   {\sum |P_{truth}|^2}
   \right]


Threshold and Pass Criteria
^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. list-table::
   :header-rows: 1
   :widths: 25 40 35

   * - Quantity
     - Threshold
     - Pass Condition
   * - :math:`\rho_I` (Stokes I)
     - :math:`\tau_I = \rho(I_{truth}, I_{blur})`
     - :math:`\rho \geq \tau_I`
   * - :math:`\rho_P`, :math:`\rho_X`
     - :math:`\tau_P = \cos(2 \Delta\chi)` where :math:`\Delta\chi = 20°`
     - :math:`\rho \geq 0.766`

The scalar threshold is **adaptive**: computed by comparing truth with a 
blurred version (using a Gaussian fit to the dirty beam).

----

Hotspot Tracking
----------------

**Script**: ``hotspot.py``

**Purpose**: Extract and track the brightest dynamic feature in the image 
sequence and compare its trajectory with ground truth.


Method
^^^^^^

1. **Isolate Dynamic Component**:

   .. math::

      I_{dyn}(\mathbf{x}, t) = \max\left(0,\ I(\mathbf{x}, t) - I_{median}(\mathbf{x})\right)

2. **Gaussian Fitting**: For each frame, fit a 2D Gaussian to the dynamic component:

   .. math::

      G(x, y) = A \cdot \exp\left( -\frac{(x - x_0)^2 + (y - y_0)^2}{2\sigma^2} \right)

3. **Derived Quantities**:

   - Position: :math:`(x, y)` [μas]
   - Distance: :math:`r = \sqrt{x^2 + y^2}` [μas]
   - Angle: Sky PA East of North [°]
   - FWHM: :math:`2.355 \sigma` [μas]
   - Flux: :math:`A \cdot 2\pi \sigma_x \sigma_y` [Jy]


Threshold and Pass Criteria
^^^^^^^^^^^^^^^^^^^^^^^^^^^

A time step **passes** if the reconstruction interval overlaps with the 
truth tolerance band:

.. math::

   [\mu_{rec} - \sigma_{rec},\ \mu_{rec} + \sigma_{rec}] \cap 
   [\text{Truth} - \delta,\ \text{Truth} + \delta] \neq \emptyset

.. list-table::
   :header-rows: 1
   :widths: 30 20 30 20

   * - Quantity
     - Type
     - Threshold :math:`\delta`
     - Unit
   * - Position (x, y)
     - Absolute
     - 5.0
     - μas
   * - Distance
     - Absolute
     - 5.0
     - μas
   * - FWHM
     - Absolute
     - 5.0
     - μas
   * - Angle (PA)
     - Absolute
     - 20.0
     - °
   * - Flux
     - Relative
     - 25%
     - —

----

Ring Extraction (REx)
---------------------

**Script**: ``rex.py``

**Purpose**: Extract morphological and polarimetric ring parameters.


Ring Morphology
^^^^^^^^^^^^^^^

1. **Center Finding**: Locate ring centroid
2. **Polar Unwrapping**: Interpolate image onto :math:`(r, \phi)` grid
3. **Ring Fitting**:

   - **Diameter**: :math:`D = 2 \langle R_{peak} \rangle_\phi`
   - **Width**: :math:`W = \langle R_{out} - R_{in} \rangle_\phi` (FWHM)
   - **Asymmetry**: First Fourier mode amplitude

     .. math::

        A = \left| \frac{\sum_\phi I(r, \phi) e^{i\phi}}{\sum_\phi I(r, \phi)} \right|

   - **True Diameter** (corrected for width):

     .. math::

        D_{true} = \frac{D}{1 - \frac{1}{4\ln 2}\left(\frac{W}{D}\right)^2}


Polarization Quantities
^^^^^^^^^^^^^^^^^^^^^^^

- **Net fractional polarization**: 

  .. math::

     m_{net} = \frac{\sqrt{Q_{tot}^2 + U_{tot}^2}}{I_{tot}}

- **Average fractional polarization**: 

  .. math::

     \langle m \rangle = \frac{\sum \sqrt{Q^2 + U^2}}{I_{tot}}

- **:math:`\beta_2` mode** (m=2 azimuthal polarization):

  .. math::

     \beta_2 = \frac{\sum P(r, \phi) \cdot e^{-2i\phi} \cdot r}{\sum I(r, \phi) \cdot r}

  where :math:`P = Q + iU`.


Threshold and Pass Criteria
^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. list-table::
   :header-rows: 1
   :widths: 30 20 30 20

   * - Quantity
     - Type
     - Threshold
     - Note
   * - D
     - Absolute
     - 5.0 μas
     - Ring diameter
   * - W
     - Absolute
     - 5.0 μas
     - Ring width (FWHM)
   * - :math:`D_{true}`
     - Relative
     - 10% of truth
     - Corrected diameter
   * - A
     - Relative
     - 10% of truth
     - Asymmetry amplitude
   * - PA
     - Scaled
     - :math:`26° \cdot A_0 / A_{truth}`
     - :math:`A_0 = 0.718`
   * - :math:`\angle\beta_2`
     - Absolute
     - 26°
     - Polarization angle
   * - :math:`m_{net}`, :math:`\langle m \rangle`
     - Relative
     - 10% of truth
     - Polarization fractions
   * - :math:`|\beta_2|`, :math:`v_{net}`
     - Relative
     - 10% of truth
     - Polarization magnitudes

**Pass Condition**: :math:`|\text{diff}| - \sigma \leq \text{threshold}`

----

VIDA Polarimetric Fitting
-------------------------

**Script**: ``vida_pol.py`` (Python wrapper) + ``vida_pol.jl`` (Julia)

**Purpose**: Fit parametric ring models using the VIDA library for 
accurate extraction of ring and polarization parameters.


Model
^^^^^

Fits an MRing template with:

- Ring radius :math:`r_0`
- Ring width :math:`\sigma`
- Fourier mode amplitudes :math:`s_n` and phases :math:`\xi_n`
- Ellipticity :math:`\tau` and position angle :math:`\xi_\tau`
- Center offset :math:`(x_0, y_0)`


Derived Quantities
^^^^^^^^^^^^^^^^^^

- **Diameter**: :math:`d = 2 r_0`
- **Width**: :math:`w = 2.355 \sigma` (FWHM)
- **Asymmetry**: :math:`A = s_1 / 2`
- **Position Angle**: :math:`PA = \xi_1` (converted to degrees)
- **:math:`\beta_2`**: From LP mode integration

Thresholds are identical to REx (see above).

----

Pattern Speed
-------------

**Script**: ``patternspeed.py``

**Purpose**: Determine rotation speed :math:`\Omega_P` of image features 
using spatiotemporal autocorrelation.


Method
^^^^^^

1. **Cylinder Sampling**: Extract :math:`I(\phi, t)` at ring radius :math:`R`:

   .. math::

      q(\phi, t) = \frac{I(\phi, t) - \langle I \rangle_\phi}{\langle I \rangle_\phi}

2. **Autocorrelation**:

   .. math::

      \mathcal{C}(\Delta\phi, \Delta t) = \mathcal{F}^{-1}\left\{ |\mathcal{F}(q)|^2 \right\}

3. **Thresholding**: Identify significant region where 
   :math:`\mathcal{C} > \xi_{crit}`:

   .. math::

      \xi_{crit} = F_{pipe} \cdot \sigma_{ACF}

4. **Speed Calculation**:

   .. math::

      \Omega_P = \frac{\langle \Delta\phi \cdot \Delta t \rangle}{\langle \Delta t^2 \rangle}


Pipeline-Specific Factors
^^^^^^^^^^^^^^^^^^^^^^^^^

.. list-table::
   :header-rows: 1
   :widths: 40 30

   * - Pipeline / Source
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

Visibility Variance
-------------------

**Script**: ``visualize.py``

**Purpose**: Visualize data quality through visibility variance maps in the 
:math:`(u, v)` plane.


Metrics
^^^^^^^

**Amplitude Variance** (coefficient of variation):

.. math::

   V_{amp} = \frac{\text{std}(|V|)}{\text{mean}(|V|)}

**Phase Variance** (circular dispersion):

.. math::

   R = \left| \frac{1}{N} \sum_{t} e^{i\Phi_t} \right|

.. math::

   V_{phase} = 1 - R

where :math:`R` is the mean resultant vector length:

- :math:`R = 1`: Perfect phase coherence
- :math:`R = 0`: Uniform random phases

----

Thresholds Summary Table
------------------------

.. list-table::
   :header-rows: 1
   :widths: 25 15 25 35

   * - Metric
     - Type
     - Threshold
     - Pass Condition
   * - :math:`\chi^2` (all)
     - Absolute
     - 1.0
     - Reference line (lower is better)
   * - :math:`\rho_I`
     - Adaptive
     - :math:`\rho(I_{truth}, I_{blur})`
     - :math:`\rho \geq \tau`
   * - :math:`\rho_P`, :math:`\rho_X`
     - Adaptive
     - 0.766 (:math:`\cos 40°`)
     - :math:`\rho \geq \tau`
   * - Hotspot Position
     - Absolute
     - 5 μas
     - Interval overlap
   * - Hotspot Angle
     - Absolute
     - 20°
     - Interval overlap
   * - Hotspot Flux
     - Relative
     - 25%
     - Interval overlap
   * - Ring D, W
     - Absolute
     - 5 μas
     - :math:`|\text{diff}| - \sigma \leq \tau`
   * - :math:`D_{true}`, A, magnitudes
     - Relative
     - 10%
     - :math:`|\text{diff}| - \sigma \leq \tau`
   * - PA
     - Scaled
     - :math:`26° \cdot A_0/A_{truth}`
     - :math:`|\text{diff}| - \sigma \leq \tau`
   * - :math:`\angle\beta_2`
     - Absolute
     - 26°
     - :math:`|\text{diff}| - \sigma \leq \tau`
