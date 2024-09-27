from __future__ import annotations


class Options:
    """Options for the floquet analysis.

    fit_range_fraction: float
        Fraction of the amplitude range to sweep over before changing the definition of
        the bare state to that of the fitted state from the previous range. For instance
        if fit_range_fraction=0.4, then the amplitude range is split up into three
        chunks: the first 40% of the amplitude linspace, then from 40% -> 80%, then
        from 80% to the full range. For the first fraction of amplitudes, they are
        compared to the bare eigenstates for identification. For the second range, they
        are compared to the fitted state from the first range. And so on. Defaults to
        1.0, indicating that no iteration is performed.
    floquet_sampling_time_fraction: float
        What point of the drive period we want to sample the Floquet modes. Defaults to
        0.0, indicating the floquet modes at t=0*T where T is the drive period.
    fit_cutoff: int
        Cutoff for the fit polynomial of the displaced state. Defaults to 4.
    overlap_cutoff: float
        Cutoff for fitting overlaps. Floquet modes with overlap with the "bare" state
        below this cutoff are not included in the fit (as they may be experiencing a
        resonance). Defaults to 0.8.
    nsteps: int
        QuTiP integration parameter, number of steps the solver can take. Defaults
        to 30_000.
    num_cpus: int
        number of cpus to use in parallel computation of Floquet modes over the
        different values of omega_d, amp. Defaults to 1.
    save_floquet_mode_data: bool
        Indicating whether to save the extracted Floquet modes themselves. Such data is
        often unnecessary and requires a fair amount of storage, so the default is
        False.
    """

    def __init__(
        self,
        fit_range_fraction: float = 1.0,
        floquet_sampling_time_fraction: float = 0.0,
        fit_cutoff: int = 4,
        overlap_cutoff: float = 0.8,
        nsteps: int = 30_000,
        num_cpus: int = 1,
        save_floquet_mode_data: bool = False,
    ):
        self.fit_range_fraction = fit_range_fraction
        self.floquet_sampling_time_fraction = floquet_sampling_time_fraction
        self.fit_cutoff = fit_cutoff
        self.overlap_cutoff = overlap_cutoff
        self.nsteps = nsteps
        self.num_cpus = num_cpus
        self.save_floquet_mode_data = save_floquet_mode_data
