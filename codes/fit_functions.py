#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
import KC_population_calcium_rate_model_functions as mdl


# Functions related to fitting the rate model to data

def simulate_KD_model(t, starr, params, inpscale, dt, n_units, noise=None, n_str=0):
    """
    Simulate the KD model with the given parameters and input scaling factor.


    Parameters
    ----------
    t : 1-D array
        Time array
    starr : 1-D array
        Stimulus array
    params: lmfit.Parameters object or list
        Parameters for the KD model
    inpscale: 1-D array
        Input scaling factor for each KC
    dt: float
        Time step for the simulations
    n_units: int
        Number of KCs
    noise: 2-D array, optional
        Noise array for each KC. The default is None since we do not need it for fitting the model
    n_str: float, optional
        Standard deviation of the OU process. The default is 0.

    Returns
    -------
    CaKCcal : 2-D array
        Calcium transients for each KC
    adapt : 2-D array
        Adaptation traces for each KC
    """
    # This will be used to simulate the WT model using the KD parameters, which will then be passed to a residual function to fit the WT model.
    # Unpack the model parameters
    try:
        tauKCdec = params['tauKCdec'].value
        tauinp = params['tauinp'].value
        tauadapt = params['tauadapt'].value
        adaptscale = params['adaptscale'].value
        bline = params['bline'].value
    except TypeError:
        tauKCdec, tauinp, tauadapt, adaptscale, tauinh, inhfactor, infp, slf, bline, = params

    # Preallocate y (state vector) parameters
    CaKCcal = np.zeros([len(t), n_units])
    CaKCcal[0] = bline
    adapt = np.zeros([len(t), n_units])

    # Adjust noise strength to time constant
    n_str = n_str * np.sqrt(2 / tauKCdec * dt)

    # Simulate
    for i in range(len(t) - 1):
        # Calcium transients
        CaKCcal[i + 1] = CaKCcal[i] + ((inpscale * starr[i]) / tauinp - (CaKCcal[i] - bline) / tauKCdec) * dt
        if noise is not None:
            CaKCcal[i + 1] += noise[i] * n_str

        # Adaptation
        adapt[i + 1] = mdl.adaptation_dynamics(adapt[i], CaKCcal[i], tauadapt, adaptscale, dt)
        CaKCcal[i + 1] -= (adapt[i] * CaKCcal[i]) * (1 / tauKCdec) * dt

        # Make sure everything is nonzero
        CaKCcal[i + 1] = np.clip(CaKCcal[i + 1], 0, None)

    return CaKCcal, adapt


def simulate_WT_model(t, starr, inhparams, fittedparams, inpscale, dt, n_units, noise=None, n_str=0):
    """
    Simulate the WT model with the given parameters and input scaling factor.


    Parameters
    ----------
    t : 1-D array
        Time array
    starr : 1-D array
        Stimulus array
    inhparams : lmfit.Parameters object or list
        Parameters for the inhibition in the WT model
    fittedparams : lmfit.Parameters object or list
        Parameters for the rest of the WT model, fitted using the KD model. This is the variable params in function simulate_KD_model.
    inpscale : 1-D array
        Input scaling factor for each KC
    dt : float
        Time step for the simulations
    n_units: int
        Number of KCs
    noise : 2D array, optional
        Noise array for each KC. The default is None since we do not need it for fitting the model
    n_str : float, optional
        Standard deviation of the OU process. The default is 0.

    Returns
    -------
    CaKClobe : 2-D array
        Calcium transients for each KC in the lobe
    adapt : 2-D array
        Adaptation traces for each KC
    inh : 2-D array
        Inhibition traces for each KC
    """
    # This will be used to simulate the WT model using the KD parameters, which will then be passed to a residual function to fit the WT model.
    # Unpack the (fitted) parameters
    try:
        tauKCdec = fittedparams['tauKCdec'].value
        tauinp = fittedparams['tauinp'].value
        tauadapt = fittedparams['tauadapt'].value
        adaptscale = fittedparams['adaptscale'].value
        bline = fittedparams['bline'].value
    except TypeError:
        tauKCdec, tauinp, tauadapt, adaptscale, bline, = fittedparams

    # Inh params
    try:
        tauinh = inhparams['tauinh'].value
        inhfactor = inhparams['inhfactor'].value
        infp = inhparams['infp'].value
        slf = inhparams['slf'].value
    except TypeError:
        tauinh, inhfactor, infp, slf = inhparams

    # Preallocate y (state vector) parameters
    CaKCcal = np.zeros([len(t), n_units])
    CaKCcal[0] = bline
    CaKClobe = np.zeros([len(t), n_units])
    CaKClobe[0] = bline
    adapt = np.zeros([len(t), n_units])
    inh = np.zeros([len(t), n_units])

    # Generate the inhibition connectivity matrix
    inhscale = np.ones([n_units] * 2)  # lateral inhibition connectivity matrix between KCs. Uniform for now
    inhscale[np.diag_indices(n_units)] = 0  # Remove self-inhibition
    inhscale /= np.sum(inhscale, axis=1)[:, None]  # Normalize so that each KC receives the same amount of inhibition from all other KCs.
    inhscale *= inhfactor  # Ramp up the degree of inhibition

    # Adjust noise strength to time constant
    n_str = n_str * np.sqrt(2 / tauKCdec * dt)

    # Simulate
    for i in range(len(t) - 1):
        # Calcium transients
        CaKCcal[i + 1] = CaKCcal[i] + ((inpscale * starr[i]) / tauinp - (CaKCcal[i] - bline) / tauKCdec) * dt
        CaKClobe[i + 1] = CaKClobe[i] + ((inpscale * starr[i]) / tauinp - (CaKClobe[i] - bline) / tauKCdec) * dt

        if noise is not None:
            CaKCcal[i + 1] += noise[i] * n_str
            CaKClobe[i + 1] += noise[i] * n_str

        # Adaptation
        adapt[i + 1] = mdl.adaptation_dynamics(adapt[i], CaKCcal[i], tauadapt, adaptscale, dt)
        CaKCcal[i + 1] -= (adapt[i] * CaKCcal[i]) * (1 / tauKCdec) * dt
        CaKClobe[i + 1] -= (adapt[i] * CaKClobe[i]) * (1 / tauKCdec) * dt

        # inhibition (as in WT model)
        inh[i + 1] = mdl.inhibition_dynamics(inh[i], CaKClobe[i], tauinh, inhscale, dt) \
                     * mdl.activity_dependent_inhibition_modulation_sigmoidal(CaKClobe[i], infp, slf)  # modulated inhibition
        CaKClobe[i + 1] -= (inh[i]) * (1 / tauKCdec) * dt

        # Make sure everything is nonzero
        CaKCcal[i + 1] = np.clip(CaKCcal[i + 1], 0, None)
        CaKClobe[i + 1] = np.clip(CaKClobe[i + 1], 0, None)

    return CaKClobe, CaKCcal, adapt, inh


def calculate_residual(params, t, starr, data, sterr, inpscale, restparams, dt, n_units, fit_WT=True, normfac=1):
    """
    Calculate the objective function to be minimized. This is the L1-normed squared sum of the residual between the model and the data.
    Parameters
    ----------
    params : lmfit.Parameters object or list
        Parameters to be fitted for the model
    t : 1-D array
        Time array
    starr : 1-D array
        Stimulus array
    data : 1-D array
        Data to be fitted
    sterr : 1-D array
        Standard error of the data
    inpscale : 1-D array
        Input scaling factor for each KC
    restparams : lmfit.Parameters object or list
        Parameters for the model that are kept constant, this is relevant when fitting the WT model (i.e. when fit_WT=True)
    dt : float
        Time step for the simulations
    n_units : int
        Number of KCs
    fit_WT : boolean, optional
        Whether to fit the WT model or the KD model. The default is True.
    normfac : float, optional
        L1 Normalization factor for the parameters. The default is 1.

    Returns
    -------
    residual : float
        Objective function to be minimized
    """
    # Calculate the model prediction
    if fit_WT:
        model = np.mean(simulate_WT_model(t, starr, params, restparams, inpscale, dt, n_units)[0][:, inpscale > 0], axis=1)
    else:
        model = np.mean(simulate_KD_model(t, starr, params, inpscale, dt, n_units)[0][:, inpscale > 0], axis=1)
    # Calculate the residual
    residual = np.sum(((model - data) / sterr) ** 2) + normfac * np.sum([np.abs(params[p].value) for p in params])
    return residual
