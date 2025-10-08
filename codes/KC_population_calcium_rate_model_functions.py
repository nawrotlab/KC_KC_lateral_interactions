#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np

#Functions
#-------------
def generate_step_stimulus(tend, ston, stoff, dt, stamp=1):
    """
    Generate a step stimulus.

    Parameters
    ----------
    tend : float
        End time point of the stimulus protocol in seconds.
    ston : float
        Stimulus onset in seconds.
    stoff : float
        Stimulus offset in seconds.
    dt : float
        Time step size (i.e. inverse of sampling rate) of the stimulation in seconds.
    stamp : float, optional
        Amplitude of the stimulus. The default is 1.
    Returns
    -------
    tarr : 1-D array
        Time array.
    starr : 1-D array
        Stimulus array.

    """
    tarr = np.round(np.arange(0, tend+dt, dt), np.ceil(-np.log10(dt)).astype(int)) # time array in seconds
    starr = np.zeros(tarr.shape) # the stimulus array
    starr[(tarr>=ston) & (tarr<=stoff)] = stamp
    if tarr[-1] > tend:
        # Remove from tarr and starr the time points larger than tend
        tarr = tarr[:-1]
        starr = starr[:-1]
    return tarr, starr


def generate_stimulus_stream(ston, nstim, stdur, ISIdur, ststr, restt, dt):
    """
    Generate a stimulus stream. This is a wrapper around generate_step_stimulus that allows for multiple stimulus steps

    Parameters
    ----------
    ston : float
        Stimulus onset in seconds.
    nstim : int
        Number of step stimuli.
    stdur : float or 1-D array
        Duration of each stimulus step in seconds. If float, all stimulus steps are the same length, otherwise the
        order decides the duration of each stimulus step. Shape nstim if array
    ststr : float or 1-D array
        Strength of each stimulus step in seconds. If float, all stimulus steps are the same strength, otherwise the
        order decides the strength of each stimulus step. Shape nstim if array
    ISIdur : float or 1-D array
        Inter-stimulus interval duration in seconds. Similar to stdur if float, all ISI are the same length,
        otherwise the order decides the duration of each ISI. Shape nstim - 1 if array.
    restt : float
        Resting time after the last stimulus step in seconds.
    dt : float
        Time step size (i.e. inverse of sampling rate) of the stimulation in seconds.

    Returns
    -------
    tarr : 1-D array
        Time array.
    starr : 1-D array
        Stimulus array.

    """
    # Check if stdur and ISIdur are floats and generate boolean handlers for the 4 different cases
    if isinstance(stdur, float) | isinstance(stdur, int):
        # One stimulus step duration is present
        onestep = True
    else:
        onestep = False
    if isinstance(ISIdur, float) | isinstance(ISIdur, int):
        # One ISI duration is present
        oneisi = True
    else:
        oneisi= False
    if isinstance(ststr, float) | isinstance(ststr, int):
        # One stimulus strength is present
        onestr = True
    else:
        onestr = False

    if onestep & oneisi:
        # If both are floats just concatenate
        __, fss = generate_step_stimulus(ston+stdur, ston, ston+stdur, dt) # First stimulus step with the onset followed by the stimulation
        __, rss = generate_step_stimulus(ISIdur+stdur, ISIdur, ISIdur+stdur, dt) # Remeaining stimulus step with ISI followed by the stimulation

        if onestr:
            starr = np.concatenate([fss, np.tile(rss[1:], nstim-1)])*ststr # Concatenate the stimulus steps and add the resting time at the end
                                                                           # Remove t=0 for rss since it is already included in fss
        else:
            starr = np.concatenate([fss*ststr[0], np.tile(rss[1:], nstim-1)*ststr[1:]])

    elif oneisi:
        # Only ISI is fixed, stdurs change for each stimulus step.
        if onestr:
            __, starr = generate_step_stimulus(ston+stdur[0], ston, ston+stdur[0], dt) # First stimulus step with the onset followed by the stimulation
            for i in range(1, nstim):
                __, rss = generate_step_stimulus(ISIdur + stdur[i], ISIdur, ISIdur+stdur[i], dt)
                starr = np.concatenate([starr, rss[1:]]) # Concatenate the stimulus steps and add the resting time at the end
                                                         # Remove t=0 for rss since it is already included in fss
            starr *= ststr
        else:
            __, starr = generate_step_stimulus(ston+stdur[0], ston, ston+stdur[0], dt) # First stimulus step with the onset followed by the stimulation
            starr *= ststr[0]
            for i in range(1, nstim):
                __, rss = generate_step_stimulus(ISIdur + stdur[i], ISIdur, ISIdur+stdur[i], dt)
                starr = np.concatenate([starr, rss[1:]*ststr[i]]) # Concatenate the stimulus steps and add the resting time at the end
                                                                  # Remove t=0 for rss since it is already included in fss

    elif onestep:
        if onestr:
            # Only stimulus duration is fixed, ISIdurs change for each stimulus step
            __, starr = generate_step_stimulus(ston+stdur, ston, ston+stdur, dt) # First stimulus step with the onset followed by the stimulation
                                                                                 # Remove t=0 for rss since it is already included in fss
            for i in range(0, nstim-1):
                __, rss = generate_step_stimulus(ISIdur[i]+stdur, ISIdur[i], ISIdur[i]+stdur, dt)
                starr = np.concatenate([starr, rss[1:]]) # Concatenate the stimulus steps and add the resting time at the end
            starr *= ststr  # Multiply with the stimulus strength
        else:
            # Only stimulus duration is fixed, ISIdurs change for each stimulus step.
            __, starr = generate_step_stimulus(ston+stdur, ston, ston+stdur, dt) # First stimulus step with the onset followed by the stimulation
                                                                                 # Remove t=0 for rss since it is already included in fss
            starr *= ststr[0]
            for i in range(0, nstim-1):
                __, rss = generate_step_stimulus(ISIdur[i]+stdur, ISIdur[i], ISIdur[i]+stdur, dt)
                starr = np.concatenate([starr, rss[1:]*ststr[i]]) # Concatenate the stimulus steps and add the resting time at the end

    else:
        if onestr:
            #Both stimulus duration and ISIdurs change for each stimulus step.
            __, starr = generate_step_stimulus(ston+stdur[0], ston, ston+stdur[0], dt) # First stimulus step with the onset followed by the stimulation
            for i in range(1, nstim):
                __, rss = generate_step_stimulus(ISIdur[i-1] + stdur[i], ISIdur[i-1], ISIdur[i-1] + stdur[i], dt)
                starr = np.concatenate([starr, rss[1:]])  # Concatenate the stimulus steps and add the resting time at the end
                                                          # Remove t=0 for rss since it is already included in fss
            starr *= ststr  # Multiply with the stimulus strength
        else:
            #Both stimulus duration and ISIdurs change for each stimulus step.
            __, starr = generate_step_stimulus(ston+stdur[0], ston, ston+stdur[0], dt) # First stimulus step with the onset followed by the stimulation
            starr *= ststr[0]
            for i in range(1, nstim):
                __, rss = generate_step_stimulus(ISIdur[i-1] + stdur[i], ISIdur[i-1], ISIdur[i-1] + stdur[i], dt)
                starr = np.concatenate([starr, rss[1:]*ststr[i]])  # Concatenate the stimulus steps and add the resting time at the end
                                                                   # Remove t=0 for rss since it is already included in fss

    starr = np.concatenate([starr, np.zeros(int(restt/dt))]) # Concatenate the stimulus steps and add the resting time at the end

    # Create the time array related to stimulus
    tend = (len(starr)-1) * dt # end time point of the stimulus protocol in seconds.
    # Rounding for numerical stability, number of decimals in the round should be around the same order of magnitude
    # as the sampling rate (i.e. inverse of dt).
    tarr = np.round(np.linspace(0, tend, len(starr)), np.ceil(-np.log10(dt)).astype(int))  # time array in seconds
    return tarr, starr


def adaptation_dynamics(adaptval, CaKCval, tauadapt, adaptscale, dt):
    """
    Simulate the adaptation dynamics for each stimulus time step for a given KC.

    Parameters
    ----------
    adaptval : float or 1-D array
        Current value of the adaptation variable.
    CaKCval : float or 1-D array
        Current value of the KC calcium trace (i.e. KC activation degree).
    tauadadpt : float
        Adaptation time constant.
    adaptscale : float
        Degree of adaptation strength on the neuron.
    dt : float
        Time step size (i.e. inverse of sampling rate) of the stimulation in seconds.

    Returns
    -------
    newadapt : float
        Updated value of the adaptation variable in the next step.

    """
    return adaptval + (adaptscale*CaKCval - adaptval) / tauadapt * dt

    
def inhibition_dynamics(inhval, CaKCval, tauinh, inhscale, dt):
    """    
    Simulate the lateral inhibition dynamics for each stimulus time step for a given KC neuron.

    Parameters
    ----------
    inhval : float
        Current value of the lateral inhibition variable.
    CaKCval : float
        Current value of the neighboring KC calcium traces (i.e. their activity degree).
    tainh : float
        Lateral inhibition time constant.
    inhscale : float or n-D array
        Degree of lateral inhibition strength on the neuron if float. Else it is the connectivity matrix (shape nKCs * nKCs)
        where first dimension is the KC inhibited, second dimension is the KC inhibiting.
        For inhscale to be connectivity matrix, CaKCval should be a vector of length nKCs.
    dt : float
        Time step size (i.e. inverse of sampling rate) of the stimulation in seconds.


    Returns
    -------
    newinh : float
        Updated value of the lateral inhibition variable in the next step.

    """
    try:
        # In this case the inhscale is the connectivity matrix.
        return inhval + (inhscale @ CaKCval - inhval) / tauinh * dt
    except TypeError:
        # This is when the model is being run for a single KC and the inhibition is a scalar
        return inhval + (inhscale*CaKCval - inhval) / tauinh * dt


def activity_dependent_inhibition_modulation_sigmoidal(CaKCval, modinf, modslf):
    """
    Calculate the activity dependent (multiplicative) inhibition modulation based on the sigmoidal function.
    
    Parameters
    ----------
    CaKCval : float or 1-D array
        The activity degree of the given KC(s).
    modinf : float
        The inflection point of the sigmoidal function.
    modslf : float
        The slope factor of the sigmoidal function.

    Returns
    -------
    modval : float or 1D array
        The activity dependent inhibition modulation value(s).
    """
    return 1 / (1 + np.exp(-(modinf - CaKCval) / modslf))


def DAN_dynamics(danval, taudan, dt, shock=0, valpred=0):
    """
    Model the DAN activity. DAN response reflects the prediction error of the model, defined as the difference between the real shock stimulus strength
    and the predicted valence i.e. activity difference between avoidance and approach MBONs.


    Parameters
    ----------
    danval : float
        Current value of the DAN activity.
    taudan : float
        DAN time constant.
    dt : float
        Time step size (i.e. inverse of sampling rate) of the stimulation in seconds.
    shock : float, optional
        Current shock stimulus value. The default is 0. This value corresponds to the real valence of the US.
    valpred : float, optional
        Current predicted valence for the given odor.
        Specifically, it is calculated as the difference between the activity of the avoidance MBON and the approach MBON.
        The default and initial value is 0.

    Returns
    -------
    new_danval : float
        Updated value of DAN activity driven cAMP levels.

    """
    shock_on = shock > 0 # turn on the valence prediction circuitry only when there is shock happening

    new_danval = danval - (danval - shock_on * (shock - valpred)) / taudan * dt
    return np.clip(new_danval, 0, None) # Make sure cAMP levels are always non-negative


def KC_MBON_coincidence_based_weight_change(weightval, CaKCval, danval, dt, learnrate=1):
    """
    Calculate the synaptic weight change between KC and MBON when coincident KC and DAN activity occurs.

    
    Parameters
    ----------
    weightval : float
        Current synaptic weight value between KC and MBON.
    CaKCval : float
        Current KC calcium level value (or activity degree).
    danval : float
        Current DAN-driven cAMP level value.
    dt : float
        Time step size (i.e. inverse of sampling rate) of the stimulation in seconds.
    learnrate : float, optional
        Learning rate specifying the synaptic weight change magnitude. The default is 1. This scales the synaptic modulation strength per time step.
    Returns
    -------
    newweight : float
        Updated synaptic weight value between KC and MBON.

    """
    newweight = weightval - learnrate * (CaKCval * danval) * dt  # Reduction in KC::MBON synaptic weight depends on the MBON activity as well (3 factor learning rule,
                                                               # this enables learning rate to reduce over trials.)
    return np.clip(newweight, 0, None)  # Make sure weights are always non-negative

def sigmoidal_func(x, slf, infp, mn=-1, amp=2):
    """
    Sigmoidal function with 4 free parameters

    Parameters
    ----------
    x : 1-D array
        The x value over which the sigmoidal is calculated.
    mn : float or 1-D array
        Infimum of sigmoidal. Function asymptotically arrives to this value at -inf.
    amp : float or 1-D array
        Amplitude of the sigmoidal from infimum to supremum. Supremum can be found via amp+mn
        Function asymptotically arrives to amp+mn value at +inf. Cannot be smaller than 0.
    slf : float or 1-D array
        Slope factor. The slope of the sigmoidal is amp/(4*slf) at inflection point. Can be negative or positive.
    infp : float or 1-D array
        The x coordinate of the inflection point.

    Returns
    -------
    sigm : 1_D array
        The sigmoidal function.

    """
    if type(mn) == np.ndarray:
        # Cheap trick to allow broadcasting
        return mn + amp / (1 + np.exp((infp - x.reshape([len(x), 1])) / slf))

    else:
        return mn + amp / (1 + np.exp((infp - x) / slf))
