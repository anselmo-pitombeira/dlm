# -*- coding: utf-8 -*-
"""
Created on Sat Sep 19 16:11:47 2015

@author: Anselmo
"""

import numpy as np
from scipy.stats import norm     ##Distribuição normal

def first_order_constant(Y, m0, C0, W, V, d = None):
    
    """u'First order polynomial constant dynamic linear model
    Y - Data vector
    m0 - Initial prior expected level
    C0 - Initial prior level variance
    W - Evolution variance
    V - Observation variance
    d - Discount rate, typically between 0.8 and 1.0"""
    
    m = m0    ##Initial mean
    C = C0    ##Initial variance
    mm = []   ##Save all means
    CC = []   ##Save all variances
    ff = []   ##Save all forecasts
    QQ = []   ##Save all forecast variances
    
    for n in range(Y.size):
        ##Prior in t (apply evolution variance)
        if d==None:
            R = C+W
        else:
            R = C/float(d)
        ##1-step forecast
        f = m
        ff.append(f)
        Q = R+V    ##Apply error variance
        QQ.append(Q)
        ##Posterior update
        e = Y[n]-f
        A = R/Q
        m = m+A*e
        mm.append(m)
        C = A*V
        CC.append(C)
        
    return np.array(mm), np.array(CC), np.array(ff), np.array(QQ)
    
    

def first_order_constant_obs(y, m0, C0, d, n0, S0):
    
    """First order polynomial dynamic linear model
    This dlm assumes that the constant observation variance is not known
    a priori, and updates an estimated St of it at each time t.
    Y - Data vector
    m0 - Initial prior mean level
    C0 - Initial prior level variance
    d - Discount rate used in determining the a priori variance R.
        Typical values between 0.8 and 1.0
    n0 - Initial estimate of the degrees of freedom
    S0 - Initial estimate of observation variance V"""
    
    m = m0    ##Initial mean
    C = C0    ##Initial variance
    n = n0    ##Initial degrees of freedom
    S = S0    ##Initial estimate of observation variance
    mm = []   ##Save all estimates of systemic mean
    CC = []   ##Save all estimates of the systemic variance
    SS = []   ##Save all estimates of the observation variance
    ff = []   ##Save all forecasts
    QQ = []   ##Save all forecast variances
    
    F = np.array([1])
    G = np.eye(1)
    
    for i in range(y.size):
#        R = C/float(d)    ##Prior in t (apply evolution variance)
#        f = m             ##1-step forecast
#        Q = R+S           ##Error variance (S is an estimate for observation variance V)
#        e = Y[i]-f        ##Forecast error
#        A = R/Q           ##Adjustment factor
#        m = m+A*e         ##Update estimate of systematic mean
#        C = A*S           ##Update variance
#        n = n+1           ##Update degrees of freedom of gamma distribution
#        S = S+S/float(n)*(e**2/Q-1)    ##Update observation variance estimated
        W = (1-d)/d*C
        m, C, f, Q, n, S = update_posterior_V_unknown(y[i], m, C, F, G, W, n, S)
        mm.append(m)
        CC.append(C)
        ff.append(f)
        QQ.append(Q)
        SS.append(S)
        
    return np.array(mm), np.array(CC), np.array(ff), np.array(QQ), np.array(SS)
    

def first_order_seasonal(Y, m0, C0, m0_, C0_, n0=None, S0=None, V=None, d_level=None, d_seasonal=None, p=None):
    
    """First order polynomial seasonal dynamic linear model
    This dlm assumes that the constant observation variance is not known
    a priori, and updates an estimated St of it at each time t.
    Y - Data vector
    m0 - Initial prior mean level
    C0 - Initial prior level variance
    d_level - Discount rate used in determining the a priori variance R.
        Typical values between 0.8 and 1.0
    d_seasonal - Discount factor of seasonal component
    n0 - Initial estimate of the degrees of freedom
    S0 - Initial estimate of observation variance V
    m0_ - Initial prior for seasonal effects
    C0_ - Initial covariance matrix for seasonal factors
    p - Number of seasonal effects"""
    
    ##Set observational variance parameters
    n = n0
    S = S0
    
#    ##Correct q0 and R0
#    tr = np.trace(C0_)    ##Save trace of covariance matrix of seasonal effects
#    U = C0_.sum()   ## Alternatively: np.dot(np.ones(p), np.dot(R0, np.ones(p)))
#    A_ = np.dot(C0_, np.ones(p))/U
#    m0_ = m0_-m0_.sum()*A_
#    C0_ = C0_-U*np.outer(A_, A_)
#    
#    ##Correct trace
#    correction = tr/np.trace(C0_)
#    C0_ = correction*C0_  
    
#    print m0_
#    print C0_
    
    ##Mount initial m vector
#    m = np.concatenate(np.array([m0]), m0_)
    m = np.zeros(p+1)
    m[0] = m0
    m[1:] = m0_
    
    ##Mount initial C matrix
    C = np.zeros((p+1, p+1))
    C[0,0] = C0
    C[1:p+1, 1:p+1] = C0_
    
    ##Mount permutation matrix
    P = np.zeros((p,p))
    P[0:p-1, 1:p] = np.eye(p-1)
    P[p-1,0] = 1
    
    ##Mount G matrix
    G = np.zeros((p+1, p+1))
    G[1:p+1, 1:p+1] = P
    G[0,0] = 1
    
    ##Mount F vector
    F = np.zeros(p+1)
    F[0:2] = 1
    
#    n = n0    ##Initial degrees of freedom
#    S = S0    ##Initial estimate of observation variance
    mm = []   ##Save all estimates of systemic mean
    CC = []   ##Save all estimates of the systemic variance
#    SS = []   ##Save all estimates of the observation variance
    ff = []   ##Save all forecasts
    QQ = []   ##Save all forecast variances
    
    for i in range(Y.size):
        
        ##Correct vector of seasonal effecs its covariance matrix
        
        ##The if below was newcessary due to accumlation of errors
        ##It basically renormalize the vector of seasonal effects (which must sum to zero)
        ##And the covariance matrix, whose columns and ros must all sum to zero.
        U = C[1:p+1, 1:p+1].sum()   ## Alternatively: np.dot(np.ones(p), np.dot(R0, np.ones(p)))
        if not np.isclose(np.abs(U), 0.0):
            tr = np.trace(C[1:p+1, 1:p+1])    ##Save trace of covariance matrix of seasonal effects
            A_ = np.dot(C[1:p+1, 1:p+1], np.ones(p))/U
            m[1:] = m[1:]-m[1:].sum()*A_
            C[1:p+1, 1:p+1] = C[1:p+1, 1:p+1]-U*np.outer(A_, A_)
        
            ##Correct trace
            correction = tr/np.trace(C[1:p+1, 1:p+1])
            C[1:p+1, 1:p+1] = correction*C[1:p+1, 1:p+1]  
        
        ##Calculate evolution matrix for level of the series
        W_level = (1-d_level)/d_level*C[0,0]   ##Determine evolution matrix for level term
        W_seasonal = (1-d_seasonal)/d_seasonal*P.dot(C[1:p+1, 1:p+1]).dot(P.T)    ##Determine evolution matrix for seasonal terms
        W = np.zeros_like(C)
        W[0,0] = W_level
        W[1:p+1, 1:p+1] = W_seasonal
        y = Y[i]    ##Current observation
        ##Update posterior parameters
#        m, C, f, Q = update_posterior(y, m, C, F, G, W, V)
        m, C, f, Q, n, S = update_posterior_V_unknown(y, m, C, F, G, W, n, S)
#        print m
        
        mm.append(m)
        CC.append(C)
        ff.append(f)
        QQ.append(Q)
#        SS.append(S)
        
    return np.array(mm), np.array(CC), np.array(ff), np.array(QQ)


def second_order_seasonal(Y, m0, C0, m0_, C0_, n0=None, S0=None, V=None, d_level=None, d_seasonal=None, p=None):
    
    """Second order polynomial seasonal dynamic linear model
    This dlm assumes that the constant observation variance is not known
    a priori, and updates an estimated St of it at each time t.
    Y - Data vector
    m0 - Initial prior mean level
    C0 - Initial prior level variance
    d_level - Discount rate used in determining the a priori variance R.
        Typical values between 0.8 and 1.0
    d_seasonal - Discount factor of seasonal component
    n0 - Initial estimate of the degrees of freedom
    S0 - Initial estimate of observation variance V
    m0_ - Initial prior for seasonal effects
    C0_ - Initial covariance matrix for seasonal factors
    p - Number of seasonal effects"""
    
    ##Set observational variance parameters
    n = n0    ##Initial degrees of freedom
    S = S0    ##Initial estimate of observation variance
    
    ##Mount initial m vector
    m = np.zeros(p+2)    ##p seasonal effects plus level and trend.
    m[0:2] = m0
    m[2:] = m0_
    
    ##Mount initial C matrix
    C = np.zeros((p+2, p+2))
    C[0:2,0:2] = C0
    C[2:p+2, 2:p+2] = C0_
    
    ##Mount permutation matrix
    P = np.zeros((p,p))
    P[0:p-1, 1:p] = np.eye(p-1)
    P[p-1,0] = 1
    
    ##Mount Jordan form matrix
    J = np.zeros((2,2))
    J[0,0] = J[0,1] = J[1,1] = 1
    
    ##Mount G matrix
    G = np.zeros((p+2, p+2))
    G[2:p+2, 2:p+2] = P
    G[0:2,0:2] = J
    
    ##Mount F vector
    F = np.zeros(p+2)
    F[0] = F[2] = 1
    
    mm = []   ##Save all estimates of systemic mean
    CC = []   ##Save all estimates of the systemic variance
    SS = []   ##Save all estimates of the observation variance
    ff = []   ##Save all forecasts
    QQ = []   ##Save all forecast variances
    
    for i in range(Y.size):
        
        ##Correct vector of seasonal effecs its covariance matrix
        
        ##The if below was newcessary due to accumlation of errors
        ##It basically renormalize the vector of seasonal effects (which must sum to zero)
        ##And the covariance matrix, whose columns and ros must all sum to zero.
        U = C[2:p+2, 2:p+2].sum()   ## Alternatively: np.dot(np.ones(p), np.dot(R0, np.ones(p)))
        if not np.isclose(np.abs(U), 0.0):
            tr = np.trace(C[2:p+2, 2:p+2])    ##Save trace of covariance matrix of seasonal effects
            A_ = np.dot(C[2:p+2, 2:p+2], np.ones(p))/U
            m[2:] = m[2:]-m[2:].sum()*A_
            C[2:p+2, 2:p+2] = C[2:p+2, 2:p+2]-U*np.outer(A_, A_)
        
            ##Correct trace
            correction = tr/np.trace(C[2:p+2, 2:p+2])
            C[2:p+2, 2:p+2] = correction*C[2:p+2, 2:p+2]  
        
        ##Calculate evolution matrix for level of the series
        W_level = (1-d_level)/d_level*J.dot(C[0:2,0:2]).dot(J.T)   ##Determine evolution matrix for trend term
        W_seasonal = (1-d_seasonal)/d_seasonal*P.dot(C[2:p+2, 2:p+2]).dot(P.T)    ##Determine evolution matrix for seasonal terms
        W = np.zeros_like(C)
        W[0:2,0:2] = W_level
        W[2:p+2, 2:p+2] = W_seasonal
        y = Y[i]    ##Current observation
        ##Update posterior parameters
#        m, C, f, Q = update_posterior(y, m, C, F, G, W, V)
        m, C, f, Q, n, S = update_posterior_V_unknown(y, m, C, F, G, W, n, S)
#        print m
        
        mm.append(m)
        CC.append(C)
        ff.append(f)
        QQ.append(Q)
        SS.append(S)
        
    return np.array(mm), np.array(CC), np.array(ff), np.array(QQ), np.array(SS)


def second_order_polynomial(Y, m0, C0, n0=None, S0=None, d=0.9):
    
    """Second order polynomial dynamic linear model
    This dlm assumes that the constant observation variance is not known
    a priori, and updates an estimated St of it at each time t.
    Y - Data vector
    m0 - Initial prior mean level
    C0 - Initial prior level variance
    d - Discount rate used in determining the a priori variance R.
        Typical values between 0.8 and 1.0
    n0 - Initial estimate of the degrees of freedom
    S0 - Initial estimate of observation variance V"""
    
    ##Set observational variance parameters
    n = n0    ##Initial degrees of freedom
    S = S0    ##Initial estimate of observation variance
    
    ##Initial m vector
    m = m0
    
    ##Initial C matrix
    C = C0 
    
    ##Mount Jordan form matrix
    G = np.zeros((2,2))
    G[0,0] = G[0,1] = G[1,1] = 1

    ##Mount F vector
    F = np.zeros(2)
    F[0] = 1
    
    mm = []   ##Save all estimates of systemic mean
    CC = []   ##Save all estimates of the systemic variance
    SS = []   ##Save all estimates of the observation variance
    ff = []   ##Save all forecasts
    QQ = []   ##Save all forecast variances
    
    for i in range(Y.size):
        
        ##Calculate evolution matrix for level of the series
        W = (1-d)/d*C   ##Determine evolution matrix for trend term
        y = Y[i]    ##Current observation
        ##Update posterior parameters
#        m, C, f, Q = update_posterior(y, m, C, F, G, W, V)
        m, C, f, Q, n, S = update_posterior_V_unknown(y, m, C, F, G, W, n, S)
#        print m
        
        mm.append(m)
        CC.append(C)
        ff.append(f)
        QQ.append(Q)
        SS.append(S)
        
    return np.array(mm), np.array(CC), np.array(ff), np.array(QQ), np.array(SS)


def fourier_dlm(Y, m0, C0, n0=None, S0=None, T=None, d=0.9):
    
    """A second order polynomial dynamic linear model with
    trigonemetric terms in the regression vector F to
    model periodic patterns.
    
    The parameter vector of this DLM has four parameters:
    
    theta_1: level of the time series (mu)
    theta_2: the increase or decrease in level (beta)
    theta_3: the amplitude of the cosine function
    theta_4: the amplitude of the sine function
    
    This dlm assumes that the constant observation variance is not known
    a priori, and updates an estimated St of it at each time t.
    Y - Data vector
    m0 - Initial prior mean level
    C0 - Initial prior level variance
    d - Discount rate used in determining the a priori variance R.
        Typical values between 0.8 and 1.0
    T - The period length of the series  
    n0 - Initial estimate of the degrees of freedom
    S0 - Initial estimate of observation variance V"""
    
    ##Set observational variance parameters
    n = n0    ##Initial degrees of freedom
    S = S0    ##Initial estimate of observation variance
    
    ##Initial m vector
    m = m0
    
    ##Initial C matrix
    C = C0 
    
    ##Mount system matrix
    G = np.eye(4)
    G[0,1] = 1

    
    mm = []   ##Save all estimates of systemic mean
    CC = []   ##Save all estimates of the systemic variance
    SS = []   ##Save all estimates of the observation variance
    ff = []   ##Save all forecasts
    QQ = []   ##Save all forecast variances
    
    for i in range(Y.size):
        
        ##Calculate evolution matrix for level of the series
        W = (1-d)/d*C   ##Determine evolution matrix for trend term
        y = Y[i]    ##Current observation
        
        ##Current time
        t = i    ##
        
        ##Update posterior parameters
        m, C, f, Q, n, S = update_posterior_Fourier(y, m, C, G, W, n, S, t, T)
#        print m
        
        mm.append(m)
        CC.append(C)
        ff.append(f)
        QQ.append(Q)
        SS.append(S)
        
    return np.array(mm), np.array(CC), np.array(ff), np.array(QQ), np.array(SS)


def multiprocess_dlm(Y, m0, C0, n0=None, S0=None, T=None, d=None):
    
    """
    A multiprocess DLM which runs two dlms each time and computes the probabilities
    of each one.
    
    
    The employd DLMs are second order polynomial dynamic linear model with
    trigonometric terms in the regression vector F to
    model periodic patterns.
    
    The parameter vector of this DLM has four parameters:
    
    theta_1: level of the time series (mu)
    theta_2: the increase or decrease in level (beta)
    theta_3: the amplitude of the cosine function
    theta_4: the amplitude of the sine function
    
    This dlm assumes that the constant observation variance is not known
    a priori, and updates an estimated St of it at each time t.
    Y - Data vector
    m0 - Initial prior mean level
    C0 - Initial prior level variance
    d - Discount rate used in determining the a priori variance R.
        Typical values between 0.8 and 1.0
    T - The period length of the series  
    n0 - Initial estimate of the degrees of freedom
    S0 - Initial estimate of observation variance V"""
    
    ##Set observational variance parameters
    n_1 = n_2 = n0    ##Initial degrees of freedom
    S_1 = S_2 = S0    ##Initial estimate of observation variance
    
    
    ##Initial m vectors (1 is for first DLM and 2 is for the second DLM)
    m_1 = m_2 = m0
    
    ##Initial C matrices
    C_1 = C_2 = C0
    
    d_1 = d[0]    ##First discount factor
    d_2 = d[1]    ##Second discount factor
    
    T_1 = T[0]    ##Period of first DLM
    T_2 = T[1]    ##Period of second
    
    p_1 = 0.5      ##Initial probability of first DLM
    p_2 = 0.5      ##Initial probability of second DLM
    
    ##Mount system matrix
    G = np.eye(4)
    G[0,1] = 1

    p_1_list = []
    p_2_list = []
    
    
    for i in range(Y.size):
        
        ##Calculate evolution matrices for level of the series
        W_1 = (1-d_1)/d_1*C_1   ##Determine evolution matrix for trend term
        W_2 = (1-d_2)/d_2*C_2
        
        ##Take current observation
        y = Y[i]
        
        ##Current time
        t = i    ##
        
        ##Update posterior parameters
        m_1, C_1, f_1, Q_1, n_1, S_1 = update_posterior_Fourier(y, m_1, C_1, G, W_1, n_1, S_1, t, T_1)
        m_2, C_2, f_2, Q_2, n_2, S_2 = update_posterior_Fourier(y, m_2, C_2, G, W_2, n_2, S_2, t, T_2)
        
        ##COmpute likelihood of each dlm
        l_1 = norm.pdf(y, f_1, Q_1)
        l_2 = norm.pdf(y, f_2, Q_2)
        
        const_normalizacao = l_1*p_1+l_2*p_2
        
        ##Atualiza probabilidades de cada DLM
        
        p_1 = l_1*p_1/const_normalizacao
        p_2 = l_2*p_2/const_normalizacao
        
        p_1_list.append(p_1)
        p_2_list.append(p_2)
        
        
    return np.array(p_1_list), np.array(p_2_list)

def update_posterior_Fourier(y, m, C, G, W, n, S, t, T):
    
    """This function updates the posterior distribution for a univariate dlm
       Notice that the regression vector F uses a Fourier basis
       Inputs:
           n: Number of degrees of freedom
           S: Current estimate of observational variance
           T: Length of period in time series
           t: Current time"""
    
    ##Prior at t-1
    a = np.dot(G, m)
    R = np.dot(G, np.dot(C, G.T))+W
    
    ##Mount regression vector at time t
    F = np.zeros(4)
    F[0] = 1
    F[1] = 0
    F[2] = np.sin(2*np.pi/T*t)
    F[3] = np.cos(2*np.pi/T*t)
    
    ##One-step forecast distribution
    f = np.dot(F, a)
    Q = np.dot(F, np.dot(R,F))+S    
    ##Posterior at time t
#    A = np.dot(np.dot(R,F), np.linalg.inv(Q))    ##Adjustment factor
    A = (1/Q)*np.dot(R,F)    ##Adjustment factor
    e = y-f    ##Observational error (This is a scalar)
    ##Update observational variance parameters
    n = n+1
    S_new = S+S/n*(e**2/Q-1)
    m = a+e*A    ##Posterior mean
    C = S_new/S*(R-Q*np.outer(A,A))   ##Posterior covariance matrix
    
    return m, C, f, Q, n, S_new


def update_posterior(y, m, C, F, G, W, V, d=None):
    
    """This function updates the posterior distribution for a univariate dlm"""
    
    ##Prior at t-1
    a = np.dot(G, m)
    R = np.dot(G, np.dot(C, G.T))+W
    ##One-step forecast distribution
    f = np.dot(F, a)   ##Notice that F does not need to be transposed due to the dot operation
    Q = np.dot(F, np.dot(R,F))+V
    ##Posterior at time t
#    A = np.dot(np.dot(R,F), np.linalg.inv(Q))    ##Adjustment factor
    A = (1/Q)*np.dot(R,F)    ##Adjustment factor
    e = y-f    ##This is a scalar
    m = a+e*A    ##Posterior mean
    C = R-Q*np.outer(A,A)    ##Posterior covariance matrix
    
    return m, C, f, Q
    
def update_posterior_V_unknown(y, m, C, F, G, W, n, S):
    
    """This function updates the posterior distribution for a univariate dlm
       Inputs:
           n: Number of degrees of freedom
           S: Current estimate of observational variance"""
    
    ##Prior at t-1
    a = np.dot(G, m)
    R = np.dot(G, np.dot(C, G.T))+W
    ##One-step forecast distribution
    f = np.dot(F, a)
    Q = np.dot(F, np.dot(R,F))+S    
    ##Posterior at time t
#    A = np.dot(np.dot(R,F), np.linalg.inv(Q))    ##Adjustment factor
    A = (1/Q)*np.dot(R,F)    ##Adjustment factor
    e = y-f    ##Observational error (This is a scalar)
    ##Update observational variance parameters
    n = n+1
    S_new = S+S/n*(e**2/Q-1)
    m = a+e*A    ##Posterior mean
    C = S_new/S*(R-Q*np.outer(A,A))   ##Posterior covariance matrix
    
    return m, C, f, Q, n, S_new
    
    
def limiting_A(r):

    "Calculate limiting adaptive ratio A given signa-to-noise ratio r"    
    
    return r/2.0*(np.sqrt(1+4/np.float(r))-1)


def bayesian_smoother(G, m, C, m_bar, C_bar):
    
    """Estimate the marginal conditional distributions \teta_{t-k} | D_t
    
       inputs:
           G: sequence of system matrices from time 1 to t
           m: sequence of posterior mean vectors from time 1 to t
           C: sequence of posterior covariance matrices from time 1 to t
           m_bar: sequence of prior mean vectors from time 1 to t
           C_bar: sequence of prior covariance matrices from time 1 to t"""
    
    ##Initialization   
    t, n = m.shape    ##Number of time steps, number of OD pairs
    m_bar_k = np.zeros((t,n))           ##An array whose line k contains m_bar_t(-k) (the marginal conditional mean of \teta_{t-k} | D_t at time t-k)
    C_bar_k = np.zeros((t,n,n))         ####An array whose element k contains the covariance matrix C_bar_t(-k) (the marginal conditional covariance matrix of \teta_{t-k} | D_t at time t-k)
    m_bar_k[-1,:] = m[-1,:]          ##m_bar_t(0) = m_t    (marginal conditional mean m_bar_t(-k=0) equals posterior mean m_t)
    C_bar_k[-1,:,:] = C[-1,:,:]    ##C_bar_t(0) = C_t    (marginal conditional covariance matrix C_bar_t(-k=0) equals posterior covariance matrix C_t)
    T = t-1    ##Index of last elements, to facilitate expression understainding below
    
    ##Backward loop in order to recursively compute m_bar_t(-k) and C_bar_t(-k)
    for k in range(1, T+1):    ##Notice that last value of k = T    
#        JT = lin.solve(C_bar_s[t+1,:,:].T, C_s[t,:,:].T, sym_pos=True)    ##Calculates the transposed backwards adjust matrix
#        J = JT.T    ##Untranspose backward adjust matrix
        Bt_k = np.dot(np.dot(C[T-k, :, :], G[T-k+1,:,:].T), np.linalg.inv(C_bar[T-k+1, :, :]))    ##Notice that T = t-1 is the index of last value in the arrays, with t equals the total number of elements
        m_bar_k[T-k,:] = m[T-k, :]+np.dot(Bt_k, m_bar_k[T-k+1,:]-m_bar[T-k+1, :])
        C_bar_k[T-k,:,:] = C[T-k,:,:]+np.dot(np.dot(Bt_k, C_bar_k[T-k+1,:,:]-C_bar[T-k+1,:,:]), Bt_k.T)
        
    return m_bar_k, C_bar_k