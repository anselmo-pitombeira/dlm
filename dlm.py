
# -*- coding: utf-8 -*-
"""
Created on Sat Sep 19 16:11:47 2015

@author: Anselmo
"""

import numpy as np
from scipy.stats import norm, multivariate_normal, multivariate_t
from scipy.stats import t as t_student
from scipy.linalg import block_diag, solve, LinAlgError
import scipy.linalg as lin
from numpy.linalg import slogdet
import numba as nb
from numba.experimental import jitclass

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



# def multiprocess_dlm(Y, m0, C0, n0=None, S0=None, T=None, d=None):
    
#     """
#     A multiprocess DLM which runs two dlms each time and computes the probabilities
#     of each one.
    
    
#     The employd DLMs are second order polynomial dynamic linear model with
#     trigonometric terms in the regression vector F to
#     model periodic patterns.
    
#     The parameter vector of this DLM has four parameters:
    
#     theta_1: level of the time series (mu)
#     theta_2: the increase or decrease in level (beta)
#     theta_3: the amplitude of the cosine function
#     theta_4: the amplitude of the sine function
    
#     This dlm assumes that the constant observation variance is not known
#     a priori, and updates an estimated St of it at each time t.
#     Y - Data vector
#     m0 - Initial prior mean level
#     C0 - Initial prior level variance
#     d - Discount rate used in determining the a priori variance R.
#         Typical values between 0.8 and 1.0
#     T - The period length of the series  
#     n0 - Initial estimate of the degrees of freedom
#     S0 - Initial estimate of observation variance V"""
    
#     ##Set observational variance parameters
#     n_1 = n_2 = n0    ##Initial degrees of freedom
#     S_1 = S_2 = S0    ##Initial estimate of observation variance
    
    
#     ##Initial m vectors (1 is for first DLM and 2 is for the second DLM)
#     m_1 = m_2 = m0
    
#     ##Initial C matrices
#     C_1 = C_2 = C0
    
#     d_1 = d[0]    ##First discount factor
#     d_2 = d[1]    ##Second discount factor
    
#     T_1 = T[0]    ##Period of first DLM
#     T_2 = T[1]    ##Period of second
    
#     p_1 = 0.5      ##Initial probability of first DLM
#     p_2 = 0.5      ##Initial probability of second DLM
    
#     ##Mount system matrix
#     G = np.eye(4)
#     G[0,1] = 1

#     p_1_list = []
#     p_2_list = []
    
    
#     for i in range(Y.size):
        
#         ##Calculate evolution matrices for level of the series
#         W_1 = (1-d_1)/d_1*C_1   ##Determine evolution matrix for trend term
#         W_2 = (1-d_2)/d_2*C_2
        
#         ##Take current observation
#         y = Y[i]
        
#         ##Current time
#         t = i    ##
        
#         ##Update posterior parameters
#         m_1, C_1, f_1, Q_1, n_1, S_1 = update_posterior_Fourier(y, m_1, C_1, G, W_1, n_1, S_1, t, T_1)
#         m_2, C_2, f_2, Q_2, n_2, S_2 = update_posterior_Fourier(y, m_2, C_2, G, W_2, n_2, S_2, t, T_2)
        
#         ##COmpute likelihood of each dlm
#         l_1 = norm.pdf(y, f_1, Q_1)
#         l_2 = norm.pdf(y, f_2, Q_2)
        
#         const_normalizacao = l_1*p_1+l_2*p_2
        
#         ##Atualiza probabilidades de cada DLM
        
#         p_1 = l_1*p_1/const_normalizacao
#         p_2 = l_2*p_2/const_normalizacao
        
#         if p_1 >=1.0-1e-6:
#             p_1 = 0.99
#             p_2 = 0.01
            
#         if p_2 >=1.0-1e-6:
#             p_1 = 0.01
#             p_2 = 0.99
        
#         p_1_list.append(p_1)
#         p_2_list.append(p_2)
        
        
#     return np.array(p_1_list), np.array(p_2_list)


# def multiprocess_dlm(Y, m0, C0, n0=None, S0=None, T=None, d=None):
    
#     """
#     A multiprocess DLM which runs multiple dlms each time step
#     for a given time series Y and computes the probabilities
#     of each one.
    
    
#     The employd DLMs are second order polynomial dynamic linear model with
#     trigonometric terms in the regression vector F to
#     model periodic patterns.
    
#     The parameter vector of this DLM has four parameters:
    
#     theta_1: level of the time series (mu)
#     theta_2: the increase or decrease in level (beta)
#     theta_3: the amplitude of the cosine function
#     theta_4: the amplitude of the sine function
    
#     This dlm assumes that the constant observation variance is not known
#     a priori, and updates an estimated St of it at each time t.
#     Y - Data vector
#     m0 - Initial prior mean level
#     C0 - Initial prior level variance
#     d - Discount rate used in determining the a priori variance R.
#         Typical values between 0.8 and 1.0
#     T - Vector of period parameters for each DLM
#     n0 - Initial estimate of the degrees of freedom
#     S0 - Initial estimate of observation variance V"""
    
    
#     ##Set observational variance parameters
#     # n_1 = n_2 = n0    ##Initial degrees of freedom
#     # S_1 = S_2 = S0    ##Initial estimate of observation variance
    
    
#     ##Initial m vectors (1 is for first DLM and 2 is for the second DLM)
#     # m_1 = m_2 = m0
    
#     ##Initial C matrices
#     # C_1 = C_2 = C0
    
#     # d_1 = d[0]    ##First discount factor
#     # d_2 = d[1]    ##Second discount factor
    
    
    
    
#     T_1 = T[0]    ##Period of first DLM
#     T_2 = T[1]    ##Period of second
    
#     p_1 = 0.5      ##Initial probability of first DLM
#     p_2 = 0.5      ##Initial probability of second DLM
    
#     ##Mount system matrix
#     G = np.eye(4)
#     G[0,1] = 1

#     p_1_list = []
#     p_2_list = []
    
    
#     for i in range(Y.size):
        
#         ##Calculate evolution matrices for level of the series
#         W_1 = (1-d_1)/d_1*C_1   ##Determine evolution matrix for trend term
#         W_2 = (1-d_2)/d_2*C_2
        
#         ##Take current observation
#         y = Y[i]
        
#         ##Current time
#         t = i    ##
        
#         ##Update posterior parameters
#         m_1, C_1, f_1, Q_1, n_1, S_1 = update_posterior_Fourier(y, m_1, C_1, G, W_1, n_1, S_1, t, T_1)
#         m_2, C_2, f_2, Q_2, n_2, S_2 = update_posterior_Fourier(y, m_2, C_2, G, W_2, n_2, S_2, t, T_2)
        
#         ##COmpute likelihood of each dlm
#         l_1 = norm.pdf(y, f_1, Q_1)
#         l_2 = norm.pdf(y, f_2, Q_2)
        
#         const_normalizacao = l_1*p_1+l_2*p_2
        
#         ##Atualiza probabilidades de cada DLM
        
#         p_1 = l_1*p_1/const_normalizacao
#         p_2 = l_2*p_2/const_normalizacao
        
#         if p_1 >=1.0-1e-6:
#             p_1 = 0.99
#             p_2 = 0.01
            
#         if p_2 >=1.0-1e-6:
#             p_1 = 0.01
#             p_2 = 0.99
        
#         p_1_list.append(p_1)
#         p_2_list.append(p_2)
        
        
#     return np.array(p_1_list), np.array(p_2_list)

def multiprocess_dlm(Y, dlms):
    
    """
    A multiprocess DLM which runs multiple dlms each time step
    for a given time series Y and computes the probabilities
    of each one.
        
    Y - Time series as a numpy array
    dlms - list of DLMs to be run
    """
    
    n = len(dlms)
    
    probs = []
    
    ##Initialize prior probabilities of DLMs
    
    p = np.ones(n)/n
    
    for t in range(Y.shape[0]):
        
        ##Take current observation
        y = Y[t]
        
        likelihoods = []
        
        for dlm in dlms:
            
            a,R = dlm.predict_state()
            
            ##F = dlm.mount_regression_vector(t)
            
            ##F = dlm.F
            
            f,Q = dlm.predict_observation(a,R)
            
            dlm.update_state(y,a,R,f,Q)
            
            ##Compute likelihood of dlm
            ##likeli = norm.pdf(y, f, Q)
            likeli = multivariate_normal.pdf(y, f, Q)
            
            likelihoods.append(likeli)
            
            
        likelihoods = np.array(likelihoods)
        
        ##Update probability of each DLM
        ##Notice the use of Bayes theorem
        
        ##Computer normalization constant
        ##norm_const = (likelihoods*p).sum()
        norm_const = np.dot(likelihoods, p)
                
        p = likelihoods*p/norm_const
        
        ##Avoids probability to be exactly one or zero, which causes degeneration
        p[p>1.0-1e-12] = 1.0-1e-12
        p[p<0.0+1e-12] = 0.0+1e-12
        
        ##Renormalize probabilities. This is necessary since
        ##after the correction, the sum of probabilities will not be equals 1.
        
        ##Compute renormalization constant
        re_norm_const = p.sum()
        
        ##Renormalize
        p = p/re_norm_const
        probs.append(p)
        
    probs = np.array(probs)
        
    return probs


def multiprocess_matrixvariate_dlm(Y, dlms):
    
    """
    A multiprocess DLM which runs multiple dlms each time step
    for a given time series Y and computes the probabilities
    of each one.
    
    
    The employd DLMs are matrixvariate dynamic linear model.
        
    
    Y - Time series as a numpy array
    dlms - list of DLMs to be run
    """
    
    n = len(dlms)
    
    probs = []
    
    ##Initialize prior probabilities of DLMs
    
    p = np.ones(n)/n
    
    for t in range(Y.shape[0]):
        
        ##Take current observation
        y = Y[t]
        
        ##Make y  a column vector   (so that the transpose operation can be used)
        y = y.reshape(-1,1)
        
        likelihoods = []
        
        for dlm_model in dlms:
            
            a,R = dlm_model.predict_state()
            
            f,Q = dlm_model.predict_observation(a,R)
            
            ##Compute likelihood of dlm
            
            ##Notice that we turn y and f f into a (q,) shape array (requirement of the multivariate_t pdf)
            ##likeli = multivariate_t.pdf(x=y.squeeze(), loc=f.squeeze(), shape=Q*dlm_model.S, df = dlm_model.n)
               
            likeli = multivariate_t.pdf(x=y.squeeze(), loc=f.squeeze(), shape=Q*dlm_model.S, df = dlm_model.h)
            
            likelihoods.append(likeli)
            
            dlm_model.update_state(y,a,R,f,Q)
            
        likelihoods = np.array(likelihoods)
        
        ##norm_const = (likelihoods).sum()
        ##p = likelihoods/norm_const
        ##probs.append(p)
   
        
        ##Update probability of each DLM
        ##Notice the use of Bayes theorem
        
        ##Compute normalization constant
        norm_const = np.dot(likelihoods, p)
                
        p = likelihoods*p/norm_const
        
        ##Avoids probability to be exactly one or zero, which causes degeneration
        p[p>1.0-1e-12] = 1.0-1e-12
        p[p<0.0+1e-12] = 0.0+1e-12
        
        ##Renormalize probabilities. This is necessary since
        ##after the correction, the sum of probabilities will not be equals 1.
        
        ##Compute renormalization constant
        re_norm_const = p.sum()
        
        ##Renormalize
        p = p/re_norm_const
        probs.append(p)
        
    probs = np.array(probs)
        
    return probs


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

def compute_A_factor(R, F, Q):
    
    """
    A = RF^TQ^-1"""
    
    try:
        #X = solve(Q, F, sym_pos=True, check_finite=False)    ##X = F^t*Q^-1. Notice that Q is symmetric and positive-definite. Uses posv function from LAPACK
         X = solve(Q, F, check_finite=False)   
    except LinAlgError:
        print("F", F)
        print("Matrix Q", Q)
        raise LinAlgError
    
    X = X.T    ##Result from solve is X transposed! Untranspose X
    A = np.dot(R, X)    ##Adjustment Factor
    
    return A    ##This makes an external copy of A?


def log_likelihood(Y,ff,QQ):
    
    """
    Y - Numpy array of observations, T x n, in which T is the number
        of observations and n is the dimension of the observation vector
    """
    
    T = Y.shape[0]    ##Number of observations
    
    
    ##Log-likelihood value
    L = 0
    
    for t in range(T):
        
        y = Y[t]
        f = ff[t]
        Q = QQ[t]
        
        ##CALCULATE LOG-LIKELIHOOD AT TIME t
        
        ##First term of likelihood function
        sign, value = slogdet(Q)
        logdet = sign*value
        
        ##Deviation term (observed - predicted)
        e = y-f
        ##Calculate the quadratic term
        try:
            quad = np.dot(e, solve(Q, e, sym_pos=True, check_finite=False))
        except LinAlgError:
            
            ##print("\nResort to pseudo-inverse")
            
            pinv = np.linalg.pinv(Q)
            quad = np.dot(e, np.dot(pinv, e))
            
            singular_values = np.linalg.svd(Q,compute_uv=False)
            tol = singular_values.max()*np.finfo(float).eps*y.size    ##Tolerance for singular values
            pseudo_det = singular_values[singular_values>tol].prod()
            logdet = np.log(pseudo_det)
            
        ##Return likelihood for time t
        L = L-0.5*logdet-0.5*quad
        
    return L


def log_likelihood_matrixvariate_DLM(Y,ff,QQ,DD,hh,D0,h0):
    
    """
    Computes log likelihood of the data for the matrixvariate DLM.
    
    The log likelihood is the sum of the logarithms of multivarite Student's t density.
    
    Y: Numpy array of observations, T x n, in which T is the number
        of observations and n is the dimension of the observation vector
    """
    
    T = Y.shape[0]    ##Number of observations
    
    ##Log-likelihood value
    L = 0
    
    for t in range(T):
        
        y = Y[t]
        f = ff[t]
        Q = QQ[t]
        
        f = f.squeeze()   ##Turn f into a (q,) shape array
         
        if t == 0:
            D = D0    ##We neeed this since DD[0] is the D value at time t = 1, and we need the prior value of D at time t = 0 (this is, after the first update)
            h = h0    ##The same with hh, hh[0] is the value at time t = 1 (this is, after the first update)
        else:
            D = DD[t-1]    ##Notice that we take the prior parameters before update, and these are from time t-1
            h = hh[t-1]
        
        ##Compute estimate of \Sigma covariance matrix (mean value of inverse Wishart distribution)
        S = D/h
        
        ##Accumulate L
        L += multivariate_t.logpdf(x=y, loc=f, shape=Q*S, df = h)
            
        
    return L
    
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


spec = [

('m', nb.float64[:]),
('C', nb.float64[:,:]),
('n', nb.int64),
('S', nb.float64),
('d1', nb.float64),
('d2', nb.float64),
('T', nb.int64),
('h', nb.int64),
('G', nb.float64[:,:]),
('F', nb.float64[:]),
]

#@jitclass(spec)
class Fourier_dlm:
    
    """
    Defines a first order DLM with harmonics evolving over time.
    
    This a DLM with two main components:
        - The level and trend of the time series.
        - A vector of h harmonic components evolving over time.
    """
    
    def __init__(self,m0,C0,n0,S0,T,h=1,d1=0.99,d2=0.99):
        
        """
        m0 - Initial prior mean level
        C0 - Initial prior level variance
        n0 - Initial estimate of the degrees of freedom
        S0 - Initial estimate of observation variance V
        T - The period length of the trigonometric terms 
        h - Number of harmonic terms
        d1 - Discount factor used in determining the a priori variance R
             relative to the level of the time series. Typical values between 0.8 and 1.0
        d2 - Discount factor relative to the harmonics component.
            
        """
        
        self.m = m0
        self.C = C0
        self.n = n0
        self.S = S0
        self.d1 = d1
        self.d2 = d2
        self.T = T
        self.h = h
        
        ##MOUNT SYSTEM MATRIX
        
        ##Matrix of local level and trend terms
        G1 = np.zeros((2,2))
        G1[0,0] = G1[0,1] = G1[1,1] = 1
        ##G1 = np.eye(1)
        ##G1[0,1] = 1
        
        ##Matrices corresponding to harmonics
        H_matrices = []
        
        for j in range(1,h+1):
        
            omega = 2*np.pi*j/T    ##Angular frequencies
            
            H = np.eye(2)
            H[0,0] = np.cos(omega)
            H[0,1] = np.sin(omega)
            H[1,0] = -np.sin(omega)
            H[1,1] = np.cos(omega)
            
            H_matrices.append(H)
            
        H = lin.block_diag(*H_matrices)
        
        #H = matriz_bloco_diagonal(H_matrices)
        
        ##Glue matrices G1 and H together as a single
        ##block diagonal matrix:
            
        G = lin.block_diag(G1,H)
        #G = matriz_bloco_diagonal([G1,H])
        
        #print("Matriz G = ", G)
        
        self.G = G
        
        ##Mount regression vector
        ##This vector has
        ##2 components corresponding to the local
        ##level and linear trend
        ##2 components for each harmonic
        ##corresponding to the main harmonic and its conjugate
        ##Only the main harmonic is used in the regression vector
        ##This amounts to 2h terms
        
        F_size = 2+2*self.h
        
        F = np.zeros(F_size)
        F[0] = 1    ##Local level term
        F[1] = 0    ##Linear trend term
        
        for j in range(2, F_size-1, 2):
        ##We step 2 by 2
            
            F[j] = 1    ##Main harmonic
            F[j+1] = 0  ##Conjugate harmonic
            
        #print("Vetor de regressão = ", F)
        
        self.F = F
    
    
    def predict_state(self):
        
        """Generate the prior distribution of the state at next time.
           Notice that the regression vector F uses a Fourier basis
           Inputs:
               n: Number of degrees of freedom
               S: Current estimate of observational variance
               T: Length of period in time series
               t: Current time"""
               
        ##Parameters of prior distribution at time t
        ##(Prediction distribution of the state/parameters vectors)
        a = np.dot(self.G, self.m)
        
        ##Compute R matrix at time t (covariance matrix of prior distribution at time t)
        
        ##Compute P matrix (Covariance o r.v. G * \theta)
        P = np.dot(self.G, np.dot(self.C, self.G.T))
        
        ##Extract blocks of P corresponding to trend and periodic components
        ##P1 = P[0,0]*np.eye(1)  ##Notice this is a 1 x 1 matrix, not a scalar
        P1 = P[0:2,0:2]    ##Level and trend components
        P2 = P[2:,2:]      ##Periodic/Harmonic components
        
        ##Updates R matrices for components with different discount factors
        R1 = 1/self.d1*P1    ##Level and trend components
        R2 = 1/self.d2*P2    ##Periodic/Harmonic components
        
        ##R is equal P except for the updated blocks R1 and R2
        R = P   ##Notice that R is the same object in memory as P
        
        ##Correct R with the updated R1 and R2 components
        
        R[0:2,0:2] = R1
        R[2:,2:] = R2
        
        return a, R
    
    def predict_observation(self, a, R, F):
        
        ##One-step forecast distribution
        f = np.dot(F, a)
        Q = np.dot(F, np.dot(R,F))+self.S
        
        
        return f,Q
    
    def update_state(self,y,F,a,R,f,Q):
        
        ##Update observational variance parameters
        
        A = (1/Q)*np.dot(R,F)    ##Adjustment factor (Kalman gain)
        e = y-f                  ##Observational error (This is a scalar)
        self.n = self.n+1
        S_new = self.S+self.S/self.n*(e**2/Q-1)
        self.m = a+e*A    ##Posterior mean
        self.C = S_new/self.S*(R-Q*np.outer(A,A))   ##Posterior covariance matrix
        self.S = S_new
        
        
    def apply_DLM(self, Y):
        
        
        mm = []   ##Save all estimates of systemic mean
        CC = []   ##Save all estimates of the systemic variance
        SS = []   ##Save all estimates of the observation variance
        ff = []   ##Save all forecasts
        QQ = []   ##Save all forecast variances
        
        for t in range(Y.size):
            
            a,R = self.predict_state()
            
            F = self.F
            
            f,Q = self.predict_observation(a,R,F)
            
            ##Take current observation at time t
            y = Y[t]
            
            self.update_state(y,F,a,R,f,Q)
            
            
            ##Save statistics
            mm.append(self.m)
            CC.append(self.C)
            ff.append(f)
            QQ.append(Q)
            SS.append(self.S)
            
        return np.array(mm), np.array(CC), np.array(ff), np.array(QQ), np.array(SS)
        
        
class dynamic_harmonic_dlm:
    
    """
    Defines a DLM with harmonics whose period T
    is dynamic and so evolves over time.
    
    This a DLM with two main components:
        - The level and trend of the time series.
        - A vector of h harmonic components evolving over time.
    """
    
    def __init__(self,m0,C0,n0,S0,h=1,d1=0.99,d2=0.99):
        
        """
        m0 - Initial prior mean vector of parameters
        C0 - Initial prior covariance matrix of parameters
        n0 - Initial estimate of the degrees of freedom
        S0 - Initial estimate of observation variance V
        h - Number of harmonic terms
        d1 - Discount factor used in determining the a priori variance R
             relative to the level of the time series. Typical values between 0.8 and 1.0
        d2 - Discount factor relative to the harmonics component.
            
        """
        
        self.m = m0
        self.C = C0
        self.n = n0
        self.S = S0
        self.d1 = d1
        self.d2 = d2
        ##self.T = T
        self.h = h
        
        
        
        ##Mount regression vector
        ##This vector has
        ##2 components corresponding to the local
        ##level and linear trend
        ##2 components for each harmonic
        ##corresponding to the main harmonic and its conjugate
        ##Only the main harmonic is used in the regression vector
        ##This amounts to 2h terms
        
        F_size = 3+2*self.h
        
        F = np.zeros(F_size)
        F[0] = 1    ##Local level term
        F[1] = 0    ##Linear trend term
        F[2] = 0    ##Period term
        
        for j in range(3, F_size-1, 2):
        ##We step 2 by 2
            
            F[j] = 1    ##Main harmonic
            F[j+1] = 0  ##Conjugate harmonic
            
        ##print("Vetor de regressão = ", F)
        
        self.F = F
        
    def mount_system_matrix(self):
        
        """
        We need this since system matrix G_t is dynamic,
        so it has to be computed at each time step.
        """
        
        ##MOUNT SYSTEM MATRIX
        
        ##Matrix of local level, trend and period terms
        G1 = np.eye(3)    ##Identity matrix
        G1[0,1] = 1
        
        ##Take current estimate of period
        
        T = self.m[2]   ##Notice that T is the third component of the parameter vector
        
        ##Correct for possibly negative periods:
            
        if T < 0:
            T = 1
        
        ##Matrices corresponding to harmonics
        H_matrices = []
        
        for j in range(1,self.h+1):
        
            omega = 2*np.pi/T*j    ##Angular frequencies
            
            H = np.eye(2)
            H[0,0] = np.cos(omega)
            H[0,1] = np.sin(omega)
            H[1,0] = -np.sin(omega)
            H[1,1] = np.cos(omega)
            
            H_matrices.append(H)
            
        H = block_diag(*H_matrices)
        
        ##Glue matrices G1 and H together as a single
        ##block diagonal matrix:
            
        G = block_diag(G1,H)
        
        ##print("Matriz G = ", G)
        
        return G
        
    def predict_state(self):
        
        """Generate the prior distribution of the state at next time.
           Notice that the regression vector F uses a Fourier basis
           Inputs:
               n: Number of degrees of freedom
               S: Current estimate of observational variance
               T: Length of period in time series
               t: Current time"""
               
        ##Parameters of prior distribution at time t
        ##(Prediction distribution of the state/parameters vectors)
        
        G = self.mount_system_matrix()
        
        a = np.dot(G, self.m)
        
        ##Compute R matrix at time t (covariance matrix of prior distribution at time t)
        
        ##Compute P matrix (Covariance o r.v. G * \theta)
        P = np.dot(G, np.dot(self.C, G.T))
        
        ##Extract blocks of P corresponding to trend and periodic components
        ##P1 = P[0,0]*np.eye(1)  ##Notice this is a 1 x 1 matrix, not a scalar
        P1 = P[0:3,0:3]    ##Level, trend and period components
        P2 = P[3:,3:]      ##Periodic/Harmonic components
        
        ##Updates R matrices for components with different discount factors
        R1 = 1/self.d1*P1    ##Level and trend components
        R2 = 1/self.d2*P2    ##Periodic/Harmonic components
        
        ##R is equal P except for the updated blocks R1 and R2
        R = P   ##Notice that R is the same object in memory as P
        
        ##Correct R with the updated R1 and R2 components
        
        R[0:3,0:3] = R1
        R[3:,3:] = R2
        
        return a, R
    
    def predict_observation(self, a, R, F):
        
        ##One-step forecast distribution
        f = np.dot(F, a)
        Q = np.dot(F, np.dot(R,F))+self.S
        
        
        return f,Q
    
    
    def update_state(self,y,F,a,R,f,Q):
        
        ##Update observational variance parameters
        
        A = (1/Q)*np.dot(R,F)    ##Adjustment factor (Kalman gain)
        e = y-f                  ##Observational error (This is a scalar)
        self.n = self.n+1
        S_new = self.S+self.S/self.n*(e**2/Q-1)
        self.m = a+e*A    ##Posterior mean
        self.C = S_new/self.S*(R-Q*np.outer(A,A))   ##Posterior covariance matrix
        self.S = S_new
        
        
    def apply_DLM(self, Y):
        
        
        mm = []   ##Save all estimates of systemic mean
        CC = []   ##Save all estimates of the systemic variance
        SS = []   ##Save all estimates of the observation variance
        ff = []   ##Save all forecasts
        QQ = []   ##Save all forecast variances
        
        for t in range(Y.size):
            
            ##print("Period at time = "+str(t)+"=", self.m[2])
            
            print(self.m)
            
            a,R = self.predict_state()
            
            F = self.F
            
            f,Q = self.predict_observation(a,R,F)
            
            ##Take current observation at time t
            y = Y[t]
            
            self.update_state(y,F,a,R,f,Q)
            
            
            ##Save statistics
            mm.append(self.m)
            
            CC.append(self.C)
            ff.append(f)
            QQ.append(Q)
            SS.append(self.S)
            
        return np.array(mm), np.array(CC), np.array(ff), np.array(QQ), np.array(SS)
        
    
    
class tvar_dlm:
    
    """
    Defines a time-varying autoregressive dlm.
    """
    
    def __init__(self,m0,C0,n0,S0,F0,p,d1=0.99):
        
        """
        m0 - Initial prior mean level
        C0 - Initial prior level variance
        n0 - Initial estimate of the degrees of freedom
        S0 - Initial estimate of observation variance V
        F0 - Initial vector of p observations
        p -  Order of the AR model
        d1 - Discount factor used in determining the a priori variance R
             relative to the level of the time series. Typical values between 0.8 and 1.0
            
        """
        
        ##State of the filter at any time t
        
        self.m = m0
        self.C = C0
        self.n = n0
        self.p = p
        self.S = S0
        self.d1 = d1
        self.F = F0
        
        ##SYSTEM MATRIX IS JUST AN p x p IDENTITY MATRIX
        G = np.eye(p)
        self.G = G
        
    def predict_state(self):
        
        """Generate the prior distribution of the state at next time.
           Notice that the regression vector F uses a Fourier basis
           Inputs:
               n: Number of degrees of freedom
               S: Current estimate of observational variance
               T: Length of period in time series
               t: Current time"""
               
        ##Parameters of prior distribution at time t
        ##(Prediction distribution of the state/parameters vectors)
        a = np.dot(self.G, self.m)
        
        ##Compute R matrix at time t (covariance matrix of prior distribution at time t)
        
        ##Compute P matrix (Covariance o r.v. G * \theta)
        P = np.dot(self.G, np.dot(self.C, self.G.T))
        
        ##Updates R matrix
        R = 1/self.d1*P    
        
        return a, R
    
    def predict_observation(self, a, R):
        
        ##One-step forecast distribution
        F = self.F
        f = np.dot(F, a)
        Q = np.dot(F, np.dot(R,F))+self.S
        
        
        return f,Q
    
    
    def update_state(self,y,a,R,f,Q):
        
        ##Update observational variance parameters
        
        
        A = (1/Q)*np.dot(R,self.F)    ##Adjustment factor (Kalman gain)
        e = y-f                  ##Observational error (This is a scalar)
        self.n = self.n+1
        S_new = self.S+self.S/self.n*(e**2/Q-1)
        self.m = a+e*A    ##Posterior mean
        self.C = S_new/self.S*(R-Q*np.outer(A,A))   ##Posterior covariance matrix
        self.S = S_new
        
        ##Update F vector (In AR model, F corresponds to the p prior observations of the time series)
        
        new_F = np.zeros_like(self.F)
        new_F[0] = y
        new_F[1:] = self.F[0:self.p-1]
        self.F = new_F
        
        
    def apply_DLM(self, Y):
        
        
        mm = []   ##Save all estimates of systemic mean
        CC = []   ##Save all estimates of the systemic variance
        SS = []   ##Save all estimates of the observation variance
        ff = []   ##Save all forecasts
        QQ = []   ##Save all forecast variances
        
        for t in range(Y.size):
            
            a,R = self.predict_state()
            
            f,Q = self.predict_observation(a,R)
            
            ##Take current observation at time t
            y = Y[t]
            
            self.update_state(y,a,R,f,Q)
            
            
            ##Save statistics
            mm.append(self.m)
            CC.append(self.C)
            ff.append(f)
            QQ.append(Q)
            SS.append(self.S)
            
        return np.array(mm), np.array(CC), np.array(ff), np.array(QQ), np.array(SS)
    
    
class latent_AR_dlm:
    
    """
    Defines a DLM with a latent autoregressive component.
    """
    
    def __init__(self,m0,C0,n0,S0,p,phi,d1=0.99,d2=0.99):
        
        """
        m0 - Initial prior mean level
        C0 - Initial prior level variance
        n0 - Initial estimate of the degrees of freedom
        S0 - Initial estimate of observation variance V
        p -  Order of the AR model
        phi - a p-vector of AR parameters
        d1 - Discount factor used in determining the a priori variance R
             relative to the level of the time series. Typical values between 0.8 and 1.0
        d2 - Discount factor used in determining the a priori variance R
                  relative to the autoregressive latent component.
            
        """
        
        ##State of the filter at any time t
        
        self.m = m0
        self.C = C0
        self.n = n0
        self.p = p
        self.phi = phi
        self.S = S0
        self.d1 = d1
        self.d2 = d2
        
        ##MOUNT SYSTEM MATRIX
        
        ##Matrix of local level and latent autoregressive terms
        G = np.zeros((1+p,1+p))
        G[0,0] = 1    ##Local level term
        G[1,1:] = phi 
        G[2:, 1:p] = np.eye(p-1)
        
        self.G = G
        
        F = np.zeros(1+p)   ##Level plus p AR terms
        F[0] = F[1] = 1
        
        self.F = F
        
        
    def predict_state(self):
        
        """Generate the prior distribution of the state at next time.
           Notice that the regression vector F uses a Fourier basis
           Inputs:
               n: Number of degrees of freedom
               S: Current estimate of observational variance
               T: Length of period in time series
               t: Current time"""
               
        ##Parameters of prior distribution at time t
        ##(Prediction distribution of the state/parameters vectors)
        a = np.dot(self.G, self.m)
        
        ##Compute R matrix at time t (covariance matrix of prior distribution at time t)
        
        ##Compute P matrix (Covariance o r.v. G * \theta)
        P = np.dot(self.G, np.dot(self.C, self.G.T))
        
        P1 = P[0,0]        ##Level components
        P2 = P[1:,1:]      ##AR components
        
        ##Updates R matrices for components with different discount factors
        R1 = 1/self.d1*P1    ##Level and trend components
        R2 = 1/self.d2*P2    ##Periodic/Harmonic components
        
        ##R is equal P except for the updated blocks R1 and R2
        R = P   ##Notice that R is the same object in memory as P
        
        ##Correct R with the updated R1 and R2 components
        
        R[0,0] = R1
        R[1:,1:] = R2
        
        return a, R
    
    def predict_observation(self, a, R):
        
        ##One-step forecast distribution
        F = self.F
        f = np.dot(F, a)
        Q = np.dot(F, np.dot(R,F))+self.S
        
        
        return f,Q
    
    
    def update_state(self,y,a,R,f,Q):
        
        ##Update observational variance parameters
        
        
        A = (1/Q)*np.dot(R,self.F)    ##Adjustment factor (Kalman gain)
        e = y-f                  ##Observational error (This is a scalar)
        self.n = self.n+1
        S_new = self.S+self.S/self.n*(e**2/Q-1)
        self.m = a+e*A    ##Posterior mean
        self.C = S_new/self.S*(R-Q*np.outer(A,A))   ##Posterior covariance matrix
        self.S = S_new
        
        
    def apply_DLM(self, Y):
        
        
        mm = []   ##Save all estimates of systemic mean
        CC = []   ##Save all estimates of the systemic variance
        SS = []   ##Save all estimates of the observation variance
        ff = []   ##Save all forecasts
        QQ = []   ##Save all forecast variances
        
        for t in range(Y.size):
            
            a,R = self.predict_state()
            
            f,Q = self.predict_observation(a,R)
            
            ##Take current observation at time t
            y = Y[t]
            
            self.update_state(y,a,R,f,Q)
            
            
            ##Save statistics
            mm.append(self.m)
            CC.append(self.C)
            ff.append(f)
            QQ.append(Q)
            SS.append(self.S)
            
        return np.array(mm), np.array(CC), np.array(ff), np.array(QQ), np.array(SS)
    
    
    
class multivariate_dlm:
    
    """
    Defines a general DLM. """
    
    def __init__(self,G,F,m0,C0,V,d=0.99):
        
        """
        m0 - Initial prior mean level
        C0 - Initial prior level variance
        V - Observational covariance matrix
        d - Discount factor used in determining the a priori variance R
        """
        
        ##Save initial state of the DLM
        self.m0 = m0    
        self.C0 = C0
        
        ##Parameters
        self.G = G
        self.F = F
        self.d = d
        self.V = V
        
        ##Running state
        self.m = m0
        self.C = C0
        
    
    def predict_state(self):
        
        """Generate the prior distribution of the state at next time."""
               
        ##Parameters of prior distribution at time t
        ##(Prediction distribution of the state/parameters vectors)
        a = np.dot(self.G, self.m)
        
        ##Compute R matrix at time t (covariance matrix of prior distribution at time t)
        
        ##Compute P matrix (Covariance o r.v. G * \theta)
        P = np.dot(self.G, np.dot(self.C, self.G.T))
                
        ##Updates R matrices for components with different discount factors
        R = 1/self.d*P    ##Level and trend components
        
        return a, R
    
    def predict_observation(self, a, R):
        
        ##One-step forecast distribution
        F = self.F
        f = np.dot(F, a)
        Q = np.dot(F, np.dot(R,F.T))+self.V
        
        
        return f,Q
    
    
    def update_state(self,y,a,R,f,Q):
        
        ##Update observational variance parameters 
        A = compute_A_factor(R, self.F, Q)   ##Adjustment factor (Kalman gain)
        ##A = np.dot(np.dot(R,self.F), np.linalg.inv(Q)) 
        e = y-f

        m = a+np.dot(A, e)    ##Posterior mean
        
        C = R - np.dot(A, np.dot(Q, A.T)) ##Posterior covariance matrix
    #    I = eye(R.shape[0])    ##Identity matrix
    #    C = dot(I-dot(A, F), R) ##Posterior covariance matrix (Alternative calculus, source: https://en.wikipedia.org/wiki/Kalman_filter#Overview_of_the_calculation)
    
        ##print("Shape C = ", C.shape)
        
        self.m = m
        self.C = C    
        
        
    def apply_DLM(self, Y):
        
        """
        Apply DLM to data Y.
        
        Y: 
        """
        
        
        mm = []   ##Save all estimates of systemic mean
        CC = []   ##Save all estimates of the systemic variance
        ff = []   ##Save all forecasts
        QQ = []   ##Save all forecast variances
        
        
        T = Y.shape[0]    ##Take total number of observations
        
        for t in range(T):
            
            a,R = self.predict_state()
            
            f,Q = self.predict_observation(a,R)
            
            ##Take current observation at time t
            y = Y[t]
            
            self.update_state(y,a,R,f,Q)
            
            
            ##Save statistics
            mm.append(self.m)
            CC.append(self.C)
            ff.append(f)
            QQ.append(Q)
            
            
        ##return mm,CC,ff,QQ    
        return np.array(mm), np.array(CC), np.array(ff), np.array(QQ)
    
    def restart_dlm(self):
        
        ##Make running state equal to initial state
        self.m = self.m0    #
        self.C = self.C0
    

class matrixvariate_dlm:
    
    """
    Defines a matrix variate DLM. In this kind of DLM,
    a q-dimensional multivariate vector y has
    any of its components $y_j, j =1, ..., q$,
    modeled as a univariate DLM with the same structure (same $F_t$ and $G_t$)
    Each n-dimensional vector of parameters
    $\theta_j, j =1, ... q$, is arranged as columns
    of a n x q matriz $\Theta$. The n x q matrix $\Omega_t$
    of evolution errors is assumed to follow a matrixnormal
    probability density.
    """
    
    def __init__(self,G,F,V,m0,C0,D0,h0,delta=0.99,lambd=0.99, fix_sigma=False):
        
        """
        n: Size of parameter vectors (same size for each univariate DLM)
        q: Size of observation vector (number of univariate DLMs)
        
        Prior parameters
            m0 - Initial prior mean matrix of the state matrix \Omega_t
            C0 - Initial prior covariance matrix of the state columns of state matrix \Omega_t
            D0 - Initial prior scale matrix of Wishart distribution for the inverse covariance matrix \Sigma
            h0 - Initial prior number of degrees of freedom in the Wishart distribution for the inverse covariance matrix \Sigma
        Structure parameters
            F - Regression vector of each univariate DLM (n x 1)
            G - System matrix of each DLM   (n x n)
            V - Constant observational variance of each univariate DLM (scalar)
            d - Discount factor used in determining the a priori variance matrix R
        """
        
        ##Save initial state of the DLM
        self.m0 = m0
        self.C0 = C0
        self.D0 = D0
        self.h0 = h0
        self.fix_sigma = fix_sigma
        
        ##Structural parameters
        self.G = G
        self.F = F.reshape(-1,1)    ##Make F a column vector
        self.delta = delta
        self.lambd = lambd
        self.V = V
        
        ##State variables, initialized as provided initial prior values
        #(notice that this is the estimated state at any time. Actual state is latent and inacessible)
        self.m = m0
        self.C = C0
        self.D = D0
        self.h = h0
        self.S = self.D/self.h
        
        #if fix_sigma == True:    ##Fix \Sigma covariance matrix
        #    self.S = self.D/self.h
        
    def predict_state(self):
        
        """Generate the prior distribution of the state matrix at next time t"""
               
        ##Parameters of prior distribution at time t
        ##(Prediction distribution of the state/parameters vectors)
        a = np.dot(self.G, self.m)   ##Notice "a" is a n x q matrix
        
        ##Compute R matrix at time t (covariance matrix of prior distribution at time t)
        
        ##Compute P matrix (Covariance og r.v. G * \theta)
        P = np.dot(self.G, np.dot(self.C, self.G.T))
                
        ##Update R matrix as P matrix inflated by the discount factor
        R = 1/self.delta*P
        
        return a, R
    
    def predict_observation(self, a, R):
        
        ##One-step forecast distribution
        F = self.F
        f = np.dot(F.T, a)    ##F is a (n x 1) column vector. f is (1 x q) row vector
        Q = np.dot(F.T, np.dot(R,F))+self.V ##Q is a scalar
        
        ##Make f a column vector
        f = f.T
        
        return f,Q    ##    
    
    
    def update_state(self,y,a,R,f,Q):
        
        ##Compute error term
        e = y-f
        
        ##Adjustment factor (Kalman gain), A is a n x 1 vector
        A = np.dot(R,self.F)/Q
        
        ##Posterior mean. Notice A is n x 1 and e.T is a 1 x q vector, so that m is n x q
        m = a+np.dot(A, e.T)       
        
        ##Posterior covariance matrix between components of parameter vector
        C = R - np.dot(A,A.T)*Q
        
        new_h = self.lambd*self.h+1
        #new_n = self.d2*self.n+1
        
        
        ##Update parameters of Wishart distribution only if \Sigma is not fixed
        if self.fix_sigma == False:
            D = self.lambd*self.D+np.dot(e,e.T)/Q    ##Update scale matrix D_t of inverse Wishart distribution
            h = new_h    ##Update degrees of freedom
        
        ##Save new state values
        self.m = m
        self.C = C
        
        if self.fix_sigma == False:
            self.D = D
            self.h = h
            self.S = self.D/self.h
        
    def apply_DLM(self, Y):
        
        """
        Apply DLM to data Y.
        
        Y: 
        """
        
        
        mm = []   ##Save all estimates of systemic mean
        CC = []   ##Save all estimates of the systemic variance
        ff = []   ##Save all forecasts
        QQ = []   ##Save all forecast variances
        ##SS = []   ##Save all covariance matrices
        ##nn = []   ##Save all degrees of freedom
        DD = []   ##Save all scale matrices
        hh = []   ##Save all degrees of freedom
           
        T = Y.shape[0]    ##Take total number of observations
        
        for t in range(T):
            
            a,R = self.predict_state()
            
            f,Q = self.predict_observation(a,R)
            
            ##Take current observation at time t
            y = Y[t]
            
            ##Make y  a column vector   (so that the transpose operation can be used)
            y = y.reshape(-1,1)
            
            self.update_state(y,a,R,f,Q)
            
            ##Save statistics
            mm.append(self.m)
            CC.append(self.C)
            ff.append(f)
            QQ.append(Q)
            DD.append(self.D)
            hh.append(self.h)
            
        return np.array(mm), np.array(CC), np.array(ff), np.array(QQ),np.array(DD), np.array(hh)
    
    def restart_dlm(self):
        
        ##Make running state equal to initial state
        self.m = self.m0
        self.C = self.C0
        ##self.S = self.S0
        ##self.n = self.n0
        self.D = self.D0
        self.h = self.h0
        
class random_walk_DLM(multivariate_dlm):
       
    def __init__(self,m0,C0,V,d=0.99):
        
        """
        A simple multivariate random walk dynamic linear model.
        
        This DLM assums the time-varying mean vector \mu_t of the
        multivariate time series evolves as a random walk. No other features,
        such as trend or seasonality/periodicities are assumed.
        
        m0 - Initial prior mean level
        C0 - Initial prior level variance
        V - Observational covariance matrix
        d - Discount factor used in determining the a priori variance R
        """
        
        ##Mount G and F matrices
        
        ##Matrix of local level
        n = len(m0)
        G = np.eye(n)
        
        m = V.shape[0]
        F = np.eye(m)
        
        
        ##Initializes parent DLM
        multivariate_dlm.__init__(self,G,F,m0,C0,V,d)

    
class trend_matrixvariate_DLM(matrixvariate_dlm):
       
    def __init__(self,V,m0,C0,D0,h0,delta=0.99,lambd=0.99,fix_sigma=False):
        
        """
        A second order matrix variate dynamic linear model.
        
        m0 - Initial prior mean 
        C0 - Initial prior covariance matrix
        V - Observational variance
        delta- Discount factor to state parameters
        lambd - Discount factor coupling covariance matrix
        """
        
        ##Mount G and F matrices
        
        ##Second order DLM (2 state parameters: mean level and trend)
        nn = 2                     ##Dimension of state vector
        
        ##System matrix for the second order DLM (a Jordan matrix with order 2)
        G = np.eye(nn)
        G[0,1] = 1
        

        ##Regression vector for the second order DLM (observation is equals to the mean level)
        F = np.zeros(nn)
        F[0] = 1
        
        ##Initializes parent DLM
        matrixvariate_dlm.__init__(self,G,F,V,m0,C0,D0,h0,delta,lambd, fix_sigma=fix_sigma)


class trend_DLM(multivariate_dlm):
       
    def __init__(self,m0,C0,V,d=0.99):
        
        """
        
        A multivariate dynamic linear model with a trend component/feature.
        
        This DLM assums the time-varying mean vector \mu_t of the
        multivariate time series evolves as a random walk with
        an additional trend feature.
        
        m0 - Initial prior mean level
        C0 - Initial prior level variance
        V - Observational covariance matrix
        d - Discount factor used in determining the a priori variance R
        """
        
        ##Mount G and F matrices
        
        n = len(m0)    ##Size of state vector
        m = V.shape[0] ##Size of observation vector
        
        G_matrices = []
        
        for j in range(int(n/2)):
            
            G = np.eye(2)
            ##Add the tren term to G:
            G[0,1] = 1
            
            G_matrices.append(G)
            
        G = block_diag(*G_matrices)
        
        
        ##Mount F matrix
        F = np.zeros((m,n))
        
        for j in range(m):
            F[j,2*j] = 1
        
        ##print("F = ", F)
        
        ##Initializes parent DLM
        multivariate_dlm.__init__(self,G,F,m0,C0,V,d)
        

       
class mixture_Fourier_DLM:

    """
    Defines a mixture of univariate dynamic linear models (a multiprocess DLM)
    with Fourier components.
    
    It runs multiple FOurier dlms each time step for a given time series Y and computes the probabilities
    of each one. The one-step prediction is given by a mixure of the predictions of each DLM.
        
    Y - Time series as a numpy array
    dlms - list of DLMs to be run
    """

    def __init__(self, m0,C0,n0,S0,T_min,T_max,n = 10,h=2,d1=0.99,d2=0.995):

        """
        n: Number of DLMs in the mixture.
        m0 - Initial prior mean level
        C0 - Initial prior level variance
        n0 - Initial estimate of the degrees of freedom
        S0 - Initial estimate of observation variance V
        T_min - The minimum period length of the trigonometric terms
        T_max - The maximum period length of the trigonometric terms    
        h - Number of harmonic terms
        d1 - Discount factor used in determining the a priori variance R relative to the level of the time series. Typical values between 0.8 and 1.0
        d2 - Discount factor relative to the harmonics component.
        """
        
        ##Partition the interval of periods in n points
        self.Ts = np.linspace(T_min,T_max,n)
        
        ##Container for component DLMs
        self.dlms = []
        
        ##Probability state of the mixture DLM. All DLMs equally probable at the start
        self.p = np.ones(n)/n
        
        for i in range(n):
        
            T = self.Ts[i]  
            dlm_model = Fourier_dlm(m0,C0,n0,S0,T,h,d1,d2)
            self.dlms.append(dlm_model)
    
    def one_step_forecast(self):
    
        """
        Produce a forecast from the current states of the component DLMs.
        """
        
        forecasts = np.empty(len(self.dlms))
        q_variances = np.empty(len(self.dlms))
        
        ##Predict state and forecast for each component dlm
        for i, dlm_model in enumerate(self.dlms):
        
            a,R = dlm_model.predict_state()
            f,Q = dlm_model.predict_observation(a,R,dlm_model.F)
            
            ##Save forecast
            forecasts[i] = f
            q_variances[i] = Q
            
        ##Compute mixed forecast and variances
        mix_forecast = np.dot(forecasts, self.p)
        mix_variance = np.dot(q_variances, self.p)
        
        return mix_forecast, mix_variance
        
        
    def update_state(self, y):
    
        """
        Update state of the mixture model given an observation y.
        """
        
        ##Container for likelihoods of each DLM
        likelihoods = np.empty(len(self.dlms))
                
        for i,dlm_model in enumerate(self.dlms):
                          
            a,R = dlm_model.predict_state()
            f,Q = dlm_model.predict_observation(a,R,dlm_model.F)
            
            ##Update state of component model
            dlm_model.update_state(y,dlm_model.F,a,R,f,Q)
            
            ##Compute likelihood of dlm
            likeli = t_student.pdf(y,df=dlm_model.n,loc=f, scale=Q)
            ##Save likelihood
            likelihoods[i] = likeli
             
        ##Update probability of each DLM
        ##Notice the use of Bayes theorem
        
        ##Take current probabilities od each dlm
        p = self.p
        
        ##Compute normalization constant
        norm_const = np.dot(likelihoods, p)
        
        ##Compute posterior probability
        p = likelihoods*p/norm_const
        
        ##Avoids probability to be exactly one or zero, which causes degeneration
        p[p>1.0-1e-12] = 1.0-1e-12
        p[p<0.0+1e-12] = 0.0+1e-12
        
        ##Renormalize probabilities. This is necessary since
        ##after the correction, the sum of probabilities will not be equals 1.
        
        ##Compute renormalization constant
        re_norm_const = p.sum()
        
        ##Renormalize
        p = p/re_norm_const
        
        ##Update mixture model state
        self.p = p
        
        return p
        
        
    def apply(self, Y):
    
        """
        Apply the mixture model to a given time series.
        Y: Time series
        """
    
        n = len(self.dlms)
        
        probs = []
        mixture_forecasts_all_t = []
        
        ##Initialize prior probabilities of DLMs  (all have equal probabilities
        p = np.ones(n)/n
        
        for t in range(Y.shape[0]):
        
            ##print("t =", t)
            
            ##Take current observation
            y = Y[t]
            
            ##Container for likelihoods of each DLM
            likelihoods = []
            
            ##Container for forecasts of each DLM
            forecasts = []
            
            for dlm_model in self.dlms:
                              
                a,R = dlm_model.predict_state()
                
                f,Q = dlm_model.predict_observation(a,R,dlm_model.F)
                
                dlm_model.update_state(y,dlm_model.F,a,R,f,Q)
                
                ##Compute likelihood of dlm
                likeli = t_student.pdf(y,df=dlm_model.n,loc=f, scale=Q)
                likelihoods.append(likeli)
                forecasts.append(f)
                
            ##Cast to numpy arrays
            likelihoods = np.array(likelihoods)
            forecasts = np.array(forecasts)
            
            ##Update probability of each DLM
            ##Notice the use of Bayes theorem
            
            ##Computer normalization constant
            ##norm_const = (likelihoods*p).sum()
            norm_const = np.dot(likelihoods, p)
                    
            p = likelihoods*p/norm_const
            
            ##Avoids probability to be exactly one or zero, which causes degeneration
            p[p>1.0-1e-12] = 1.0-1e-12
            p[p<0.0+1e-12] = 0.0+1e-12
            
            ##Renormalize probabilities. This is necessary since
            ##after the correction, the sum of probabilities will not be equals 1.
            
            ##Compute renormalization constant
            re_norm_const = p.sum()
            
            ##Renormalize
            p = p/re_norm_const
            
            ##Save probs at time t
            probs.append(p)
            
            ##Compute mixture of forecasts
            f_mixture = np.dot(p,forecasts)
            
            ##Save forecas at time t
            mixture_forecasts_all_t.append(f_mixture)
            
        probs = np.array(probs)
        mixture_forecasts_all_t = np.array(mixture_forecasts_all_t)
        
        return probs, mixture_forecasts_all_t
        
        
@nb.njit
def matriz_bloco_diagonal(matrizes):
    if len(matrizes) == 0:
        raise ValueError("A lista de matrizes não pode estar vazia")

    num_blocos = len(matrizes)

    # Calcule o tamanho total da matriz bloco diagonal
    tamanho_bloco = [matriz.shape[0] for matriz in matrizes]
    tamanho_matriz = sum(tamanho_bloco)

    # Inicialize a matriz bloco diagonal com zeros
    matriz_bloco = np.zeros((tamanho_matriz, tamanho_matriz))

    # Preencha os blocos diagonais com as matrizes
    start_row = 0
    start_col = 0
    for matriz in matrizes:
        end_row = start_row + matriz.shape[0]
        end_col = start_col + matriz.shape[1]
        matriz_bloco[start_row:end_row, start_col:end_col] = matriz
        start_row = end_row
        start_col = end_col

    return matriz_bloco