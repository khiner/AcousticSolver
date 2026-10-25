"""Independent spherical reference solutions and relative-error checks."""
import numpy as np
from scipy.special import spherical_jn


def relative(a,b):
    if not np.isfinite(a).all() or not np.isfinite(b).all():
        raise AssertionError('Nonfinite comparison')
    return float(np.linalg.norm(a-b)/max(np.linalg.norm(b),1e-300))


def constants(f):
    rho,c,mu,cp,cv,conductivity = 1.199238418502307,343.98620840428265,1.8126668012737552e-5,1012.3677822977633,722.5520495555156,.02517780857486048
    omega = 2*np.pi*f
    gamma,k = cp/cv,omega/c
    lh,lv = conductivity/(rho*cp*c),(.6+4/3)*mu/(rho*c)
    lvhp = (gamma-1)*(lh-lv)
    ka2,kh2 = k*k/(1-1j*k*(lv+(gamma-1)*lh)-k*k*lh*lvhp),1j*k/(lh*(1+1j*k*lvhp))
    tau = lambda z:(gamma-1)/(rho*(cp-cv)*gamma*(1+1j*lh*z/k))
    phi = lambda z:-1j/(rho*omega*(1+1j*lv*z/k))
    return rho,c,k,np.sqrt(ka2),np.sqrt(kh2),np.sqrt(1j*omega*rho/mu),tau(ka2)/tau(kh2),phi(ka2),phi(kh2)


def translation_pressure(points,f,radius=1):
    rho,c,k,_,_,kv,_,_,_ = constants(f)
    b,beta = radius*k,radius*kv
    a1 = -.01/(3j*k)*b**3*np.exp(-1j*b)*(3*beta+3j-1j*beta**2)/(beta**2*(b*b-2)-b*b+1j*(beta*b*b+2*b*beta**2))
    r = np.linalg.norm(points,axis=1)
    h1 = -np.exp(1j*k*r)*(k*r+1j)/(k*r)**2
    return -3*rho*c*k*a1*h1*points[:,2]/r


def radial_pressure(points,f,radius,exterior):
    _,_,_,ka,kh,_,ratio,phia,phih = constants(f)
    derivative = (lambda k:1j*k-1/radius) if exterior else (lambda k:k/np.tan(k*radius)-1/radius)
    pa = .01/(phia*derivative(ka)-ratio*phih*derivative(kh))
    r = np.linalg.norm(points,axis=1)
    if exterior:
        mode = lambda k:radius/r*np.exp(1j*k*(r-radius))
    else:
        mode = lambda k:spherical_jn(0,k*r)/spherical_jn(0,k*radius)
    return pa, pa*(mode(ka)-ratio*mode(kh))
