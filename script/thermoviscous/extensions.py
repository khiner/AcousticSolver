"""Focused acceleration, element and broadband checks for ValidateThermoviscous."""
import runpy
import numpy as np
from .common import ROOT, run, complex_values
from .mesh import sphere
from .analytic import relative, translation_pressure


def quad_sphere(count, size, radius=1):
    local=[(-1,-1),(1,-1),(1,1),(-1,1),(0,-1),(1,0),(0,1),(-1,0),(0,0)][:size]
    vertices,faces,indices=[],[],{}
    for axis in range(3):
        for sign in (-1,1):
            normal=np.eye(3)[axis]*sign;u=np.eye(3)[(axis+1)%3];v=np.cross(normal,u)
            for i in range(count):
                for j in range(count):
                    nodes=[]
                    for x,y in local:
                        p=normal+(-1+(2*i+1+x)/count)*u+(-1+(2*j+1+y)/count)*v
                        p=p/np.linalg.norm(p)*radius;key=tuple(np.round(p,14))
                        if key not in indices: indices[key]=len(vertices);vertices.append(p)
                        nodes.append(indices[key])
                    faces.append(nodes)
    return np.array(vertices),np.array(faces)


def validate(out):
    results={}
    vertices,triangles=sphere(1,1)
    base={'vertices':vertices.tolist(),'triangles':triangles.tolist(),'frequency':100,'velocity':[0,0,.01],
          'quadrature':12,'tolerance':1e-9,'inner_tolerance':1e-11}
    acceleration={}
    for exterior in (True,False):
        scene=dict(base,exterior=exterior)
        prefix='' if exterior else 'interior-'
        dense=run(scene,out/(prefix+'dense'))
        variants=[('acoustic_backend',name,2e-4) for name in ('fmm','hmatrix')]
        if exterior: variants += [('method',name,1e-7) for name in ('direct1','direct4','direct10')]
        for key,name,tolerance in variants:
            value=run(dict(scene,**{key:name}),out/(prefix+name))
            error=relative(complex_values(value['acoustic_pressure']),complex_values(dense['acoustic_pressure']))
            assert error<tolerance,(name,error)
            acceleration[prefix+name]=error
    results['methods']=acceleration
    v,t=sphere(3,1)
    large=dict(base,vertices=v.tolist(),triangles=t.tolist(),compression_tolerance=1e-6)
    fast=run(dict(large,acoustic_backend='fmm'),out/'probes-fmm',export='operator-probes')
    hierarchical=run(dict(large,acoustic_backend='hmatrix'),out/'probes-hmatrix',export='operator-probes')
    probe_errors=[relative(complex_values(a[key]),complex_values(b[key]))
                  for a,b in zip(hierarchical['operator_probes'],fast['operator_probes']) for key in ('H','G')]
    assert max(probe_errors)<1e-6,probe_errors
    kernel_bytes=2*16*len(v)*len(t)*12
    assert hierarchical['acoustic_bytes']<.8*kernel_bytes
    results['compressed_operators']={'probe_errors':probe_errors,'stored_bytes':hierarchical['acoustic_bytes'],
                                    'uncompressed_quadrature_kernel_bytes':kernel_bytes}
    element_errors={}
    cases=[(f'quad{size}',size,(2,4) if size==9 else (1,2),'geometry',.15) for size in (4,8,9)]
    cases += [('discontinuous-triangle',6,(1,2),'linear',.1),
              ('discontinuous-quad',9,(1,2),'linear',.1),
              ('discontinuous-quadratic-triangle',6,(1,2),'quadratic',.01),
              ('discontinuous-quadratic-quad',9,(2,4),'quadratic',.01)]
    for name,size,levels,order,tolerance in cases:
        errors=[]
        for level in levels:
            v,t=sphere(level,1) if size==6 else quad_sphere(level,size)
            scene={'vertices':v.tolist(),'triangles' if size==6 else 'quadrilaterals':t.tolist(),
                   'frequency':100,'velocity':[0,0,.01],'tolerance':1e-10,'inner_tolerance':1e-12,
                   'physics_order':order,'discontinuous':order!='geometry',
                   'acoustic_backend':'dense' if order=='geometry' else 'fmm'}
            value=run(scene,out/f'{name}-{level}')
            expected=translation_pressure(np.array(value['collocation_points']),100)
            errors.append(relative(complex_values(value['acoustic_pressure']),expected))
            if order=='linear': assert value['divergence_residual']<1e-3,value['divergence_residual']
        assert errors[-1]<errors[0] and errors[-1]<tolerance,(name,errors)
        element_errors[name]=errors
    results['elements']=element_errors
    # Verify FFT sign and displacement regularization against a delayed analytic transfer.
    renderer=runpy.run_path(str(ROOT/'script/RenderThermoviscous'))
    synthesize=renderer['synthesize']
    rate,n=8000,512;f=np.fft.rfftfreq(n,1/rate);omega=2*np.pi*f;delay=5
    transfer=np.exp(1j*omega*delay/rate)[:,None]
    impulse=synthesize(f,transfer,rate,n)
    recovered=np.fft.rfft(impulse,axis=0)*np.exp(1j*omega*n/4/rate)[:,None]
    expected=np.conjugate((-1j*omega[:,None])*transfer)
    assert relative(recovered[1:-1],expected[1:-1])<1e-12
    results['fft_relative_error']=relative(recovered[1:-1],expected[1:-1])
    band=.5*(1+np.cos(np.pi*np.clip((f-.405*rate)/(.045*rate),0,1)))
    time=np.arange(2000)/rate;z=(time-.05)*400
    excitation=(1-2*z*z)*np.exp(-z*z)
    audio=renderer['convolve'](excitation,synthesize(f,transfer*band[:,None],rate,n),rate,n)[:,0]
    expected=np.pad(excitation,(delay,n-1-delay))
    error=relative(audio,expected)
    assert error<3e-4,error
    results['convolution_relative_error']=error
    return results
