# universal numpy/cupy
try:
    import cupy as xp

    # xp.cuda.runtime.setDevice(0)
    from cupyx import scatter_add
    from cupyx.scipy import special

    use_cupy = True 

except (ModuleNotFoundError, ImportError) as e:
    import numpy as xp
    print(f'Using cpu: {e}')
    def scatter_add(a, slices, value):
        xp.add.at(a, slices, value)
    
    use_cupy = False

EPS = float( xp.finfo(xp.float64).eps * 2 )
INF = float( xp.sqrt(xp.finfo(xp.float64).max) / 10 )