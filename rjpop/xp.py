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
    from scipy import special
    
    use_cupy = False