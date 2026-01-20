import openxr as xr
print("openxr loaded from:", xr.__file__)
print("has XR_ constants:", any(n.startswith("XR_") for n in dir(xr)))
print("sample names:", [n for n in dir(xr) if n.startswith("XR_")][:10])
