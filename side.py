import cv2

print(f"OpenCV: {cv2.__version__}")

build_info = cv2.getBuildInformation()

cuda_unavailable = "Unavailable:                 alphamat cannops cudaarithm" in build_info
print(f"CUDA modules in this build: {'NO' if cuda_unavailable else 'UNKNOWN'}")

print(f"OpenCL supported by build: {cv2.ocl.haveOpenCL()}")
cv2.ocl.setUseOpenCL(True)
print(f"OpenCL enabled at runtime: {cv2.ocl.useOpenCL()}")

if cv2.ocl.haveOpenCL() and cv2.ocl.useOpenCL():
	print("GPU path: OpenCL/UMat active")
else:
	print("GPU path: CPU fallback")
