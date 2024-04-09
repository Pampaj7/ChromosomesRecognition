import cv2

# Check OpenCV version
print(cv2.__version__)

# Try importing the ximgproc module
try:
    ximgproc = cv2.ximgproc
    print("ximgproc module is available.")
except AttributeError:
    print("ximgproc module is not available.")
