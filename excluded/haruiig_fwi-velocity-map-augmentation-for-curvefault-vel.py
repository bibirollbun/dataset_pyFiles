# plotting utils
import numpy as np
import matplotlib.pyplot as plt

# for plotting
def plot_vel(vel_arr: np.ndarray, title: str = "Velocity Model"):
    import matplotlib.pyplot as plt
    assert vel_arr.ndim == 2
    fig, ax = plt.subplots(figsize=(6, 6))
    im = ax.imshow(vel_arr, aspect="auto", cmap="seismic", vmin=1500, vmax=4500)
    ax.set_title(title)
    ax.set_xlabel("X")
    ax.set_ylabel("Z")
    plt.colorbar(im)
    plt.show()
    plt.close(fig)


# Transformation corresponding to Equation (A)
def transform_curve(c: np.ndarray, a: float, k: float) -> np.ndarray:
    """
    Transformation based on the formula: c_i(x,y) = c_{i-1}(x, y + a*sin(2πkx)).

    Args:
        c (np.ndarray): Input 2D velocity model, with shape (z, x).
        a (float): Amplitude of the distortion in pixels. A positive value has the effect of lifting the layers upwards.
        k (float): Wavenumber of the distortion. This corresponds to the number of waves across the entire width of the image.

    Returns:
        np.ndarray: The transformed 2D velocity model.
    """
    assert c.ndim == 2
    ZB, XB = c.shape

    ret: np.ndarray = np.zeros_like(c)
    
    # Set the value of each pixel in the output image based on the source z-coordinate.
    for x in range(XB):
        for z in range(ZB):
            # Calculate the source z-coordinate.
            source_z: float = z + a * np.sin(2 * np.pi * k * (x / (XB - 1)))
            from_z: int = int(source_z)

            # Clamp the coordinate to be within the image bounds.
            if from_z < 0:
                from_z = 0
            elif from_z >= ZB:
                from_z = ZB - 1

            ret[z, x] = c[from_z, x]
    return ret


fvbs = np.load("/kaggle/input/waveform-inversion/train_samples/FlatVel_B/model/model1.npy")
cvbs = np.load("/kaggle/input/waveform-inversion/train_samples/CurveVel_B/model/model1.npy")
 
vel = fvbs[2][0] # shape: (70, 70)
plot_vel(vel, "FlatVelB #3 Map(original)")
plot_vel(transform_curve(transform_curve(vel, a=-3.5, k=0.8), a=3.5, k=2.5) , "Transformed FlatVelB #3 Map(a=-3.5, k=0.8 and a=3.5, k=2.5)")

plot_vel(cvbs[3][0], "CurveVel_B Map(for reference)")


# Transformation corresponding to Equation B
def transform_fault(c: np.ndarray, a: float, k: float, s: float, s2: float, f_a: float, f_b: float, apply_down_side: bool=True) -> np.ndarray:
    """
    Applies a fault-like transformation based on a dividing line f(x).

    Let f(x) be the fault line, defined as: f(x) = f_a * x + f_b.
    The transformation is based on the formula:
    c_i(x,y) = c_{i-1}(x + s, y + a * sin(2πkx) + s')  (if condition is met)
    c_i(x,y) = c_{i-1}(x, y)                         (otherwise)

    Args:
        c (np.ndarray): Input 2D velocity model, with shape (z, x).
        a (float): Amplitude of the distortion in pixels. A positive value lifts the layers upwards.
        k (float): Wavenumber of the distortion. Corresponds to the number of waves across the image width.
        s (float): Shift amount in the x-direction (in pixels).
        s2 (float): Shift amount in the y-direction (in pixels), corresponding to s' in the paper.
        f_a (float): Slope of the fault line f(x).
        f_b (float): Y-intercept of the fault line f(x).
        apply_down_side (bool):
            If True, applies the transformation to the region y >= f(x) (the area below the line).
            If False, applies the transformation to the region y < f(x) (the area above the line).
    """
    assert c.ndim == 2
    ZB, XB = c.shape

    ret: np.ndarray = np.zeros_like(c)
    
    # Iterate through each pixel of the output image.
    for x in range(XB):
        for z in range(ZB):
            # Calculate the fault line's position at the current x.
            f_x: float = f_a * x + f_b
            
            # Determine if the current pixel (z, x) is in the region to be transformed.
            should_apply = (z >= f_x and apply_down_side) or (z < f_x and not apply_down_side)

            if should_apply:
                # Calculate the source coordinates from which to pull the velocity value.
                source_z: float = z + a * np.sin(2 * np.pi * k * (x / (XB - 1))) + s2
                source_x: float = x + s

                from_z: int = int(source_z)
                from_x: int = int(source_x)

                # Clamp the coordinates to be within the image bounds.
                from_z = max(0, from_z)
                from_z = min(ZB - 1, from_z)
                from_x = max(0, from_x)
                from_x = min(XB - 1, from_x)

                ret[z, x] = c[from_z, from_x]
            else:
                # If not in the transformed region, keep the original value.
                ret[z, x] = c[z, x]
    return ret


cfbs = np.load("/kaggle/input/waveform-inversion/train_samples/CurveFault_B/vel6_1_0.npy")

arr = fvbs[2][0]
arr = transform_fault(arr, a=4.2, k=3.1, s=15, s2=10, f_a=6.5, f_b=-200, apply_down_side=False)
arr = transform_fault(arr, a=2.4, k=2.1, s=2, s2=-2, f_a=1.2, f_b=-20, apply_down_side=False)
arr = transform_fault(arr, a=-3.5, k=1.2, s=5, s2=-5, f_a=2.0, f_b=-10, apply_down_side=True)
plot_vel(arr, title="Transformed Vel Map")
plot_vel(fvbs[2][0], title="Original Vel Map(FlatVel_B)")
plot_vel(cfbs[0][0], title="CurveFault_B Vel Map(For Reference)")


