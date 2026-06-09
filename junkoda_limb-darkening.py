import numpy as np
import matplotlib.pyplot as plt
import math

Rs = 1     # radius of the star, i.e., lengths are in units of the stellar radius
Rp = 0.3   # radius of the planet


# Explain mu
# written by ChatGPT

R = 1                   # radius of the star
theta = np.radians(45)
L1 = 0.5                # length normal
L2 = 0.5                # length tangent

# Figure
fig, ax = plt.subplots()
origin = np.array([0, 0])

# Half circle
theta_vals = np.linspace(np.pi, 2 * np.pi, 300)
x_circle = R * np.cos(theta_vals)
y_circle = R * np.sin(theta_vals)
ax.plot(x_circle, y_circle, 'k')

# Angle theta
xP = R * np.cos(theta)
yP = -R * np.sin(theta)
ax.plot([0, xP], [0, yP], 'b')

# Line of sight
ax.plot([xP, xP], [yP, yP - L1], 'g')

# Tangent space
dx = -np.sin(theta)
dy = -np.cos(theta)
xL = xP + L2 * dx
yL = yP + L2 * dy
xR = xP - L2 * dx
yR = yP - L2 * dy
ax.plot([xL, xR], [yL, yR], 'r')

# Normal vector
n_len = 0.5
xn = xP + n_len * np.cos(theta)
yn = yP - n_len * np.sin(theta)
ax.annotate('', xy=(xn, yn), xytext=(xP, yP),
            arrowprops=dict(arrowstyle='->', color='black'))
ax.text(xn + 0.05, yn, 'n', fontsize=12)

# Annotate theta
# You can also use patches.Arc. I would not do this, ChatGPT is doing.
arc_theta = np.linspace(-np.pi / 2, -np.pi / 2 + theta, 100)
r_arc = 0.3
x_arc = xP + r_arc * np.cos(arc_theta)
y_arc = yP + r_arc * np.sin(arc_theta)
ax.plot(x_arc, y_arc, 'k')

dx = 0.1
theta_label_angle = -np.pi / 2 + theta / 2
x_theta_text = xP + (r_arc + dx) * np.cos(theta_label_angle)
y_theta_text = yP + (r_arc + dx) * np.sin(theta_label_angle)
ax.text(x_theta_text, y_theta_text, r'$\theta$', fontsize=12, ha='center', va='center')

# Annotate line of sight
los_arrow_len = 0.2
los_start = yP - L1 - 0.3
los_end = los_start + los_arrow_len
ax.annotate('', xy=(xP, los_end), xytext=(xP, los_start),
            arrowprops=dict(arrowstyle='->', color='darkgreen'))

ax.text(xP + 0.05, los_start + los_arrow_len / 2, 'line of sight', 
        fontsize=10, verticalalignment='center', color='darkgreen')

# Axis lines
x_arrow_len = 1.3
z_arrow_len = 1.3
arrow_style = dict(arrowstyle='->', color='lightgray', linewidth=1.5)

# x axis
ax.annotate('', xy=(x_arrow_len, 0), xytext=(0, 0), arrowprops=arrow_style)
ax.text(x_arrow_len + 0.05, 0, 'x', fontsize=12, color='gray', va='center')

# z axis (line of sight)
z_arrow_start = -R - 0.2
z_arrow_end = 0.2
ax.annotate('', xy=(0, z_arrow_end), xytext=(0, z_arrow_start),
            arrowprops=arrow_style)

ax.text(0, z_arrow_end + 0.05, 'z', fontsize=12, color='gray', ha='center')

# Plot range
ax.set_aspect('equal')
ax.set_xlim(-1.5, 1.5)
ax.set_ylim(-1.5, 0.5)
ax.axis('off')

plt.show()


R = Rs    # radius of the star
r = Rp    # radius of the planet
x = -0.8   # position of the star

fig, ax = plt.subplots()

# Parameter for drawing circle
# Again, ChatGPT is doing this, instead of patches.Circle
theta_vals = np.linspace(0, 2 * np.pi, 300)

x_circle = R * np.cos(theta_vals)
y_circle = R * np.sin(theta_vals)
plt.plot(x_circle, y_circle, color='orange')

# Circle r (planet)
x_circle = x + r * np.cos(theta_vals)
y_circle = r * np.sin(theta_vals)
ax.plot(x_circle, y_circle, color='black')

# Length x
arrow_style = dict(arrowstyle='-', color='blue', linewidth=1.5, shrinkA=0, shrinkB=0)
plt.annotate('', xy=(x, 0), xytext=(0, 0), arrowprops=arrow_style)

plt.text(x / 2, -0.15, 'd', fontsize=12, color='blue', ha='center')

# Length R
cost = (x**2 + R**2 - r**2) / (2 * x * R)
sint = math.sqrt(1 - cost ** 2)

arrow_style = dict(arrowstyle='-', color='green', linewidth=1.5)
plt.annotate('', xy=(cost, sint), xytext=(0, 0), arrowprops=arrow_style)
plt.text(x / 2, 0.3, '$R_s$', fontsize=12, color='green', ha='center', va='center')

# Length r
arrow_style = dict(arrowstyle='-', color='red', linewidth=1.5, shrinkA=0, shrinkB=0)
plt.annotate('', xy=(cost, sint), xytext=(x, 0), arrowprops=arrow_style)
plt.text(x -0.12 , 0.0, '$R_p$', fontsize=12, color='red', ha='center', va='center')

# Angle alpha
theta = math.atan2(sint, cost)
arc_theta = np.linspace(theta, np.pi, 100)
r_arc = 0.3
x_arc = r_arc * np.cos(arc_theta)
y_arc = r_arc * np.sin(arc_theta)
ax.plot(x_arc, y_arc, 'cyan')

# Angle beta
cos_beta = (x**2 + r**2 - R**2) / abs(2 * x * r)
sin_beta = math.sqrt(1 - cos_beta ** 2)
beta = math.atan2(sin_beta, cos_beta)
arc_theta = np.linspace(0, beta, 100)
r_arc = 0.1
x_arc = r_arc * np.cos(arc_theta)
y_arc = r_arc * np.sin(arc_theta)
ax.plot(x + x_arc, y_arc, 'magenta')

plt.text(0, 0.7, 'star', fontsize=12, color='orange', ha='center', va='center')
plt.text(x, -0.4, 'planet', fontsize=12, color='black', ha='left', va='center')
plt.text(0, 0, '$\\alpha$', fontsize=12, color='cyan', ha='left', va='center')
plt.text(x, -0.12, '$\\beta$', fontsize=12, color='magenta', ha='center', va='center')


# Figure range
ax.set_aspect('equal')
ax.set_xlim(-1.5, 1.5)
ax.set_ylim(-1.5, 1.5)
ax.axis('off')

plt.show()


def area(R, r, d):
    """
	Compute the area of overlapping disks with radii R and r; separated by a distance d.
    Pure mathematics.

    Args:
      R, r (array): Radii of two disks
      x (array): Distance betwen two centers

    Three arrays must have the same shape
    """
    assert R.shape == r.shape == d.shape

    d = np.abs(d)  # distance (d >= 0)

    # Case 1. Default: no overlap
    out = np.zeros_like(d)

    # Case 2. Planet completely inside the sterllar circle
    idx_inside = d <= np.abs(R - r)
    out[idx_inside] = np.pi * np.minimum(r, R)[idx_inside] ** 2
    
    # Case 3. Star and planet are overlapping
    idx = np.logical_and(np.abs(R - r) < d, d < R + r)
    d = d[idx]
    r = r[idx]
    R = R[idx]

    # Law of cosines
    cos_alpha = (d ** 2 + R ** 2 - r ** 2) / (2 * R * d)
    cos_beta = (r ** 2 + d ** 2 - R ** 2) / (2 * r * d)

    assert np.all(np.abs(cos_alpha) < 1.00001)
    assert np.all(np.abs(cos_beta) < 1.00001)
    cos_alpha = np.clip(cos_alpha, -1, 1)
    cos_beta = np.clip(cos_beta, -1, 1)

    # overlap = r x sin(beta) / 2
    overlap_sq = 2 * (R ** 2 * r ** 2 + r ** 2 * d ** 2 + d ** 2 * R ** 2) - (R ** 4 + r ** 4 + d ** 4)
    overlap_sq = np.clip(overlap_sq, 0, None)

    out[idx] = R ** 2 * np.arccos(cos_alpha) + r ** 2 * np.arccos(cos_beta) - 0.5 * np.sqrt(overlap_sq)
    return out


d = np.linspace(-2, 2, 201)
R = np.ones_like(d)
r = Rp * np.ones_like(d)

a = area(R, r, d)
delta = 1 - a / math.pi  # Fraction compared to star area

plt.figure(figsize=(6, 3))
plt.ylabel('Relative flux')
plt.title('No limb darkening; r/R=0.3')
plt.plot(d, delta, label='b=0')
plt.text(0, 0.92, 'â†“ Flat', ha='center')

# delta
delta1 = (Rp / Rs) ** 2
plt.axhline(1 - delta1, ls=':', color='gray', alpha=0.5)
arrow_style = dict(arrowstyle='<->', color='red', linewidth=1.5, shrinkA=0, shrinkB=0)

plt.annotate('', xy=(1.5, 1), xytext=(1.5, 1 - delta1), arrowprops=arrow_style)
plt.text(1.55, 1 - delta1 / 2, '$\\delta$', va='center', fontsize=14)

plt.show()


# Visualize impact parameter b
R = Rs
r = Rp
s = -0.5
b = -0.6

fig, ax = plt.subplots()

theta_vals = np.linspace(0, 2 * np.pi, 300)

# Star
x_circle = R * np.cos(theta_vals)
y_circle = R * np.sin(theta_vals)
plt.plot(x_circle, y_circle, color='orange')

# Planet
x_circle = s + r * np.cos(theta_vals)
y_circle = b + r * np.sin(theta_vals)
ax.plot(x_circle, y_circle, color='black')

# Orbit (linear approximation)
arrow_style = dict(arrowstyle='<-', color='gray', linewidth=1.5, shrinkA=0, shrinkB=0)
plt.annotate('s', xy=(-1.5, b), xytext=(1.5, b), arrowprops=arrow_style)
plt.text(s, b - 0.1, '(s, b)', fontsize=12, color='gray', ha='center', va='center')

arrow_style = dict(arrowstyle='-', color='red', linewidth=1.5, shrinkB=0)
plt.annotate('', xy=(0, 0), xytext=(0, b), arrowprops=arrow_style)
plt.text(0.04, b / 2, 'b', fontsize=12, color='red', ha='left', va='center')

arrow_style = dict(arrowstyle='-', color='blue', linewidth=1.5, shrinkB=0)
plt.annotate('', xy=(0, 0), xytext=(s, b), arrowprops=arrow_style)
plt.text(s / 2 - 0.1, b / 2 + 0.06, 'd', fontsize=12, color='blue', ha='center', va='center')

# Figure
plt.title('Impact parameter b: $d^2 = s^2 + b^2$')
ax.set_aspect('equal')
ax.set_xlim(-1.5, 1.5)
ax.set_ylim(-1.5, 1.5)
ax.axis('off')

plt.show()


plt.figure(figsize=(6, 3))
plt.ylabel('Relative flux')
plt.title('Impact paramter b for off-center trajectory')

d = np.linspace(-2, 2, 201)
R = np.ones_like(d)
r = Rp * np.ones_like(d)

a = area(R, r, d)
plt.plot(d, 1 - a / math.pi, label='b=0')

# Add impact paramber b
b = 0.6
s = np.linspace(-2, 2, 201)  # coordinate along the planetary orbit (linear approximation)
xx_b = np.sqrt(b ** 2 + d ** 2) * np.sign(d)
a_b = area(R, r, xx_b)

plt.plot(s, 1 - a_b / math.pi, '--', label=('b=%.1f' % b))
plt.xlabel('s')
plt.legend()
plt.show()


def intensity(d, c):
    """
    Compute the stelar intensity with Claret 4-parameter non-linear
    
    Args:
      d (array): Distance between stellar center and planet center in units of stellar radius Rs=1
      c (4, ): Limb darkening coefficent, 4-parameter nonlinear + normalization
    """
    assert len(c) == 4

    # Overall normalization, such that, the total intensity is 1
    norm = (0.5 - c[0] / 10 - c[1] / 6 - 3 * c[2] / 14 - c[3] / 4) * 2 * np.pi
    
    # From batmann batman/c_src/_nonlinear_ld.c; to avoid negative in sqrt?
    d = np.minimum(d, 0.99995)

    sqrtmu = (1 - d ** 2) ** 0.25  # d = Rs sinÎ¸, Rs = 1, mu = cos Î¸ -> mu**2 = 1 - d**2
    
    return (1 - c[0] * (1 - sqrtmu)
              - c[1] * (1 - sqrtmu ** 2)
              - c[2] * (1 - sqrtmu ** 3)
              - c[3] * (1 - sqrtmu ** 4)) / norm


c = (0.4, 0.3, 0.2, 0.1)  # I don't know what typcial coefficients are
d = np.cos(np.linspace(0, 1, 101))
ll = intensity(d, c)

plt.figure(figsize=(6, 3))
plt.title('Limb darkening function')
plt.xlabel('$d / R_s$')
plt.ylabel('$I(\mu) / \\mathcal{N}$')
plt.plot(d, ll)
plt.show()

print('Limb darkening coefficients:', c)


# Check total
c = (0.4, 0.3, 0.2, 0.1)
r = np.linspace(0, 1, 1001)
dr = np.diff(r)
r_mid = 0.5 * (r[:-1] + r[1:])
I = intensity(r_mid, c)

A = np.pi * r ** 2
dA = np.diff(A)

total = np.sum(I * dA)
print('Check total:', total)


R = Rs    # radius of the star
r = Rp    # radius of the planet
x = -0.8   # position of the star

fig, ax = plt.subplots()

# The for circle drawing
theta_vals = np.linspace(0, 2 * np.pi, 300)

x_circle = R * np.cos(theta_vals)
y_circle = R * np.sin(theta_vals)
plt.plot(x_circle, y_circle, color='orange')

# Circle r (planet)
x_circle = x + r * np.cos(theta_vals)
y_circle = r * np.sin(theta_vals)
ax.plot(x_circle, y_circle, color='black')

plt.text(0, 1.15, 'star', fontsize=12, color='orange', ha='center', va='center')
plt.text(x, -0.5, 'planet', fontsize=12, color='black', ha='center', va='center')

# Integration circles
x_circle = np.cos(theta_vals)
y_circle = np.sin(theta_vals)
plt.plot(0.65 * x_circle, 0.65 * y_circle, color='lightgray')
plt.plot(0.75 * x_circle, 0.75 * y_circle, color='lightgray')

plt.text(-0.45, 0.3, '$r_{n}$', fontsize=12, color='gray', ha='center', va='center')
plt.text(-0.70, 0.5, '$r_{n+1}$', fontsize=12, color='gray', ha='center', va='center')
plt.text(-0.70, 0.0, '$\\Delta A_{n}$', fontsize=12, color='black', ha='center', va='center')

# Figure range
ax.set_aspect('equal')
ax.set_xlim(-1.5, 1.5)
ax.set_ylim(-1.5, 1.5)
ax.axis('off')

plt.show()

print('Integrate the radius-dependent intensity on star-planet overlap.')


def limb_darkening(d, rp, c, *, nstep=101):
    """
    Compute the intensity from the star
    
    d  (float): Distance between stellar center and planet center in units of stellar radius
    rp (float): Planet radius
    c  (array): Limb darkening coefficients (4, )
    """
    assert isinstance(d, float)
    assert isinstance(rp, float)
    
    # Range of integration
    r_min = max(d - rp, 0)     # lower bound for integration
    r_max = min(d + rp, 1.0)   # upper bound for integration

    if r_min >= 1:
        return 1  # No overlap between star and planet
    elif r_max - r_min < 1e-7:
        return 1  # Overlap is too small

    # This is different from the batman code.
    # See the original code for accurate error control.
    r = np.linspace(r_min, r_max, nstep)
    dr = np.diff(r)
    r_mid = r[:-1] + dr

    rp_array = np.full_like(r, rp)
    d_array = np.full_like(r, d)
    A = area(r, rp_array, d_array)

    dA = np.diff(A)
    I = intensity(r_mid, c)

    assert len(dA) == len(I)

    # \int I(x) dA
    integ = np.sum(I * dA)  # intensity blocked by the planet

    return 1 - integ  # intensity not blocked




def plot_limb_darkening(rp, b, c, *, title=''):
    """
    rp (float): Planet radius in units of stellar radius (Rs=1)
    """
    # = 0.1  # imbact parameter
    s = np.linspace(-2, 2, 201)  # parameter along the planetary orbit (linear)
    d = np.sqrt(b ** 2 + s ** 2) #* np.sign(s)
    R = np.ones_like(d)
    r = np.full_like(d, rp)

    # Area of star-planet overlap
    area_b = area(R, r, d)
    
    ll = []
    for d1 in d:
        l = limb_darkening(d1, rp, c)
        ll.append(l)

    plt.figure(figsize=(6, 2))
    plt.title(title)
    plt.xlabel('$s / R_s$')
    plt.ylabel('Relative flux')
    plt.plot(s, 1 - area_b / math.pi, label='uniform')
    plt.plot(s, ll)
    plt.axhline(1 - rp ** 2, ls=':', color='gray', alpha=0.5)
    plt.show()

b = 0.1
c = (0.4, 0.3, 0.2, 0.1)
plot_limb_darkening(Rp, b, c, title='$R_p/R_s=0.3, b/R_s=0.1$')


# Very large Rp/Rs = 0.3 is for visualization purpose.
# Rp is much smaller in the competition
b = 0.1
c = (0.1, 0.1, 0.1, 0.1)
plot_limb_darkening(0.03, b, c, title='$R_p/R_s=0.03, b/R_s=0.1$')


# No limb darkening for c=0
b = 0.1
c = (0, 0, 0, 0)
plot_limb_darkening(Rp, b, c, title='Check c=0 case')

print('Area calculation agree when no limb darkening c=0.')

