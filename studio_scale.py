import numpy
import matplotlib.pyplot as plt

latlon_ranges = {
    'aus': (
        (-37.777567, 144.951899),
        (-34.055805, 151.151928)
    ),
    'usa': (
        (36.169924, 244.844878),
        (37.799394, 237.537206)
    ),
    'eu': (
        (41.88927, -0.11314),
        (51.50648, 12.60754)
    )
}

s_latlon = {
    'aus': (127.5, 35),
    'usa': (55, 305),
    'eu': (48, 180)
}

"""
Current model:
use min val as offset,
subtract it from other vals to get degree deltas
divide deltas by a magic scale factor.

Want to find an expression for these scale values
"""

# lon investigation:
"""lat_lon_ind = 0  # 0:lat 1:lon
keys = ['aus', 'usa', 'eu']

# scale vs min
x_vals = [(latlon_ranges[k][0][lat_lon_ind] + 180) % 360 - 180 for k in keys]
y_vals = [s_latlon[k][lat_lon_ind] for k in keys]
plt.figure()
plt.scatter(x_vals, y_vals)
plt.title("Scale vs min")
plt.show()"""

"""
Updated model (very accurate):
For lat as example:
Take minimum lat, theta
Shift to [-180, 180], theta_hat
Calculate scale using linear relationship f(theta_hat)

for lat:
    f(theta_hat) = 90 - theta_hat

for lon:
    f(theta_hat) = 180 - theta_hat
"""

# Rotation
# 0 deg pan is north
# 90 deg is east
# Pan vals are simply bearings / 360

# Tilt degs are 0 for downwards, 90 for horizon
# Tilt vals are deg / 180

# Altitude
# 1000, 2000, 10000, 100000 increase linearly
# alt val = alt in m * 0.001535670634989921 / 1e5
# = alt in m * 1.5356706349899208e-08