"""
Note, using lonlat representation for coordinates
"""
import argparse
import math
import numpy as np
import json

EARTH_RADIUS_M = 6.371e6
EARTH_CIRC_M = 2 * math.pi * EARTH_RADIUS_M
TEMPLATE_FILE = "kml_to_esp_template.esp"

def moving_average(a, n=3) :
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n


def angle180_180(a):
    """ shifts to [-180, 180] """
    return (a + 180) % 360 - 180


def get_block(label, vals, min_val=None, max_val=None):
    """
        linear_block = {  # UNUSED
            "transitionIn": {
            "x": 0,
            "y": 0,
            "influence": 0,
            "type": "linear"
        },
        "transitionOut": {
            "x": 0,
            "y": 0,
            "influence": 0,
            "type": "linear"
        },
        "transitionLinked": False
        }
    """
    n_steps = len(vals)

    ret = {
        "type": label,
        "value": {
            "relative": 0
        },
        "keyframes": [
            {
                "time": i / (n_steps - 1),
                "value": val,
            }
            for i, val in enumerate(vals)
        ],
        "intimeline": True
    }

    if min_val is not None:
        ret['value']["minValueRange"] = min_val
    if max_val is not None:
        ret['value']["maxValueRange"] = max_val
    if label == 'altitude':
        ret['value']["logarithmic"] = False

    return ret


def alts_to_vals(a):
    """
    Altitude
        1000, 2000, 10000, 100000 increase linearly
        alt val = alt in m * 0.001535670634989921 / 1e5
        = alt in m * 1.5356706349899208e-08
    """
    return a * 1.5356706349899208e-08


def get_deltas(coords, backtrace):
    """
    Returns (lon, lat) that steps backwards from each given coord in the opposite direction than the next
    Steps backwards the distance specified by backtrace (along the ground)
    """
    rads = get_rads_from_coords(coords)
    ret = []

    # m
    dx = -backtrace * np.cos(rads)
    dy = -backtrace * np.sin(rads)

    start_lats = coords[: len(rads), 1]
    dlat = dy / EARTH_CIRC_M * 360
    dlon = dx / (EARTH_CIRC_M * np.cos(np.radians(start_lats))) * 360

    return np.concatenate((dlon.reshape((-1, 1)), dlat.reshape((-1, 1))), axis=1)


def dx_dy_from_coords(c1, c2):
    """
    Returns horizontal and vertical distance in metres from c1 to c2
    c1, c2 should be (lon, lat) in degrees
    """
    dlat = c2[1] - c1[1]
    dlon = c2[0] - c1[0]
    curr_lat_rad = math.radians(c1[1])

    dy = EARTH_CIRC_M * dlat / 360
    dx = EARTH_CIRC_M * (dlon / 360) * math.cos(curr_lat_rad)
    return dx, dy


def coord_dist_m(c1, c2):
    return np.linalg.norm(dx_dy_from_coords(c1, c2))


def coord_mean(coords):
    return np.array(coords).mean(axis=0)


def get_angle_rad(c1, c2):
    dx, dy = dx_dy_from_coords(c1, c2)
    return math.atan2(dy, dx)


def get_angle_bearing(c1, c2):
    angle_rad = get_angle_rad(c1, c2)
    return (90 - math.degrees(angle_rad)) % 360


def pans_to_vals(p):
    """
    Rotation
        0 deg pan is north
        90 deg is east
        Pan vals are simply bearings / 360
    DEPRECATED
    """
    assert False, "deprecated"
    return p / 360


def get_rads_from_coords(coords):
    return np.array([get_angle_rad(coords[i], coords[i+1]) for i in range(len(coords) - 1)])


def get_pans_from_coords(coords):
    """
    Returns a set of camera pans that face from current point to next,
    using [-360]
    """
    ret = []
    prev_b = 0
    for i in range(len(coords) - 1):
        b = get_angle_bearing(coords[i], coords[i+1])  # in [0, 360]

        if abs(b - 360 - prev_b) < abs(b - prev_b):
            b -= 360
        elif abs(b + 360 - prev_b) < abs(b - prev_b):
            b += 360
        prev_b = b
        ret.append(b)

    return np.array(ret)


def tilts_to_vals(t):
    """
    Tilt degs are 0 for downwards, 90 for horizon
    Tilt vals are deg / 180
    """
    return t / 180


def coords_from_kml(contents, n=None):
    """
    Returns lonlats
    """
    marker1 = "<coordinates>"
    marker2 = "</coordinates>"
    pos1 = contents.index(marker1)
    pos2 = contents.index(marker2)
    coords_str = contents[pos1 + len(marker1) : pos2]

    coords = [tuple(map(float, c.strip().split(',')[:2])) for c in coords_str.strip().split('\n')]
    if n is not None:
        coords = coords[::len(coords) // n]
    coords = np.array(coords)
    return coords


def main(input_kml, out_file, alt_m=1000, n_steps=None, tilt_deg=30, backtrace=0, moving_agv=3, noise_dist_m=50):
    """
    Tilt: 0 is downwards, 90 is horizontal
    """

    with open(input_kml, 'r') as infile:
        contents = infile.read()
    coords = coords_from_kml(contents, n=n_steps)

    if noise_dist_m != 0:
        assert noise_dist_m > 0, "noise_dist_m must be > 0"
        prev_anchor = None
        current_cluster = None
        coords_list = []
        for coord in coords:
            if prev_anchor is None:
                prev_anchor = coord
                current_cluster = [coord]

            if coord_dist_m(coord, prev_anchor) < noise_dist_m:
                current_cluster.append(coord)
            else:
                if prev_anchor is not None:
                    coords_list.append(coord_mean(current_cluster))

                current_cluster = [coord]
                prev_anchor = coord

        coords_list.append(coord_mean(current_cluster))
        coords = np.concatenate([c.reshape((1, 2)) for c in coords_list], axis=0)

    if moving_agv != 0:
        assert int(moving_agv) == moving_agv and moving_agv > 0, "moving_agv must be int > 0"
        lon = moving_average(coords[:, 0], int(moving_agv)).reshape((-1, 1))
        lat = moving_average(coords[:, 1], int(moving_agv)).reshape((-1, 1))
        coords = np.concatenate((lon, lat), axis=1)
    n_steps = n_steps or len(coords)

    with open(TEMPLATE_FILE, 'r') as infile:
        template = infile.read()

    if backtrace is None:
        backtrace = math.tan(math.radians(tilt_deg)) * alt_m  # Situate the camera further back than the kml coord

    coord_delta_lonlats = get_deltas(coords, backtrace)
    n_deltas = len(coord_delta_lonlats)
    coords = coords[:n_deltas] + coord_delta_lonlats

    lon_vals = coords[:, 0]
    lat_vals = coords[:, 1]
    lon_min = lon_vals.min()
    lat_min = lat_vals.min()

    # scaling found empirically
    s_lat = 90 - angle180_180(lat_min)
    s_lon = 180 - angle180_180(lon_min)
    lon_vals = (lon_vals - lon_min) / s_lon
    lat_vals = (lat_vals - lat_min) / s_lat

    pan_vals = get_pans_from_coords(coords)
    pan_min = pan_vals.min()
    pan_max = pan_vals.max()
    pan_range = pan_max - pan_min
    pan_vals = (pan_vals - pan_min) / pan_range

    rot_args = [
        ('rotationX', pan_vals, pan_min, pan_max),
        ('rotationY', tilts_to_vals(np.array([tilt_deg] * 2)))  # one tilt for whole vid
    ]

    pos_args = [
        ('longitude', lon_vals, lon_min),
        ('latitude', lat_vals, lat_min),
        ('altitude', alts_to_vals(alt_m * np.ones_like(lat_vals)))
    ]

    pos_jstr = json.dumps([get_block(*arg_set) for arg_set in pos_args])
    rot_jstr = json.dumps(
        [get_block(*arg_set) for arg_set in rot_args] +
        [{"type": "rotationZ","value": {}}]  # unused
    )
    out_str = template % (pos_jstr, rot_jstr)

    with open(out_file, 'w') as outfile:
        outfile.write(out_str)
    print("Wrote to", out_file)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("kml", help="Input kml filename")
    parser.add_argument("out", help="Output esp filename", default="out.esp")
    parser.add_argument("--n_steps", type=int, help="Number of uniformly-spaced keyframes to add", default=200)
    parser.add_argument("--tilt", type=float, help="Angle (in degrees) the camera must be tilted in each keyframe", default=45)
    parser.add_argument("--alt", type=float, help="Altitude (in metres) for the camera", default=500)
    parser.add_argument("--backtrace", type=float, help="Distance in metres to shift the camera behind each kml coord", default=50)
    args = parser.parse_args()

    main(input_kml=args.kml, out_file=args.out, alt_m=args.alt, n_steps=args.n_steps, tilt_deg=args.tilt, backtrace=args.backtrace, moving_agv=3, noise_dist_m=50)
