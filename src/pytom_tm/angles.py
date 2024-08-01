import pathlib
import os
from typing import Union
from scipy.spatial.transform import Rotation
import numpy as np
import healpix as hp
import logging



def angle_to_angle_list_local(
    angle_diff_inplane: float= 10,
    angle_range_inplane: float = 10.0,
    angle_diff_cone: float= 10,
    angle_range_cone: float = 10.0,
    center_angle: tuple[float, float, float] = (0.0, 0.0, 0.0),
    sort_angles: bool = True,
    log_level: int = logging.DEBUG
    ) -> list[tuple[float, float, float]]:
    """Auto generate an angle list for a given maximum angle difference, focusing on a local region.

    Parameters
    ----------
    angle_diff: float
        maximum difference (in degrees) for the angle list
    center_angles: tuple[float, float, float]
        center of the local angular search (in degrees)
    angle_range: float
        range around the center for the local angular search (in degrees)
    sort_angles: bool, default True
        sort the list, using python default angle_list.sort(), sorts first on Z1, then X, then Z2
    log_level: int, default logging.DEBUG
        the log level to use when generating logs

    Returns
    -------
    angle_list: list[tuple[float, float, float]]
        a list where each element is a tuple of 3 floats containing
        an anti-clockwise ZXZ Euler rotation in radians
    """
   
    phi_angle_step = angle_diff_inplane
    phi_angle_shells = round(angle_range_inplane/angle_diff_inplane)
    psi_angle_step = angle_diff_cone
    psi_angle_shells = round(angle_range_cone/angle_diff_cone)
   
    theta_angle_shells = int(np.ceil(psi_angle_shells / 2))
    count = 0
    angS = []

    for phi_idx in range(-phi_angle_shells, phi_angle_shells + 1):
        for theta_idx in range(0, theta_angle_shells + 1):
            if np.sin(np.radians(theta_idx * psi_angle_step)) == 0:
                psi_delta = 360
                psi_range = [0]
            else:
                psi_delta = psi_angle_step / np.sin(np.radians(theta_idx * psi_angle_step))
                psi_range = range(0, int(np.ceil(360 / psi_delta)))
                psi_delta = 360 / int(np.ceil(360 / psi_delta))
            
            for psi_idx in psi_range:
                psi = psi_idx * psi_delta
                theta = theta_idx * psi_angle_step
                phi0 = np.degrees(np.arctan2(-np.sin(np.radians(psi)), np.cos(np.radians(psi)) * np.cos(np.radians(theta))))
                phi = phi0 + (phi_idx * phi_angle_step)
                count += 1
                angS.append([phi, psi, theta])

# Convert angS to a NumPy array for easier manipulation if needed
    angS = np.array(angS)

    rotMRootAng = compute_rotation_matrix(center_angle[0], center_angle[1], center_angle[2])
    count = 0
    angSR = []
    for angles in angS:
        count += 1
        phi, psi, theta = angles
        
        # Compute the rotation matrix for the angles
        rotMS = compute_rotation_matrix(phi, psi, theta)
        
        # Compute the combined rotation matrix
        rott = np.dot(rotMS, rotMRootAng)
        
        # Compute Euler angles from the rotation matrix
        euler_out = np.zeros(3)
        euler_out[0] = np.arctan2(rott[2, 0], rott[2, 1])
        euler_out[1] = np.arctan2(rott[0, 2], -rott[1, 2])
        euler_out[2] = np.arccos(rott[2, 2])
        
        # Convert Euler angles from radians to degrees
        euler_out = np.rad2deg(euler_out)
        
        if -(rott[2, 2] - 1) < 10e-8:
            euler_out[2] = 0
            euler_out[1] = 0
            euler_out[0] = np.rad2deg(np.arctan2(rott[1, 0], rott[0, 0]))
        
        angSR.append([euler_out[0],euler_out[2],euler_out[1]])
        
    if sort_angles:
        angSR.sort()
          
    return np.array(angSR)
    
     
            
    

def compute_rotation_matrix(phi, psi, theta):
    # Convert angles from degrees to radians
    phi = np.deg2rad(phi)
    psi = np.deg2rad(psi)
    theta = np.deg2rad(theta)
    
    # Compute the rotation matrix
    rotMRootAng = np.dot(np.dot(
        np.array([[np.cos(psi), -np.sin(psi), 0], 
                  [np.sin(psi),  np.cos(psi), 0], 
                  [0,           0,           1]]),
        np.array([[1,           0,            0], 
                  [0, np.cos(theta), -np.sin(theta)], 
                  [0, np.sin(theta),  np.cos(theta)]])),
        np.array([[np.cos(phi), -np.sin(phi), 0], 
                  [np.sin(phi),  np.cos(phi), 0], 
                  [0,           0,           1]]))
    
    return rotMRootAng
    



def angle_to_angle_list(
    angle_diff: float, sort_angles: bool = True, log_level: int = logging.DEBUG
) -> list[tuple[float, float, float]]:
    """Auto generate an angle list for a given maximum angle difference.

    The code uses healpix to determine Z1 and X and splits Z2 linearly.

    Parameters
    ----------
    angle_diff: float
        maximum difference (in degrees) for the angle list
    sort_angles: bool, default True
        sort the list, using python default angle_list.sort(), sorts first on Z1, then X, then Z2
    log_level: int, default logging.DEBUG
        the log level to use when generating logs



    Returns
    -------
    angle_list: list[tuple[float, float, float]]
        a list where each element is a tuple of 3 floats containing
        an anti-clockwise ZXZ Euler rotation in radians
    """
    # We use an approximation of the square root of the area as the median angle diff
    # This works reasonably well and is based on the following formula:
    # angle_diff = (4*np.pi/npix)**0.5 * 360/(2*np.pi)
    npix = 4 * np.pi / (angle_diff * np.pi / 180) ** 2
    nside = 0
    while hp.nside2npix(nside) < npix:
        nside += 1
    used_npix = hp.nside2npix(nside)
    used_angle_diff = (4 * np.pi / used_npix) ** 0.5 * (180 / np.pi)
    logging.log(
        log_level, f"Using an angle difference of {used_angle_diff:.4f} for Z1 and X"
    )
    theta, phi = hp.pix2ang(nside, np.arange(used_npix))
    # Now for psi
    n_psi_angles = int(np.ceil(360 / angle_diff))
    psi, used_psi_diff = np.linspace(
        0, 2 * np.pi, n_psi_angles, endpoint=False, retstep=True
    )
    logging.log(
        log_level,
        f"Using an angle difference of {np.rad2deg(used_psi_diff):.4f} for Z2",
    )
    angle_list = [(ph, th, ps) for ph, th in zip(phi, theta) for ps in psi]
    if sort_angles:
        angle_list.sort()
    return angle_list


def load_angle_list(
    file_name: pathlib.Path, sort_angles: bool = True
) -> list[tuple[float, float, float]]:
    """Load an angular search list from disk.

    Parameters
    ----------
    file_name: pathlib.Path
        path to text file containing angular search, each line should contain 3 floats of anti-clockwise ZXZ
    sort_angles: bool, default True
        sort the list, using python default angle_list.sort(), sorts first on Z1, then X, then Z2

    Returns
    -------
    angle_list: list[tuple[float, float, float]]
        a list where each element is a tuple of 3 floats containing an anti-clockwise ZXZ Euler rotation in radians
    """
    with open(str(file_name)) as fstream:
        lines = fstream.readlines()
    angle_list = [tuple(map(float, x.strip().split(" "))) for x in lines]
    if not all([len(a) == 3 for a in angle_list]):
        raise ValueError(
            "Invalid angle file provided, each line should have 3 ZXZ Euler angles!"
        )
    if sort_angles:
        angle_list.sort()  # angle list needs to be sorted otherwise symmetry reduction cannot be used!
    return angle_list


def get_angle_list(
    angle: Union[pathlib.Path, float],
    sort_angles: bool = True,
    symmetry: int = 1,
    log_level: str = "DEBUG",
):
    """Either get an angular search file from disk or generate one from a float

    Parameters
    ----------
    angle: Union[pathlib.Path, float]
        either the path to text file containing angular search,
          each line should contain 3 floats of anti-clockwise ZXZ
        or if a float:
          maximum difference (in degrees) for the angle list
    sort_angles: bool, default True
        sort the list, using python default angle_list.sort(), sorts first on Z1, then X, then Z2
    symmetry: int, default 1
        the returned list will only have Z2 angles [0, (2*pi/symmetry))
    log_level: str, default 'DEBUG'
        the log level to use when generating logs

    Returns
    -------
    angle_list: list[tuple[float, float, float]]
        a list where each element is a tuple of 3 floats containing an anti-clockwise ZXZ Euler rotation in radians
    """
    log_level = logging.getLevelNamesMapping()[log_level]
    out = None
    max_z2 = 2 * np.pi / symmetry
    try:
        angle = float(angle)
        angle_is_float = True
    except (ValueError, TypeError):
        angle_is_float = False
    if angle_is_float:
        logging.log(
            log_level,
            f"Will generate an angle list with a maximum increment of {angle}",
        )
        out = angle_to_angle_list(angle, sort_angles, log_level)
    elif isinstance(angle, (str, os.PathLike)):
        possible_file_path = pathlib.Path(angle)
        if possible_file_path.exists() and possible_file_path.suffix == ".txt":
            logging.log(
                log_level,
                "Custom file provided for the angular search. Checking if it can be read...",
            )
            out = load_angle_list(angle, sort_angles)

    if out is None:  # If no angles by now, error out
        raise ValueError("Invalid angle input provided")
    return [i for i in out if i[2] < max_z2]


def convert_euler(
    angles: tuple[float, float, float],
    order_in: str = "ZXZ",
    order_out: str = "ZXZ",
    degrees_in: bool = True,
    degrees_out: bool = True,
) -> tuple[float, float, float]:
    """Convert a single set of Euler angles from one Euler notation to another. This function makes use of
    scipy.spatial.transform.Rotation meaning that capital letters (i.e. ZXZ) specify intrinsic rotations (commonly
    used in cryo-EM) and small letters (i.e. zxz) specific extrinsic rotations.

    Parameters
    ----------
    angles: tuple[float, float, float]
        tuple of three angles
    order_in: str, default 'ZXZ'
        Euler rotation axis of input angles
    order_out: str, default 'ZXZ'
        Euler rotation axis of output angles
    degrees_in: bool, default True
        whether the input angles are in degrees
    degrees_out: bool, default True
        whether the output angles should be in degrees

    Returns
    -------
    output: tuple[float, float, float]
        tuple of three angles
    """
    r = Rotation.from_euler(order_in, angles, degrees=degrees_in)
    return tuple(r.as_euler(order_out, degrees=degrees_out))
