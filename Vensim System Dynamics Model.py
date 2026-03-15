"""
Python model 'Vensim System Dynamics Model.py'
Translated using PySD
"""

from pathlib import Path
import numpy as np

from pysd.py_backend.statefuls import Integ, Smooth
from pysd import Component

__pysd_version__ = "3.14.3"

__data = {"scope": None, "time": lambda: 0}

_root = Path(__file__).parent


component = Component()

#######################################################################
#                          CONTROL VARIABLES                          #
#######################################################################

_control_vars = {
    "initial_time": lambda: 0,
    "final_time": lambda: 119,
    "time_step": lambda: 1,
    "saveper": lambda: time_step(),
}


def _init_outer_references(data):
    for key in data:
        __data[key] = data[key]


@component.add(name="Time")
def time():
    """
    Current time of the model.
    """
    return __data["time"]()


@component.add(
    name="FINAL TIME", units="Month", comp_type="Constant", comp_subtype="Normal"
)
def final_time():
    """
    The final time for the simulation.
    """
    return __data["time"].final_time()


@component.add(
    name="INITIAL TIME", units="Month", comp_type="Constant", comp_subtype="Normal"
)
def initial_time():
    """
    The initial time for the simulation.
    """
    return __data["time"].initial_time()


@component.add(
    name="SAVEPER",
    units="Month",
    limits=(0.0, np.nan),
    comp_type="Auxiliary",
    comp_subtype="Normal",
    depends_on={"time_step": 1},
)
def saveper():
    """
    The frequency with which output is stored.
    """
    return __data["time"].saveper()


@component.add(
    name="TIME STEP",
    units="Month",
    limits=(0.0, np.nan),
    comp_type="Constant",
    comp_subtype="Normal",
)
def time_step():
    """
    The time step for the simulation.
    """
    return __data["time"].time_step()


#######################################################################
#                           MODEL VARIABLES                           #
#######################################################################


@component.add(name="d0", comp_type="Constant", comp_subtype="Normal")
def d0():
    return 0.08


@component.add(name="d1", comp_type="Constant", comp_subtype="Normal")
def d1():
    return 0.35


@component.add(
    name="Degradation",
    comp_type="Auxiliary",
    comp_subtype="Normal",
    depends_on={"vegetation_biomass": 1, "d1": 1, "soil_water": 1, "d0": 1},
)
def degradation():
    return vegetation_biomass() * (d0() + d1() * (1 - soil_water()))


@component.add(
    name="Evapotranspiration",
    comp_type="Auxiliary",
    comp_subtype="Normal",
    depends_on={"soil_water": 1, "vegetation_biomass": 1, "k_et": 1},
)
def evapotranspiration():
    return soil_water() * vegetation_biomass() * k_et()


@component.add(name="g0", comp_type="Constant", comp_subtype="Normal")
def g0():
    return 0.01


@component.add(
    name="Growth",
    comp_type="Auxiliary",
    comp_subtype="Normal",
    depends_on={"r": 1, "vegetation_biomass": 2, "soil_water": 4, "g0": 1},
)
def growth():
    return r() * vegetation_biomass() * (1 - vegetation_biomass()) * (
        soil_water() / (soil_water() + 0.5)
    ) + g0() * (soil_water() / (soil_water() + 0.5))


@component.add(name="infil c", comp_type="Constant", comp_subtype="Normal")
def infil_c():
    return 0.6


@component.add(
    name="Infiltration",
    units="mm/Month",
    comp_type="Auxiliary",
    comp_subtype="Normal",
    depends_on={"infil_c": 1, "precipitation_norm": 1, "soil_water": 1},
)
def infiltration():
    return infil_c() * precipitation_norm() * (1 - soil_water())


@component.add(name="k et", comp_type="Constant", comp_subtype="Normal")
def k_et():
    return 0.05


@component.add(name="k perc", comp_type="Constant", comp_subtype="Normal")
def k_perc():
    return 0.15


@component.add(name="r", comp_type="Constant", comp_subtype="Normal")
def r():
    return 0.5


@component.add(
    name="Percolation",
    comp_type="Auxiliary",
    comp_subtype="Normal",
    depends_on={"soil_water": 1, "k_perc": 1},
)
def percolation():
    return soil_water() * k_perc()


@component.add(
    name="NDVI sim raw",
    comp_type="Auxiliary",
    comp_subtype="Normal",
    depends_on={"vegetation_biomass": 1, "precipitation_norm": 1},
)
def ndvi_sim_raw():
    return 0.1 * vegetation_biomass() + 0.16 * precipitation_norm() + 0.09


@component.add(
    name="NDVI sim",
    comp_type="Stateful",
    comp_subtype="Smooth",
    depends_on={"_smooth_ndvi_sim": 1},
    other_deps={
        "_smooth_ndvi_sim": {
            "initial": {"ndvi_sim_raw": 1},
            "step": {"ndvi_sim_raw": 1},
        }
    },
)
def ndvi_sim():
    return float(np.maximum(0, float(np.minimum(1, _smooth_ndvi_sim()))))


_smooth_ndvi_sim = Smooth(
    lambda: ndvi_sim_raw(),
    lambda: 1,
    lambda: ndvi_sim_raw(),
    lambda: 1,
    "_smooth_ndvi_sim",
)


@component.add(
    name="NDVI obs",
    comp_type="Auxiliary",
    comp_subtype="with Lookup",
    depends_on={"time": 1},
)
def ndvi_obs():
    return np.interp(
        time(),
        [
            0.0,
            1.0,
            2.0,
            3.0,
            4.0,
            5.0,
            6.0,
            7.0,
            8.0,
            9.0,
            10.0,
            11.0,
            12.0,
            13.0,
            14.0,
            15.0,
            16.0,
            17.0,
            18.0,
            19.0,
            20.0,
            21.0,
            22.0,
            23.0,
            24.0,
            25.0,
            26.0,
            27.0,
            28.0,
            29.0,
            30.0,
            31.0,
            32.0,
            33.0,
            34.0,
            35.0,
            36.0,
            37.0,
            38.0,
            39.0,
            40.0,
            41.0,
            42.0,
            43.0,
            44.0,
            45.0,
            46.0,
            47.0,
            48.0,
            49.0,
            50.0,
            51.0,
            52.0,
            53.0,
            54.0,
            55.0,
            56.0,
            57.0,
            58.0,
            59.0,
            60.0,
            61.0,
            62.0,
            63.0,
            64.0,
            65.0,
            66.0,
            67.0,
            68.0,
            69.0,
            70.0,
            71.0,
            72.0,
            73.0,
            74.0,
            75.0,
            76.0,
            77.0,
            78.0,
            79.0,
            80.0,
            81.0,
            82.0,
            83.0,
            84.0,
            85.0,
            86.0,
            87.0,
            88.0,
            89.0,
            90.0,
            91.0,
            92.0,
            93.0,
            94.0,
            95.0,
            96.0,
            97.0,
            98.0,
            99.0,
            100.0,
            101.0,
            102.0,
            103.0,
            104.0,
            105.0,
            106.0,
            107.0,
            108.0,
            109.0,
            110.0,
            111.0,
            112.0,
            113.0,
            114.0,
            115.0,
            116.0,
            117.0,
            118.0,
            119.0,
        ],
        [
            0.093,
            0.089,
            0.088,
            0.092,
            0.108,
            0.138,
            0.168,
            0.196,
            0.149,
            0.112,
            0.103,
            0.094,
            0.091,
            0.093,
            0.093,
            0.098,
            0.114,
            0.141,
            0.156,
            0.168,
            0.147,
            0.11,
            0.092,
            0.084,
            0.087,
            0.091,
            0.088,
            0.099,
            0.13,
            0.156,
            0.19,
            0.191,
            0.152,
            0.123,
            0.103,
            0.062,
            0.071,
            0.095,
            0.091,
            0.099,
            0.141,
            0.182,
            0.205,
            0.205,
            0.167,
            0.133,
            0.101,
            0.091,
            0.084,
            0.089,
            0.088,
            0.107,
            0.14,
            0.179,
            0.208,
            0.212,
            0.163,
            0.12,
            0.106,
            0.062,
            0.067,
            0.053,
            0.089,
            0.101,
            0.138,
            0.171,
            0.192,
            0.217,
            0.179,
            0.136,
            0.112,
            0.098,
            0.02,
            0.089,
            0.092,
            0.1,
            0.137,
            0.162,
            0.207,
            0.234,
            0.182,
            0.139,
            0.115,
            0.096,
            0.1,
            0.093,
            0.095,
            0.108,
            0.146,
            0.195,
            0.219,
            0.225,
            0.171,
            0.134,
            0.11,
            0.097,
            0.046,
            0.043,
            0.097,
            0.105,
            0.15,
            0.179,
            0.215,
            0.235,
            0.184,
            0.123,
            0.102,
            0.088,
            0.084,
            0.085,
            0.089,
            0.107,
            0.157,
            0.183,
            0.218,
            0.243,
            0.185,
            0.133,
            0.072,
            0.097,
        ],
    )


@component.add(
    name="Precipitation norm",
    comp_type="Auxiliary",
    comp_subtype="Normal",
    depends_on={"precipitation": 1},
)
def precipitation_norm():
    return float(
        np.maximum(
            0,
            float(np.minimum(1, (precipitation() - 3.123) / (1240.24 - 3.123 + 1e-06))),
        )
    )


@component.add(
    name="precipitation",
    units="mm/Month",
    comp_type="Auxiliary",
    comp_subtype="with Lookup",
    depends_on={"time": 1},
)
def precipitation():
    return np.interp(
        time(),
        [
            0.0,
            1.0,
            2.0,
            3.0,
            4.0,
            5.0,
            6.0,
            7.0,
            8.0,
            9.0,
            10.0,
            11.0,
            12.0,
            13.0,
            14.0,
            15.0,
            16.0,
            17.0,
            18.0,
            19.0,
            20.0,
            21.0,
            22.0,
            23.0,
            24.0,
            25.0,
            26.0,
            27.0,
            28.0,
            29.0,
            30.0,
            31.0,
            32.0,
            33.0,
            34.0,
            35.0,
            36.0,
            37.0,
            38.0,
            39.0,
            40.0,
            41.0,
            42.0,
            43.0,
            44.0,
            45.0,
            46.0,
            47.0,
            48.0,
            49.0,
            50.0,
            51.0,
            52.0,
            53.0,
            54.0,
            55.0,
            56.0,
            57.0,
            58.0,
            59.0,
            60.0,
            61.0,
            62.0,
            63.0,
            64.0,
            65.0,
            66.0,
            67.0,
            68.0,
            69.0,
            70.0,
            71.0,
            72.0,
            73.0,
            74.0,
            75.0,
            76.0,
            77.0,
            78.0,
            79.0,
            80.0,
            81.0,
            82.0,
            83.0,
            84.0,
            85.0,
            86.0,
            87.0,
            88.0,
            89.0,
            90.0,
            91.0,
            92.0,
            93.0,
            94.0,
            95.0,
            96.0,
            97.0,
            98.0,
            99.0,
            100.0,
            101.0,
            102.0,
            103.0,
            104.0,
            105.0,
            106.0,
            107.0,
            108.0,
            109.0,
            110.0,
            111.0,
            112.0,
            113.0,
            114.0,
            115.0,
            116.0,
            117.0,
            118.0,
            119.0,
        ],
        [
            105.363,
            16.37,
            30.06,
            33.563,
            81.777,
            424.195,
            415.561,
            638.09,
            389.627,
            145.209,
            36.247,
            6.707,
            53.681,
            27.656,
            20.026,
            297.604,
            60.057,
            149.68,
            693.717,
            1240.24,
            964.497,
            274.127,
            70.089,
            73.794,
            12.124,
            20.337,
            81.708,
            306.744,
            574.445,
            838.447,
            705.575,
            348.636,
            686.369,
            73.584,
            8.548,
            124.961,
            21.486,
            35.519,
            130.537,
            333.263,
            528.53,
            472.574,
            771.325,
            823.763,
            882.742,
            197.355,
            151.944,
            7.47,
            45.055,
            15.029,
            26.528,
            28.869,
            384.339,
            493.969,
            654.074,
            1198.39,
            521.075,
            133.04,
            18.822,
            89.193,
            9.832,
            35.621,
            23.244,
            55.695,
            388.866,
            205.592,
            418.337,
            538.169,
            388.632,
            88.195,
            13.766,
            37.314,
            127.255,
            32.619,
            1.574,
            43.461,
            473.203,
            177.047,
            939.261,
            834.344,
            369.147,
            82.954,
            122.563,
            13.523,
            14.081,
            89.734,
            284.109,
            103.64,
            327.657,
            496.714,
            554.073,
            664.877,
            455.592,
            436.644,
            3.123,
            46.1,
            125.355,
            24.411,
            90.119,
            166.606,
            93.398,
            398.463,
            679.636,
            1230.01,
            1012.49,
            116.763,
            45.165,
            7.458,
            7.641,
            15.367,
            85.607,
            71.796,
            386.572,
            27.771,
            719.825,
            1094.83,
            545.976,
            118.44,
            195.001,
            12.974,
        ],
    )


@component.add(
    name="Soil Water",
    comp_type="Stateful",
    comp_subtype="Integ",
    depends_on={"_integ_soil_water": 1},
    other_deps={
        "_integ_soil_water": {
            "initial": {},
            "step": {"infiltration": 1, "evapotranspiration": 1, "percolation": 1},
        }
    },
)
def soil_water():
    return _integ_soil_water()


_integ_soil_water = Integ(
    lambda: infiltration() - evapotranspiration() - percolation(),
    lambda: 0.3,
    "_integ_soil_water",
)


@component.add(
    name="Vegetation Biomass",
    comp_type="Stateful",
    comp_subtype="Integ",
    depends_on={"_integ_vegetation_biomass": 1},
    other_deps={
        "_integ_vegetation_biomass": {
            "initial": {},
            "step": {"growth": 1, "degradation": 1},
        }
    },
)
def vegetation_biomass():
    return _integ_vegetation_biomass()


_integ_vegetation_biomass = Integ(
    lambda: growth() - degradation(), lambda: 0.08, "_integ_vegetation_biomass"
)
