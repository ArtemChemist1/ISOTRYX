"""Isotopic data from IsoCor (Millard et al., 2019).
Based on IUPAC "Isotopic Compositions of the Elements 2013".
"""

DEFAULT_ISODATA = {
    "C": {
        "abundance": [0.9893, 0.0107],
        "mass": [12.0, 13.003354835]          # nominal masses used for low‑res
    },
    "H": {
        "abundance": [0.999885, 0.000115],
        "mass": [1.0078250322, 2.0141017781]
    },
    "N": {
        "abundance": [0.99636, 0.00364],
        "mass": [14.003074004, 15.000108899]
    },
    "O": {
        "abundance": [0.99757, 0.00038, 0.00205],
        "mass": [15.99491462, 16.999131757, 17.999159613]
    },
    "S": {
        # ³²S, ³³S, ³⁴S, ³⁶S — four stable isotopes (IUPAC 2013).
        # There is no stable ³⁵S; the previous 5-entry table contained a ghost
        # placeholder that produced an incorrect correction matrix for ³⁴S tracers.
        "abundance": [0.9499, 0.0075, 0.0425, 0.0001],
        "mass": [31.972071174, 32.971458910, 33.967866980, 35.967080700]
    },
    "P": {
        "abundance": [1.0],
        "mass": [30.973761998]
    },
    "Si": {
        "abundance": [0.92223, 0.04685, 0.03092],
        "mass": [27.976926535, 28.976494665, 29.9737701]
    },
    "Cl": {
        "abundance": [0.7576, 0.2424],
        "mass": [34.96885269, 36.96590258]
    },
    "F": {
    "abundance": [1.0],
    "mass": [18.998403163]
},
    "Na": {
        "abundance": [1.0],
        "mass": [22.98976928]
    }
}