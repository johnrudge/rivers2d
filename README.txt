River Inversion 2D
==================

The files in this repository are a set of python scripts which implement the 
river profile to uplift history inversion alogrithm described in Rudge, Roberts,
White and Richardson (2015) [1]. The code uses the non-negative least square 
(NNLS) algorithm of Mathieu Blondel (https://gist.github.com/mblondel/4421380) 
to perform the inversion.

Installation
============

In order to run these scripts, you need to have the following installed:

* python 2.7
* numpy (1.7 or later)
* scipy (0.12 or later)
* dolfin 1.3.0 (www.fenicsproject.org)
* scikit-learn (0.13 or later)

For VTK output of river data, you also need the EVTK library
installed (https://bitbucket.org/pauloh/pyevtk).

Usage
=====

The main routine is rivers_2d.py, which can be run with

python rivers_2d.py

which will invert the example data for Madagascar, contained in the 
madagascar_data folder. All obs_river* files have format:

x (lon, m), y (lat, m), z (elevation, m), d (distance, m), A (area, m^2)

References
==========

[1] Rudge J.F., Roberts G.G., White N., Richardson C.N. Uplift histories of 
Africa and Australia from linear inverse modeling of drainage inventories (2015) 
J. Geophys. Res. Earth Surf. 120:1-21