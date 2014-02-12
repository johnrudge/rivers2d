River Inversion 2D
==================

The files in this repository are a set of python scripts which implement the 
river profile to uplift history inversion alogrithm described in Rudge, Roberts,
Richardson and White (2014) [1]. The code uses the non-negative least square (NNLS)
algorithm of Mathieu Blondel (https://gist.github.com/mblondel/4421380) to perform
the inversion.

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

A set of observed river profiles should be provided in the subdirectory river_obs/
with the following format:-

xxx xxx xxx 
xxx xxx xxx


