<h1 align="center" margin=0px>
<img src="https://github.com/cvxgrp/scs/blob/master/docs/src/_static/scs_logo.png" alt="Intersection of a cone and a polyhedron" width="450">
</h1>

[![Build Status](https://github.com/cvxgrp/scs/actions/workflows/build.yml/badge.svg)](https://github.com/cvxgrp/scs/actions/workflows/build.yml)
[![Documentation](https://img.shields.io/badge/docs-online-brightgreen?logo=read-the-docs&style=flat)](https://www.cvxgrp.org/scs/)
[![Coverage Status](https://coveralls.io/repos/github/cvxgrp/scs/badge.svg?branch=master)](https://coveralls.io/github/cvxgrp/scs?branch=master)


SCS (`splitting conic solver`) is a numerical optimization package for solving
large-scale convex cone problems. The current version is `3.2.7`.

The full documentation is available [here](https://www.cvxgrp.org/scs/).

If you wish to cite SCS please cite the papers listed [here](https://www.cvxgrp.org/scs/citing).

Preliminary GPU translation in scs_matrix.cu and linalg.cu was performed using Cuda version 12.8 on an Nvidia 4070 super. In order to get the Cuda 
code to work, must define CULFFLAGS, CUDAFLAGS, and CUDA_PATH in scs.mk. Was not able to replicate results or compile on a seperate pc. Translations currently are naive and involve copying over all data to the gpu device in every single translated function call meaning that performance is very slow. 

Future updates: Need to store scs data on device permenantly so that the cuda functions can access them without having to copy over data with each call. Additionally, cones.c still needs to be translated. This file contains all of the cone functions and is likely to offer the greatest speedup in terms of performance.