# Systematic Validation of Predicted Protein Structures

## 1. Gather limited proteolysis data.

## 2. Quantify enzymatic cleavage pattern at the proteome level.
### A. Pre-process data.
### B. Apply linear regression encompassing all time points for each protein.
### C. Evaluate goodness of fit.
### D. Determine slope.

## 3. Calculate potential metrics of fragmentation using the Cartesian coordinates of AlphaFold-predicted molecular structures.
### A. sequence length
total number of amino acids in the protein
### B. density
compactness
### C. radius of gyration
the radial distance to a point which would have a moment of inertia the same as the body's actual distribution of mass, if the total mass of the body were concentrated there
### D. sphericity
how closely the shape of an object resembles that of a perfect sphere
### E. SA:V ratio
ratio of surface area to volume
### F. Euler characteristic
number that describes a topological space's shape or structure regardless of the way it is bent
### G. inradius*
### H. circumradius*
### I. hydrodynamic radius
translational hydrodynamic radius calculated from a sphere with the equivalent volume as the expanded convex hull for a protein and corrected by the translational shape factor
### J. error bounds for lagrange interpolation polynomial
difference between a function and its interpolating polynomial based on the function's higher-order derivatives and the distance from interpolation points
### K. image classification (cnn)
category of image by learning hierarchical features through convolutional layers (convolutional neural network)
### TBD...

## 4. Develop machine learning model to estimate disorder content of sample proteins.
### A. For initial training, use DisProt database (experimentally determined) and add columns with calculated metrics (listed above).
### B. Apply ML model to lab experimental data.

## 5. Determine correlation between structural measures, disorder content, and cleavage pattern.
