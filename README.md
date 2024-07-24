# Systematic Validation of Predicted Protein Structures

## Gather limited proteolysis data.

## 1. Quantify enzymatic cleavage pattern at the proteome level.
### A. Pre-process data.
### B. Apply linear regression encompassing all time points for each protein.
### C. Evaluate goodness of fit.
### D. Determine slope.

## 2. Calculate potential metrics of fragmentation using the Cartesian coordinates of AlphaFold-predicted molecular structures.
### A. sequence length
### B. density
### C. radius of gyration
### D. sphericity
### E. surface area-to-volume ratio
### F. Euler characteristic
### G. inradius
### H. circumradius
### I. hydrodynamic radius
### J. lagrange approximation error
### K. image classification (cnn)
### TBD...

## 3. Develop machine learning model to estimate disorder content of sample proteins.
### A. For initial training, use DisProt database (experimentally determined) and add columns with calculated metrics (listed above).
### B. Apply ML model to lab experimental data.

## 4. Determine correlation between structural measures, disorder content, and cleavage pattern.
