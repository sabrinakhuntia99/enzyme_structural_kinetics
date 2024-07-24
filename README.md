# Systematic Validation of Predicted Protein Structures

## Gather limited proteolysis data.

## Quantify enzymatic cleavage pattern at the proteome level.
### 1. Pre-process data.
### 2. Apply linear regression.
### 3. Evaluate goodness of fit.
### 4. Determine slope.

## Calculate potential metrics of fragmentation using the Cartesian coordinates of AlphaFold-predicted molecular structures.
### 1. sequence length
### 2. density
### 3. radius of gyration
### 4. sphericity
### 5. surface area-to-volume ratio
### 6. Euler characteristic
### 7. inradius
### 8. circumradius
### 9. hydrodynamic radius
### 10. lagrange approximation error
### 11. image classification (cnn)
### TBD...

## Develop machine learning model to estimate disorder content of sample proteins.
### 1. For initial training, use DisProt database (experimentally determined) and add columns with calculated metrics (listed above).
### 2. Apply ML model to your own experimental data.

## Determine correlation between structural measures, disorder content, and cleavage pattern.
