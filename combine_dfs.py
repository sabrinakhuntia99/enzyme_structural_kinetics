import pandas as pd
import numpy as np
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.preprocessing import StandardScaler

# 1. Load and clean the data
input_file = r"C:\Users\Sabrina\Documents\GitHub\protein_structural_kinetics\data\combined_data.tsv"


def load_and_filter_data(filepath):
    # Read data with explicit NA handling
    df = pd.read_csv(filepath, sep='\t', na_values=['', 'NA', 'NaN', 'None', 'nan', 'null'])

    # Convert numeric columns, coercing errors to NA
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in df.columns:
        if col not in numeric_cols and col != 'uniprot_id':
            df[col] = pd.to_numeric(df[col], errors='coerce')

    # Remove rows where hull_presence is 0 or NA
    filtered_df = df[(df['hull_presence'] > 0) & (df['hull_presence'].notna())].copy()
    print(f"Original rows: {len(df)}")
    print(f"Rows after filtering: {len(filtered_df)}")
    print(f"Removed {len(df) - len(filtered_df)} rows with hull_presence ≤ 0 or NA")

    return filtered_df


# 2. Calculate all pairwise correlations
def calculate_all_correlations(df):
    # Select only numeric columns (excluding uniprot_id)
    numeric_df = df.select_dtypes(include=[np.number])
    variables = numeric_df.columns
    n_vars = len(variables)

    # Initialize result matrices
    results = {
        'pearson': pd.DataFrame(np.nan, index=variables, columns=variables),
        'spearman': pd.DataFrame(np.nan, index=variables, columns=variables),
        'kernel': pd.DataFrame(np.nan, index=variables, columns=variables)
    }

    # Standardize data for kernel
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(numeric_df)

    # Calculate all pairwise correlations
    for i, var1 in enumerate(variables):
        for j, var2 in enumerate(variables):
            if i < j:  # Only compute upper triangle
                # Get valid pairs (non-NA)
                valid_data = numeric_df[[var1, var2]].dropna()

                if len(valid_data) > 1:
                    x = valid_data[var1]
                    y = valid_data[var2]

                    # Pearson
                    try:
                        results['pearson'].loc[var1, var2] = results['pearson'].loc[var2, var1] = pearsonr(x, y)[0]
                    except:
                        pass

                    # Spearman
                    try:
                        results['spearman'].loc[var1, var2] = results['spearman'].loc[var2, var1] = spearmanr(x, y)[0]
                    except:
                        pass

                    # Kernel (RBF)
                    try:
                        results['kernel'].loc[var1, var2] = results['kernel'].loc[var2, var1] = rbf_kernel(
                            scaled_data[valid_data.index, i:i + 1],
                            scaled_data[valid_data.index, j:j + 1]
                        ).mean()
                    except:
                        pass

    return results


# 3. Print and save results
def save_and_display_results(correlations, output_prefix):
    for method, corr_df in correlations.items():
        # Fill diagonal with 1s
        np.fill_diagonal(corr_df.values, 1.0)

        # Print top 10 strongest correlations for each method
        print(f"\n=== Top 10 {method.upper()} correlations ===")
        corr_stack = corr_df.stack()
        corr_stack = corr_stack[corr_stack.index.get_level_values(0) < corr_stack.index.get_level_values(1)]
        top_corrs = corr_stack.abs().sort_values(ascending=False).head(10)

        print(f"{'Variable 1':<30}{'Variable 2':<30}{'Correlation':>15}")
        print("-" * 75)
        for (var1, var2), value in top_corrs.items():
            print(f"{var1:<30}{var2:<30}{value:>15.3f}")

        # Save full matrix
        output_file = f"{output_prefix}_{method}.tsv"
        corr_df.to_csv(output_file, sep='\t', float_format="%.3f")
        print(f"\nFull matrix saved to: {output_file}")


# 4. Main execution
if __name__ == "__main__":
    output_prefix = r"C:\Users\Sabrina\Documents\GitHub\protein_structural_kinetics\data\correlation_results"

    print("Loading and filtering data...")
    try:
        df = load_and_filter_data(input_file)

        print("\nCalculating correlations...")
        correlations = calculate_all_correlations(df)

        print("\n=== Analysis Results ===")
        save_and_display_results(correlations, output_prefix)
        print("\nAnalysis complete!")
    except Exception as e:
        print(f"\nError encountered: {str(e)}")
        print("Please check your input file format and data types.")