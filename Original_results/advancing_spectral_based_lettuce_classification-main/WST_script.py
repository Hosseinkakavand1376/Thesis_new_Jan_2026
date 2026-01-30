#!/usr/bin/env python
"""
This script runs the spectral data processing pipeline for all the following combinations:
  For N_PLS_COMPONENTS in [2, 3] and for the following processing pairs:
    - ("SVN", "SG")
    - ("MSC", "SG")
    - ("SG", "SVN")
    - ("SG", "MSC")

For each combination, the pipeline:
  - Loads and pivots the spectral data,
  - Applies the two-step preprocessing (first_op then second_op),
  - Performs PCA‐based outlier detection,
  - Saves three diagnostic plots (outlier detection, PCA scores with ellipse, and cleaned spectra)
    into a folder named "imgs",
  - Runs the CARS model.

Adjust file paths or parameters as needed.
"""

import sys, os
import time
sys.path.insert(0, "/Users/nicoladilillo/Projects_mac/lettuce_spectral_signature/")

import pandas as pd
import numpy as np
from WST import WST
from scipy.signal import savgol_filter
from sklearn.decomposition import PCA
from scipy.stats import chi2
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.lines import Line2D

# ------------------------------
# Define Processing Functions
# ------------------------------
def msc_normalization(spectra):
    """Apply Multiplicative Scatter Correction (MSC) to each spectrum."""
    reference = np.mean(spectra, axis=0)
    def correct_spectrum(spectrum):
        slope, intercept = np.polyfit(reference, spectrum, 1)
        return (spectrum - intercept) / slope
    corrected = spectra.apply(lambda row: correct_spectrum(row.values), axis=1)
    return pd.DataFrame(corrected.tolist(), index=spectra.index, columns=spectra.columns)

def snv_normalization(spectra):
    """Apply Standard Normal Variate (SNV) normalization to each spectrum."""
    normalized = spectra.apply(lambda row: (row - np.mean(row)) / np.std(row), axis=1)
    # Return the DataFrame as is (do not use .tolist())
    return normalized

def sg_filtering(spectra, window_length=30, polyorder=2, deriv=0):
    """Apply Savitzky–Golay filtering along the wavelength axis for each spectrum."""
    filtered = spectra.apply(lambda row: savgol_filter(row, window_length=window_length, polyorder=polyorder, deriv=deriv), axis=1)
    return pd.DataFrame(filtered.tolist(), index=spectra.index, columns=spectra.columns)

def plot_confidence_ellipse(ax, x, y, confidence=0.95, **kwargs):
    """Plot a confidence ellipse based on the covariance of x and y."""
    if x.size != y.size:
        raise ValueError("x and y must have the same size.")
    cov_xy = np.cov(x, y)
    mean_x, mean_y = np.mean(x), np.mean(y)
    eigvals, eigvecs = np.linalg.eigh(cov_xy)
    order = eigvals.argsort()[::-1]
    eigvals, eigvecs = eigvals[order], eigvecs[:, order]
    vx, vy = eigvecs[:, 0]
    theta = np.degrees(np.arctan2(vy, vx))
    chi2_val = chi2.ppf(confidence, 2)
    width, height = 2 * np.sqrt(eigvals * chi2_val)
    ellipse = patches.Ellipse((mean_x, mean_y), width, height, angle=theta,
                              fill=False, linestyle='--', linewidth=2, **kwargs)
    ax.add_patch(ellipse)

def mahalanobis_squared(score, center, inv_cov):
    diff = score - center
    return diff.T @ inv_cov @ diff

def plot_spectral(title, plot_name, all_classes, X_clean, img_folder):
    plt.figure(figsize=(12, 8))
    sns.set_context("paper", font_scale=2)
    sns.set_style("whitegrid")
    palette = sns.color_palette("viridis", n_colors=len(all_classes))
    color_mapping = {c: palette[i] for i, c in enumerate(all_classes)}
    
    for index, row in X_clean.iterrows():
        # index[1] is assumed to be the 'Class' label
        sns.lineplot(x=X_clean.columns, y=row, label=index[1], color=color_mapping[index[1]])
    plt.gca().set_xlim([X_clean.columns.min(), X_clean.columns.max()])
    plt.gca().set_ylim([X_clean.min().min(), X_clean.max().max()])
    
    legend_elements = [Line2D([0], [0], color=color_mapping[label], lw=2, label=label) for label in color_mapping]
    plt.legend(handles=legend_elements, title="", fontsize=20, title_fontsize=22)
    plt.title(title)
    plt.xlabel("Wavelength (nm)")
    plt.ylabel("Reflectance")
    plt.grid(False)
    
    plt.tight_layout(pad=0)
    plt.savefig(os.path.join(img_folder, plot_name), bbox_inches='tight')
    plt.close()
    
    print(f'Plot saved: {plot_name}')

# ------------------------------
# Main Pipeline Function
# ------------------------------
def run_pipeline(n_components, preopocessing_ops):
    pipeline_name = f"{n_components}_" +  "_".join(preopocessing_ops).upper()
    print("--------------------------------------------------")
    print("\nRunning pipeline:", pipeline_name)
    
    # --- Create images folder if not exists ---
    img_folder = f"imgs/{pipeline_name}/"
    if not os.path.exists(img_folder):
        os.makedirs(img_folder)
    
    # Load and prepare data
    df = pd.read_csv('data_melted.csv')
    
    df.loc[df.Class == 0, 'Class'] = 'Controlled (C)'
    df.loc[df.Class == 1, 'Class'] = 'Stressed Water (W)'
    
    df = df[df['Wavelength'] >= 400]
    
    df['Reflectance_log'] = np.log10(df['Reflectance'])
    col_group = ['Date', 'Class', 'Stress_weight', 'Position']
    df['Reflectance_raw'] = df['Reflectance']
    
    X_df = df.pivot_table(index=col_group, columns='Wavelength', values='Reflectance_raw')
    print("Samples per class:\n", X_df.index.get_level_values('Class').value_counts())
    
    plot_spectral("Raw Data", f"all_raw_spectra.pdf", df['Class'].unique(), X_df, img_folder)
    
    # --- Apply processing operations ---
    title = ""
    for preop in preopocessing_ops:
        if preop.upper() == "MSC":
            X_df = msc_normalization(X_df)
        elif preop.upper() in ["SVN", "SNV"]:
            X_df = snv_normalization(X_df)
        elif preop.upper() == "SG":
            X_df = sg_filtering(X_df)
        elif preop.upper() == "SG1":
            X_df = sg_filtering(X_df, deriv=1)
            
        else:
            raise ValueError("Invalid preprocessing operation. Use 'MSC', 'SVN', or 'SG'.")
        title += f"{preop.upper()} + "
    title = title.rstrip(" +")
    
    print(f"Pipeline applied!")
    X_processed = X_df
    
    # --- PCA-based Outlier Detection ---
    pca = PCA(n_components=n_components)
    scores = pca.fit_transform(X_processed)
    center = np.mean(scores, axis=0)
    cov = np.cov(scores, rowvar=False)
    inv_cov = np.linalg.inv(cov)
    distances_squared = np.array([mahalanobis_squared(score, center, inv_cov) for score in scores])
    chi2_threshold = chi2.ppf(0.95, n_components)
    print("Chi-square threshold (95% confidence):", chi2_threshold)
    outlier_mask = distances_squared > chi2_threshold
    num_outliers = np.sum(outlier_mask)
    X_clean = X_processed[~outlier_mask]
    print(f"Number of outlier samples removed: {num_outliers}")
    print("Remaining samples:", X_clean.shape[0])
    # Print the number of samples per class
    print("Samples per class after outlier removal:\n", X_clean.index.get_level_values('Class').value_counts())
    
    # --- Plot 1: Outlier Detection ---
    plt.figure(figsize=(10, 6))
    colors = ['red' if d > chi2_threshold else 'blue' for d in distances_squared]
    plt.scatter(range(len(distances_squared)), distances_squared, c=colors, s=50, label='Samples')
    plt.axhline(chi2_threshold, color='green', linestyle='--', linewidth=2, label='Chi2 Threshold (95%)')
    plt.xlabel('Sample Index')
    plt.ylabel('Squared Mahalanobis Distance')
    # plt.title(f'Outlier Detection ({pipeline_name}, {n_components} comp)')
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(img_folder, f"outlier_detection_{pipeline_name}.pdf"))
    plt.close()
    
    # --- Plot 2: PCA Scores with Confidence Ellipse ---
    plt.figure(figsize=(10, 8))
    inlier_mask = ~outlier_mask
    plt.scatter(scores[inlier_mask, 0], scores[inlier_mask, 1], c='blue', s=50, label='Inliers')
    plt.scatter(scores[outlier_mask, 0], scores[outlier_mask, 1], c='red', s=70, marker='X', label='Outliers')
    plt.xlabel('PC1')
    plt.ylabel('PC2')
    plt.title(f'PCA Scores ({pipeline_name}, {n_components} comp)')
    ax = plt.gca()
    plot_confidence_ellipse(ax, scores[inlier_mask, 0], scores[inlier_mask, 1], confidence=0.95, edgecolor='green')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(img_folder, f"pca_scores_{pipeline_name}.pdf"))
    plt.close()
    
    # --- Plot 3: Cleaned and All Spectra using Seaborn ---
    # plot_spectral(f'Cleaned Spectra ({pipeline_name}, {n_components} comp)', f"cleaned_spectra_{pipeline_name}.pdf", 
    #               df['Class'].unique(), X_clean, img_folder)
    plot_spectral(title, f"cleaned_spectra_{pipeline_name}.pdf", df['Class'].unique(), X_clean, img_folder)
    
    # plot_spectral(f'All Spectra ({pipeline_name}, {n_components} comp)', f"all_spectra_{pipeline_name}.pdf", 
    #               df['Class'].unique(), X_processed, img_folder)
    plot_spectral("", f"all_spectra_{pipeline_name}.pdf", df['Class'].unique(), X_processed, img_folder)
    
    # --- Run the CARS Model ---
    print("\nRunning CARS model...")
    start_time = time.time()
    path = os.path.join(os.path.abspath(os.getcwd()), pipeline_name, 'CARS')
    c = WST(path, col_group, X_clean, MAX_COMPONENTS=n_components, CV_FOLD=5, test_percentage=0.3, cutoff=0.5)
    c.perform_pca()
    c.cars_model(R=500, N=100, MC_SAMPLES=0.8, start=0)
    
    save_elapsed_time(start_time, path)

    # --- Run the BOSS Model ---
    print("\nRunning BOSS model...")
    start_time = time.time()
    path = os.path.join(os.path.abspath(os.getcwd()), pipeline_name, 'BOSS')
    c = WST(path, col_group, X_clean, MAX_COMPONENTS=n_components, CV_FOLD=5, test_percentage=0.3, cutoff=0.5)
    c.perform_pca()
    c.save_results()
    c.boss_model(speed=0)
    
    save_elapsed_time(start_time, path)

    # --- Run the GA-iPLS Model ---
    print("\nRunning GA-iPLS model...")
    start_time = time.time()
    path = os.path.join(os.path.abspath(os.getcwd()), pipeline_name, 'GA-iPLS')
    c = WST(path, col_group, X_clean, MAX_COMPONENTS=n_components, CV_FOLD=5, test_percentage=0.3, cutoff=0.5)
    c.perform_pca()
    c.save_results()
    c.compute_ga_ipls(population_size=40, generations=100, crossover_prob=0.6, mutation_prob=0.1, n_intervals=100)
    
    save_elapsed_time(start_time, path)

    # --- Run the GA-iPLS + BOSS Model ---
    print("\nRunning GA-iPLS + BOSS model...")
    start_time = time.time()
    
    # Select the only wavelengths that were selected by the GA-iPLS
    file_name = f'{pipeline_name}/GA-iPLS'
    path = os.path.join(os.path.abspath(os.getcwd()), file_name)
    c = WST(path, MAX_COMPONENTS=n_components, col_group=col_group, cutoff=0.5)
    w_str = c.compute_survived_wavelengths_best_score_single()
    w_int = [np.int64(i.split('.')[0]) for i in w_str]
    # Select only the wavelengths that were selected by the GA-iPLS
    X_clean = X_clean[w_int]  
    
    # Run the BOSS model with the selected wavelengths
    path = os.path.join(os.path.abspath(os.getcwd()), pipeline_name, 'GA-iPLS_BOSS')
    c = WST(path, col_group, X_clean, MAX_COMPONENTS=n_components, CV_FOLD=5, test_percentage=0.3, cutoff=0.5)
    c.perform_pca()
    c.save_results()
    c.boss_model(speed=0)
    
    save_elapsed_time(start_time, path)
       
    print(f"Pipeline {pipeline_name} completed.\n")
    print("--------------------------------------------------")
    print()
    print()

def save_elapsed_time(start_time, path):
    end_time = time.time()
    elapsed_time = end_time - start_time
    hours, remainder = divmod(elapsed_time, 3600)
    minutes, seconds = divmod(remainder, 60)
    print(f"Elapsed time: {int(hours)}:{int(minutes):02}:{int(seconds):02}")
    # Save the elapsed time to a file
    with open(os.path.join(path, f"elapsed_time.txt"), 'w') as f:
        f.write(f"Elapsed time: {int(hours)}:{int(minutes):02}:{int(seconds):02}\n")

# ------------------------------
# Main: Loop Through All Combinations
# ------------------------------
def main():
    # Define the four allowed processing pairs
    combinations = [
        ["SG", "SVN"],
        ["SG", "MSC"],
        ["SG1", "SVN"],
        ["SG1", "MSC"]
    ]

    n_components_max = 10
       
    img_folder = "imgs"
    if not os.path.exists(img_folder):
        os.makedirs(img_folder)
    
    for combination in combinations:
        run_pipeline(n_components_max, combination)

if __name__ == "__main__":
    main()
