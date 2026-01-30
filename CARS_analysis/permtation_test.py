#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os 
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

PLOT_DIR = os.path.join(os.path.abspath(os.getcwd()), 'p_plots/') 
# Create the directory if it does not exist
if not os.path.exists(PLOT_DIR):
    os.makedirs(PLOT_DIR)
sns.set_style('darkgrid')

def show_histoplot(file_name, n_components, n_wavelenghts, calibrate):
    # print(file_name.split('/')[-2] + '/' + file_name.split('/')[-1])
    path = os.path.join(os.path.abspath(os.getcwd()), file_name) 
    # data = []
    # with open(path, 'r') as f:
    #     accuracy = float(f.readline().strip().split(','))
    #     for line in f:
    #         data.append(float(line.strip()))

    # Read the CSV file with a comma as the separator
    df = pd.read_csv(path, sep=',')
    # take first row as correct accuracy and remove it
    best_model = df.iloc[0] 
    df = df.drop(0)

    # Iterate over all columns and plot the histogram            

    metrics_p = {}
    for column in df.columns[:-1]:
        # print(column)
        data = df[column]
        # print(data)
        # # Count how many Nan values are in the column
        # nan_count = data.isna().sum()
        # print(f'Column {column} has {nan_count} NaN values')
        # # Remove the NaN values
        # data = data.dropna()
        metric = best_model[column]

        plt.figure(figsize=(12, 8))
        sns.set_context("paper", font_scale=2)  # Adjust font scale for better readability
        sns.set_style("whitegrid")       

        sns.histplot(data, binwidth=0.01, kde=True, linewidth=3)
        # if column == 'Q2':
        #     plt.xlim(-0.25, 0.75)
        # else:
        #     plt.xlim(0, 1)
        # plt.xlabel(f'{column} score')

        # Write the line and the value of the accuracy
        plt.axvline(metric, color='r', linestyle='dashed', linewidth=3)
        plt.text(metric, 0.9 * plt.ylim()[1], f' {metric:.2f}', color='r')

        # Save the plot
        name = file_name.split('/')[-2] + '/' + file_name.split('/')[-1]
        path_plot = os.path.join(PLOT_DIR, name.split('.')[0] + '/')
        if not os.path.exists(path_plot):
            os.makedirs(path_plot)

        # Add labels and title
        plt.xlabel(f'{column} Score',fontsize=26)
        plt.ylabel("Occurrences",fontsize=26)
        plt.rc('xtick',labelsize=24)
        plt.rc('ytick',labelsize=24)
        # plt.grid(False)
        plt.title("p-value", fontsize=26)

        plt.savefig(path_plot + column + '.png')
        plt.savefig(path_plot + column + '.pdf')

        plt.title(f'{column} Permutation Test')
        plt.savefig(path_plot + column + 'titled_.png')
        plt.savefig(path_plot + column + 'titled_.pdf')
        # plt.show()
        plt.close()

        # calculate p-value
        p_sum = sum([1 for x in data if x >= metric])
        p = p_sum / len(data)
        metrics_p[column] = p

    # Print following format
    # For LATEX table Accuracy,Recall,Precision,F1,ROC,Q2
    print(f'{n_components} & {calibrate} & {n_wavelenghts} & {metrics_p["Accuracy"]:.2f} & {metrics_p["Recall"]:.2f} & {metrics_p["Precision"]:.2f} & {metrics_p["F1"]:.2f} & {metrics_p["Q2"]:.2f} & {metrics_p["ROC"]:.2f} \\\\ \hline') 



# In[2]:


show_histoplot('CARS_PLS_DA_SNV_2/permutation_test_1133.csv', 2, 1133, 'False')
show_histoplot('CARS_PLS_DA_SNV_2/permutation_test_12.csv', 2, 12, 'False')
show_histoplot('CARS_PLS_DA_SNV_2/permutation_test_6.csv', 2, 6, 'False')
show_histoplot('CARS_PLS_DA_SNV_calibration_2/permutation_test_1133.csv', 2, 1133, 'True')
show_histoplot('CARS_PLS_DA_SNV_calibration_2/permutation_test_10.csv', 2, 10, 'True')
show_histoplot('CARS_PLS_DA_SNV_calibration_2/permutation_test_6.csv', 2, 6, 'True')


# In[3]:


show_histoplot('CARS_PLS_DA_SNV_3/permutation_test_1133.csv', 3, 1133, 'False')
show_histoplot('CARS_PLS_DA_SNV_3/permutation_test_25.csv', 3, 25, 'False')
show_histoplot('CARS_PLS_DA_SNV_3/permutation_test_19.csv', 3, 19, 'False')
show_histoplot('CARS_PLS_DA_SNV_calibration_3/permutation_test_1133.csv', 3, 1133, 'True')
show_histoplot('CARS_PLS_DA_SNV_calibration_3/permutation_test_30.csv', 3, 30, 'True')
show_histoplot('CARS_PLS_DA_SNV_calibration_3/permutation_test_20.csv', 3, 20, 'True')


# In[ ]:


show_histoplot('CARS_PLS_DA_SNV_4/permutation_test_1133.csv', 4, 1133, 'False')
show_histoplot('CARS_PLS_DA_SNV_4/permutation_test_34.csv', 4, 35, 'False')
show_histoplot('CARS_PLS_DA_SNV_4/permutation_test_19.csv', 4, 19, 'False')
show_histoplot('CARS_PLS_DA_SNV_calibration_4/permutation_test_1133.csv', 4, 1133, 'True')
show_histoplot('CARS_PLS_DA_SNV_calibration_4/permutation_test_45.csv', 4, 19, 'True')

