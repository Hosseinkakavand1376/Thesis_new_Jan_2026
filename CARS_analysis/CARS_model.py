from itertools import combinations
import os
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, f1_score, mean_squared_error, precision_score, recall_score
from sklearn.preprocessing import LabelBinarizer
from sklearn.cross_decomposition import PLSRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import learning_curve, train_test_split
import matplotlib.pyplot as plt
import plotly.express as px
import random
import math
from sklearn.decomposition import PCA
import seaborn as sns
from sklearn.model_selection import StratifiedKFold
from tqdm import tqdm
from sklearn.metrics import roc_curve, auc

import warnings
warnings.filterwarnings("ignore")

from sklearn.base import BaseEstimator, ClassifierMixin, TransformerMixin

sns.set_context("paper", font_scale=2)  # Adjust font scale for better readability
sns.set_style("whitegrid")
color_line = '#1f77b4'

class IdentityScaler(BaseEstimator, TransformerMixin):
    """
    A custom scaler that behaves as y = x, 
    i.e., it does nothing to the data.
    """
    def fit(self, X, y=None):
        # No fitting is needed for identity scaling
        return self
    
    def transform(self, X):
        # Simply return the input data unchanged
        return X
    
    def fit_transform(self, X, y=None):
        # Fit and transform are the same for identity scaling
        return self.fit(X, y).transform(X)

class PLSDAClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, n_components=2,cutoff=0.5):
        self.n_components = n_components
        self.cutoff = cutoff
        self.pls = PLSRegression(n_components=self.n_components)

    def fit(self, X, y):
        self.classes_ = np.unique(y)
        self.y_binary = (y == self.classes_[1]).astype(int)
        self.pls.fit(X, self.y_binary)
        return self

    def predict(self, X):
        y_pred_continuous = self.pls.predict(X)       
        y_pred = (y_pred_continuous >= self.cutoff).astype(int).ravel()
        return self.classes_[y_pred]

    def predict_proba(self, X):
        y_pred_continuous = self.pls.predict(X)
        y_pred_proba = np.hstack([1 - y_pred_continuous, y_pred_continuous])
        return y_pred_proba

class CARS:
    def __init__(self, path, col_group=None, X_df=None, MAX_LABEL=10, MAX_COMPONENTS=10, CV_FOLD=5, OPTIMAL_N=20, calibration=True, class_column='Class', test_percentage=0.3, scalar=None):        
        # Max number of labels to show in the scree plot
        self.MAX_LABEL = MAX_LABEL
        # Max number of components to use
        self.MAX_COMPONENTS = MAX_COMPONENTS
        # Number of folds for cross-validation
        self.CV_FOLD = CV_FOLD
        # Default value of optimal number of components
        self.OPTIMAL_N = OPTIMAL_N

        self.MAX_W = 10
        self.TEST  = test_percentage
        
        self.col_group   = col_group
        self.col_class   = class_column 
        
        # Get the name of the file that call the function
        self.path              = self._check_exists(path)
        self.path_statistics   = self._check_exists(f'{path}/statistics')
        self.path_coefficients = self._check_exists(f'{path}/coefficients')

        # Set scalar
        self.scalar = scalar

        if X_df is not None:
            self.col_class_i  = X_df.index.names.index(class_column)
            self.class_labels = X_df.index.get_level_values(class_column).unique()
            
            if calibration:
                # Drop the 50% of rows with data equal to each class (Balanced classes)
                X_df, X_2_df = self.divide_data(X_df, 0.5, class_column)
            else:
                X_2_df = X_df

            # Extract the X_test data from X_df
            if self.TEST == None:
                X_test = X_df
            else:
                X_df, X_test = self.divide_data(X_df, self.TEST, class_column)
            
            # Extract the X_2_test data from X_df
            if self.TEST == None:
                X_2_test = X_2_df
            else:
                X_2_df, X_2_test = self.divide_data(X_2_df, self.TEST, class_column)
            
            self.df_original = X_df
            # self.X, self.y = None, None
            self.X, self.y = self._random_input(self.df_original, shuffle=True)

            self.df_test = X_test
            # self.X_test, self.y_test = None, None
            self.X_test, self.y_test = self._random_input(self.df_test, shuffle=True)
            
            self.df_original_2 = X_2_df
            self.X_2, self.y_2 = self._random_input(self.df_original_2, shuffle=True)
            
            self.df_test_2 = X_2_test
            self.X_test_2, self.y_test_2 = self._random_input(self.df_test_2, shuffle=True)

            self.N_SAMPLES = self.X.shape[0]
            self.P = self.X.shape[1]

            self.columns = self.df_original.columns
            self.wavelengths = np.array([f"{w:.3f}" for w in self.columns])
            self.wavelengths = self.df_original.columns
        else:
            self.load_results(class_column)

    def divide_data(self, X_df, frac, class_column, random_state=42):
        X_2_df = pd.DataFrame()
        for c in self.class_labels:
            X_2_df = X_2_df._append(X_df.loc[(X_df.index.get_level_values(class_column) == c)].sample(frac=frac, random_state=random_state))
                # print(f"Class {c}: {X_df.loc[(X_df.index.get_level_values('Class') == c)].sample(frac=frac, random_state=42).shape}")
            # Remove the test data from the training data
        X_df = X_df.drop(X_2_df.index)
        return X_df, X_2_df

    def _random_input(self, dataset, shuffle=False):
        # X is the feature matrix and y are labels

        # Shuffle the dataset
        if shuffle:
            df = dataset.sample(frac=1)
        
        df = dataset
        X = df.to_numpy()
        y_label = np.array([x[self.col_class_i] for x in df.index])
        # Binarize the output
        y = LabelBinarizer().fit_transform(y_label)

        return X, y

    def _check_exists(self, path):
        # Check if the path exists, if not create it
        if not os.path.exists(f'{path}'):
            os.makedirs(f'{path}')
        return path

    def pca_screen_plot(self):
        fig = px.histogram(self.pca_screen_df, x='Components', y='Variance', title='Ratio value sampling runs',
                      labels={'Components': 'Number of components', 'Variance': 'Variance'})
        fig.show()

    def perform_pca(self, survived=False, wavelentghs=None):
        index_row = self.df_original.index
        if survived:
            survived_w = self.survived_df['Wavelengths'].value_counts()
            i = survived_w[survived_w>=1].index
            X_p = self.df_original.fillna(0).values[:,i]
        elif wavelentghs is not None:
            vars_selected_i = [self.wavelengths.index(w) for w in wavelentghs]
            X_p = self.df_original.fillna(0).values[:,vars_selected_i]
        else:
            X_p = self.df_original.fillna(0).values
        
        # Step 1: Standardize the data (mean = 0, variance = 1)
        # scaler = StandardScaler()
        # X = scaler.fit_transform(X_p)
        # X = preprocessing.scale(X_p)
        X = X_p
        
        # Create a PCA instance: pca
        pca = PCA(n_components=self.MAX_COMPONENTS)

        # Apply the dimensionality reduction on X
        X_pca = pca.fit_transform(X)

        per_var = np.round(pca.explained_variance_ratio_* 100, decimals=1)
        labels = [f'PC{i}' for i in range(1, len(per_var)+1)]

        # Create a DataFrame that will have
        self.pca_screen_df = pd.DataFrame({'Components': labels, 'Variance': per_var})

        self.pca_df = pd.DataFrame(X_pca, index=index_row, columns=labels)

        # Create an empty DataFrame
        self.PCA_1_labels = f'PC1 - {per_var[0]:.2f}%'
        self.PCA_2_labels = f'PC2 - {per_var[1]:.2f}%'
        if self.MAX_COMPONENTS > 2:
            self.PCA_3_labels = f'PC3 - {per_var[2]:.2f}%'
            self.df_PCA = pd.DataFrame(columns=[self.PCA_1_labels, self.PCA_2_labels, self.PCA_3_labels, *self.col_group])
        else:
            self.df_PCA = pd.DataFrame(columns=[self.PCA_1_labels, self.PCA_2_labels, *self.col_group])

        for (i, sample) in enumerate(self.pca_df.index):
            # Create a new DataFrame to append
            new_data = pd.DataFrame({self.PCA_1_labels: [self.pca_df.PC1.iloc[i]],
                                     self.PCA_2_labels: [self.pca_df.PC2.iloc[i]]})  
                                    #  self.PCA_3_labels: [self.pca_df.PC3.iloc[i]],
                                    #  'Class'           : [sample[0]],
                                    #  'Acquisition'    : [sample[1]],
                                    #  'Position'       : [sample[2]],
                                    #  'N'              : [sample[3]]     
                
            if self.MAX_COMPONENTS > 2:
                new_data[self.PCA_3_labels] = self.pca_df.PC3.iloc[i]
            
            for (j, n) in enumerate(self.col_group):
                new_data[n] = sample[j]
                
            # Append the new data
            self.df_PCA = pd.concat([self.df_PCA, new_data], ignore_index=True)

    def pca_3d_plot(self):
        if self.MAX_COMPONENTS > 2:
            WIDTH = 1000
            HEIGHT = 1000
            OPACITY = 0.7

            fig = px.scatter_3d(self.df_PCA, x=self.PCA_1_labels, y=self.PCA_2_labels, z=self.PCA_3_labels,
                                color='Class', title='PCA plot', opacity=OPACITY, width=WIDTH, height=HEIGHT, hover_data=self.col_group)
            # Update the layout to make sure the axis labels are set
            fig.update_layout(scene=dict(aspectmode="cube"))
            fig.show()
        else:
            print("The number of components is less than 3")

    def pca_2d_plot(self, W=8, L=6, suffix=''):
        WIDTH = 500
        HEIGHT = 500
        OPACITY = 0.7
        
        fig=px.scatter(self.df_PCA, x=self.PCA_1_labels, y=self.PCA_2_labels, color='Class', title='PCA1 vs PCA2',
                       opacity=OPACITY, width=WIDTH, height=HEIGHT, hover_data=self.col_group)
        fig.show(end=False)
        # seaborn scatter plot
        sns.set(font_scale=1.2)
        
        # Create the figure with the specified size
        plt.figure(figsize=(W, L))      
        sns.scatterplot(x=self.PCA_1_labels, y=self.PCA_2_labels, data=self.df_PCA, hue='Class', palette='viridis')
        if (suffix != ''):
            plt.savefig(f'{self.path}/PCA1_vs_PCA2_{suffix}.pdf', bbox_inches='tight')
        else:
            plt.savefig(f'{self.path}/PCA1_vs_PCA2.pdf', bbox_inches='tight')
        plt.show()
        plt.close() 

        if self.MAX_COMPONENTS > 2:
            fig=px.scatter(self.df_PCA, x=self.PCA_1_labels, y=self.PCA_3_labels, color='Class', title='PCA1 vs PCA3',
                        opacity=OPACITY, width=WIDTH, height=HEIGHT, hover_data=self.col_group)
            fig.show()
            plt.figure(figsize=(W, L))      
            sns.scatterplot(x=self.PCA_1_labels, y=self.PCA_3_labels, data=self.df_PCA, hue='Class', palette='viridis')
            if (suffix != ''):
                plt.savefig(f'{self.path}/PCA1_vs_PCA3_{suffix}.pdf', bbox_inches='tight')
            else:
                plt.savefig(f'{self.path}/PCA1_vs_PCA3.pdf', bbox_inches='tight')
            plt.show()
            plt.close()
            
            fig=px.scatter(self.df_PCA, x=self.PCA_2_labels, y=self.PCA_3_labels, color='Class', title='PCA2 vs PCA3',
                        opacity=OPACITY, width=WIDTH, height=HEIGHT, hover_data=self.col_group)
            fig.show()
            plt.figure(figsize=(W, L))      
            sns.scatterplot(x=self.PCA_2_labels, y=self.PCA_3_labels, data=self.df_PCA, hue='Class', palette='viridis')
            if (suffix != ''):
                plt.savefig(f'{self.path}/PCA2_vs_PCA3_{suffix}.pdf', bbox_inches='tight')
            else:
                plt.savefig(f'{self.path}/PCA2_vs_PCA3.pdf', bbox_inches='tight')
            plt.show()
            plt.close()
        sns.set(font_scale=1)
        
    def pca_2d_pair_plot(self, name='PCA_pair_plot'):
        if self.MAX_COMPONENTS > 2:
            sns.pairplot(data=self.df_PCA, vars=[self.PCA_1_labels, self.PCA_2_labels, self.PCA_3_labels], hue='Class', palette='viridis')
            plt.savefig(f'{self.path}/{name}.pdf', bbox_inches='tight')
            plt.show()
            plt.close()
        else:
            print("The number of components is less than 3")

    def cars_model(self, R=500, N=100, MC_SAMPLES=0.8, rmsecv=False, ars=False, start=0):
        # Statistics DataFrame
        #self.statiscs_df     = pd.DataFrame(columns=['Run', 'Iteration', 'Ratio', 'Selected Variables', 'Selected Wavelengths', 'RMSECV'])
        self.statiscs_df     = pd.DataFrame(columns=['Run', 'Iteration', 'Ratio', 'Selected Variables', 'Selected Wavelengths', 'Accuracy'])
        self.coefficients_df = pd.DataFrame(columns=['Run', 'Iteration', 'Wavelengths', 'Cofficients'])

        # Calculate the number of rows to select
        K = int(MC_SAMPLES * self.N_SAMPLES)

        self._compute_cars(K, N, start, start+R, ars, rmsecv)

    def _compute_cars(self, K, N, start, end, ars, rmsecv):
        # print(K)
        for j in tqdm(range(start, end)):
            # Select all the variables at the beginning
            vars_selected_i = list(range(len(self.wavelengths)))
            self._components_number(vars_selected_i, list(range(self.N_SAMPLES)))
            
            # Shuffle input data
            self.X, self.y = self._random_input(self.df_original)

            for i in range(1, N+1):
                # Randomly choose K samples to build a PLS model
                indices = np.random.choice(self.N_SAMPLES, size=K, replace=False)
                # print(indices)

                # Compute PLS using the new selected variables and K percentage of samples
                self.pls = self._compute_model(self.X, self.y, vars_selected_i, indices)

                ext_sel_cofficients = self._compute_coefficients_and_weights(vars_selected_i)
                wavelength_weights  = self._compute_wavelength_weights(ext_sel_cofficients)

                # Calculate the ratio of variable to keep
                r = self._compute_ratio(N, i)
                n_selected_variables = int(round(r*self.P, 0))

                # Remove the wavelengths which are of relatively small absolute regression coefficients
                vars_selected_i = self._select_variables(ars, n_selected_variables, wavelength_weights)
                n_selected_variables = len(vars_selected_i)

                # Compute the new PLS model using the new selected variables and all the samples
                self._components_number(vars_selected_i, range(self.N_SAMPLES))
                indices = np.random.choice(self.N_SAMPLES, size=self.N_SAMPLES, replace=False)
                # self.pls = self._compute_model(self.X, self.y, vars_selected_i, indices)

                if rmsecv:
                    # rmsecv = self._compute_rmsecv()
                    accuracy = self._compute_accuracy(self.X, self.y, self.X_test, self.y_test, var_selected_i=vars_selected_i)
                    # print(f'{i}: {accuracy}')
                else:
                    rmsecv   = 0
                    accuracy = 1

                if n_selected_variables > self.MAX_W:
                    selected_wavelengths = f"{len(vars_selected_i)} wavelengths"
                else:
                    selected_wavelengths = ""
                    for w_i in vars_selected_i:
                        selected_wavelengths += f"{self.wavelengths[w_i]} - "
                    selected_wavelengths = selected_wavelengths[:-3]
                new_data = pd.DataFrame({'Run':                 j,
                                        'Iteration':            i,
                                        'Ratio':                r,
                                        'Selected Variables':   n_selected_variables,
                                        'Selected Wavelengths': selected_wavelengths,
                                        'N components':         self.OPTIMAL_N,
                                        'RMSECV':               rmsecv,
                                        'Accuracy':             accuracy}, index=[0])
                self.statiscs_df = pd.concat([self.statiscs_df, new_data], ignore_index=True)

                new_data = pd.DataFrame({'Run':                j,
                                        'Iteration':           i,
                                        'Wavelengths':         self.wavelengths[vars_selected_i],
                                        'Cofficients':         (self.pls.coef_).flatten()})
                self.coefficients_df = pd.concat([self.coefficients_df, new_data], ignore_index=True)

            self._save_partial(j)
        self.save_results()

    def _save_partial(self, run):
        # print("save run", run)
        self.statiscs_df    [self.statiscs_df.Run     == run].to_csv(f'{self.path_statistics}/statistics_{run}.csv',     index=False)
        self.coefficients_df[self.coefficients_df.Run == run].to_csv(f'{self.path_coefficients}/coefficients_{run}.csv', index=False)

    def _select_variables(self, ars, n_selected_variables, wavelength_weights):
        sorted_indices  = np.argsort(wavelength_weights)[::-1]
        vars_selected_i = sorted_indices[:n_selected_variables]

        # ARS
        if ars:
            vars_selected_i = np.array(list(set(random.choices(range(0, len(wavelength_weights)), wavelength_weights, k=n_selected_variables))))

        return np.sort(vars_selected_i)

    def _compute_ratio(self, N, i):
        a = (self.P/2)**(1/(N-1))
        k = math.log(self.P/2)/(N-1)
        r = a*math.exp(-k*i)
        return r

    def _compute_wavelength_weights(self, ext_sel_cofficients):
        tot = np.sum(np.abs(ext_sel_cofficients))
        return np.array([x/tot for x in np.abs(ext_sel_cofficients).flatten()])

    def _compute_coefficients_and_weights(self, vars_selected_i):
        ext_sel_cofficients = np.zeros(self.P)
        ext_sel_cofficients[vars_selected_i] = (self.pls.coef_[0]).flatten()
        return ext_sel_cofficients

    def _components_number(self, vars_selected_i, indices):
        # Create a list to store the rmse_cv values
        #rmsecv_list = []
        accuracy_list =[]
        # Range of components number to test
        if len(vars_selected_i) > self.MAX_COMPONENTS:
            all_components = range(1, self.MAX_COMPONENTS+1)
        else:
            all_components = range(1, len(vars_selected_i)+1)

        self.OPTIMAL_N = self.MAX_COMPONENTS
        return

        for n in all_components:
            self.pls = self._compute_model(vars_selected_i, indices, n)
            #rmsecv_list.append(self._compute_rmsecv())
            accuracy_list.append(self._compute_accuracy())
        #self.OPTIMAL_N = rmsecv_list.index(min(rmsecv_list)) + 1
        self.OPTIMAL_N = accuracy_list.index(max(accuracy_list)) + 1 #NELL'ACCURACY VOGLIO VALORE MASSIMO, MENTRE NELL'MSRE VOGLIO IL MINIMO

    def _compute_rmsecv(self, X, y, X_test, y_test, var_selected_i=None, model_type='PLS'):
        # y_pred_cv = cross_val_predict(self.pls, self.X, self.y, cv=self.CV_FOLD)
        y_pred_cv = self._cross_predict(X, y, X_test, var_selected_i=var_selected_i, model_type=model_type)
        rmsecv    = np.sqrt(mean_squared_error(self.y, y_pred_cv))
        return rmsecv

    def _compute_accuracy(self, X, y, X_test, y_test, var_selected_i=None, model_type='PLS', confusion_matrix=False, all=False,cutoff=None,learning_curve=False):   
        # y_pred_cv = cross_val_predict(self.pls, self.X, self.y, cv=self.CV_FOLD)
        
        if all:
            y_pred, y_train, X_train = self._cross_predict(X, y, X_test, var_selected_i=var_selected_i, model_type=model_type, q2_flag=True)
        else:
            y_pred = self._cross_predict(X, y, X_test, var_selected_i=var_selected_i, model_type=model_type)
        
        if confusion_matrix:
            self._compute_confusion_matrix(y_test, y_pred, len(var_selected_i), cutoff=cutoff)
        
        if all:
            _, _, _, roc_auc = self._compute_roc_auc(y_test, y_pred)
            all_metrics = self._calculate_accuracy(y_test, y_pred, all=True, cutoff=cutoff)
            
            if learning_curve:
                self.learning_curve(X[:,var_selected_i], y, cutoff=all_metrics[-1], len_var=len(var_selected_i), model_type=model_type)
            
            # Calculate Q^2 for this fold
            SS_press = np.sum((y_test - y_pred) ** 2)  # Prediction error on test fold
            SS_total = np.sum((y_test - np.mean(y_train)) ** 2)  # Total variance from training mean
            q2 = 1 - (SS_press / SS_total)
            
            return *all_metrics, roc_auc, q2
        else:
            return self._calculate_accuracy(y_test, y_pred, all=False, cutoff=cutoff)

    def _calculate_accuracy(self, y, y_pred, all=False, cutoff=None):
        accuracy, recall, precision, f1, cutoff = self._all_metrics(y, y_pred, cutoff)
        if all:
            return accuracy, recall, precision, f1, cutoff
        else:
            return accuracy
        

    # The Receiver Operating Characteristic (ROC) curve is a valuable
    # for evaluating the performance of binary classification models.
    def _compute_roc(self, vars_selected_i):
        
        y_pred_cv = self._cross_predict(self.X_2, self.y_2, self.X_test_2, var_selected_i=vars_selected_i)
        # print how much are lower than 0.5 and how much are greater than 0.5
        # print(np.sum(np.array(y_pred_cv) > 0.5))
        # print(np.sum(np.array(y_pred_cv) <= 0.5))
        fpr, tpr, thresholds, roc_auc = self._compute_roc_auc(self.y_test_2, y_pred_cv)
        
        return fpr, tpr, thresholds, roc_auc

    def _compute_roc_auc(self, y_test, y_pred_cv):
        fpr, tpr, thresholds = roc_curve(y_test, y_pred_cv)
        roc_auc = auc(fpr, tpr)
        return fpr, tpr, thresholds, roc_auc

    def plot_roc(self, vars_selected_i):
        fpr, tpr, thresholds, roc_auc = self._compute_roc(vars_selected_i)
        # print(fpr.shape, tpr.shape, thresholds.shape)

        # Create a dataframe
        df = pd.DataFrame({'False Positive Rate': fpr, 'True Positive Rate': tpr, 'Thresholds': thresholds})

        # Plot the ROC curve
        fig = px.line(df, x='False Positive Rate', y='True Positive Rate', markers=True,
                      hover_data='Thresholds', title=f'Receiver Operating Characteristic - AUC {roc_auc:.2f}')

        # Set the layout
        fig.update_layout(showlegend=True)

        # Show the plot
        fig.show()

    def _compute_model(self, X, y, vars_selected_i, indices, n=None, model_type='PLS'):

        if model_type == 'PLS':
            if n is not None:
                pls = PLSRegression(n_components=n)
            else:
                pls = PLSRegression(n_components=self.OPTIMAL_N)
            
            if pls.n_components > len(vars_selected_i):
                pls = PLSRegression(n_components=len(vars_selected_i))

        elif model_type == 'LR':
            pls = LogisticRegression(penalty='elasticnet', solver='saga', l1_ratio=0.1)
            pls = LogisticRegression(penalty='l2')
            # pls = LogisticRegression(penalty='elasticnet', solver='saga', l1_ratio=0.5)

        # print(f'Indicees len:     {len(indices)}')
        # print(f'Var selected len: {len(vars_selected_i)}')
        pls.fit(X[indices, :][:, vars_selected_i], y[indices])
        
        return pls

    def plot_frequency(self, starting=500, separation=100, threshold=None, width=12, height=6):  
        # Set up a rectangular figure (wider than tall)
        # plt.figure(figsize=(12, 6))  # Width of 12 and height of 6
     
        plt.figure(figsize=(width, height))
        
        # Seaborn plot
        ax = sns.barplot(self.plot_survived_df, x='Wavelengths',  y='count')

        # Fot the x-axis labels print only some values
        x_values = [float(x) for x in self.wavelengths] 

        # # Add some space to the right of the last value
        # for _ in range(10):
        #     x_values.append(x_values[-1] + 0.5)
        # print(f'X_values: {x_values[-1]}')
        
        # Extract the desired x values separated by 100 starting from a value
        desired_ticks = [x for x in range(starting, int(max(x_values)), separation)]

        # Find the closest value to the desired ticks
        tick_positions = [str(min(x_values, key=lambda x: abs(x - tick))) for tick in desired_ticks]
        # tick_positions = [min(x_values, key=lambda x: abs(x - tick)) for tick in desired_ticks]
        
        plt.xlabel('Wavelength (nm)')
        plt.ylabel("Occurrences")
        # plt.rc('xtick',labelsize=18)
        # plt.rc('ytick',labelsize=18)
        # plt.grid(False)
        
        # Set custom x-ticks at the exact positions
        plt.xticks(tick_positions, labels=desired_ticks)
        # Add space on the left and right by adjusting the x-axis limits
        # plt.xlim(tick_positions[0] - 60, tick_positions[-1] + 60)
        
        # plt.grid(True)
        # Change the color of each bar after plotting
        for bar in ax.containers[0].get_children():
            bar.set_color(color_line)  # Set the desired color here
    
        if threshold is not None:
            # Plot orizzonatl threshold line
            plt.axhline(y=threshold, color='r', linestyle='--')
            # Write threshold 
            plt.text(200, threshold+5, 'Threshold', color='red')        
    
        plt.savefig(f'{self.path}/Frequency_of_Wavelengths.pdf')
        # plt.show()
        plt.close()
        
        # Plot the frequency of survived variables
        fig = px.histogram(self.plot_survived_df, x='Wavelengths', y='count',
                            labels={'Wavelengths': 'Wavelengths', 'sum of count': 'Frequency'},
                            title='Frequency of Wavelengths')
        fig.show()

    def plot_selected_variables(self, run=0):
        # Plot the selected variables
        fig = px.line(self.statiscs_df[self.statiscs_df['Run'] == run], x='Iteration', y='Selected Variables',
                      labels={'Iteration': 'Iteration', 'Selected Variables': 'Selected Variables'},
                      hover_name='Selected Wavelengths',
                      title='Ratio value sampling runs')
        fig.show()
        
        # Seaborn plot
        plt.figure(figsize=(14, 8))

        # Add a new row for run r with iteration 0 and total number of variables
        self.statiscs_df = pd.concat([self.statiscs_df, pd.DataFrame({'Run': run, 'Iteration': 0, 'Ratio': 1, 
            'Selected Variables': self.P, 'Selected Wavelengths': 'All wavelengths', 'RMSECV': 0, 
            'Accuracy': self.statiscs_df[(self.statiscs_df['Run'] == run) & (self.statiscs_df['Iteration'] == 1)]['Accuracy'].values[0]},
            index=[0])], ignore_index=True)
            
        sns.lineplot(data=self.statiscs_df[self.statiscs_df['Run'] == run], x='Iteration', y='Selected Variables', linewidth=3, color=color_line)
        plt.xlabel('Iteration')
        plt.ylabel('Selected Variables')
        # plt.rc('xtick',labelsize=18)
        # plt.rc('ytick',labelsize=18)
        # plt.grid(False)
        plt.xlim(0, self.statiscs_df[self.statiscs_df['Run'] == run]['Iteration'].max())
        
        plt.savefig(f'{self.path}/Selected_Variables_sampling_runs.pdf')
        plt.title('Selected Variables sampling runs')
        plt.savefig(f'{self.path}/Selected_Variables_sampling_runs_titled.pdf')
        # plt.show()
        plt.close()

    def plot_ratio(self, run=0):
        # Plot the ratio
        fig = px.line(self.statiscs_df[self.statiscs_df['Run'] == run], x='Iteration', y='Ratio',
                      labels={'Iteration': 'Iteration', 'Ratio': 'Ratio'},
                      hover_name='Selected Wavelengths',
                      title='Ratio value sampling runs')
        fig.show()
        
        # Seaborn plot
        sns.lineplot(data=self.statiscs_df[self.statiscs_df['Run'] == run], x='Iteration', y='Ratio')
        plt.xlabel('Iteration')
        plt.ylabel('Ratio')
        plt.title('Ratio value sampling runs')
        plt.savefig(f'{self.path}/Ratio_value_sampling_runs.pdf')
        # plt.show()
        plt.close()

    def plot_rmsecv(self, run=0):
        # Plot the RMSECV
        fig = px.line(self.statiscs_df[self.statiscs_df['Run'] == run], x='Iteration', y='RMSECV',
                      labels={'Iteration': 'Iteration', 'RMSECV': 'RMSECV'},
                      hover_name='Selected Wavelengths',
                      title='RMSECV value sampling runs')
        fig.show()
        
        # Seaborn plot
        sns.lineplot(data=self.statiscs_df[self.statiscs_df['Run'] == run], x='Iteration', y='RMSECV')
        plt.xlabel('Iteration')
        plt.ylabel('RMSECV')
        plt.title('RMSECV value sampling runs')
        plt.savefig(f'{self.path}/RMSECV_value_sampling_runs.pdf')
        # plt.show()
        plt.close()

    def plot_accuracy(self, run=0):
        self.statiscs_df = self.statiscs_df.sort_values(by='Iteration', ascending=False)
        fig=px.line(self.statiscs_df[self.statiscs_df['Run'] == run], x='Iteration', y='Accuracy',
                    labels={'Iteration': 'Iteration', 'Accuracy': 'Accuracy'},
                    hover_name='Selected Wavelengths',
                    title='Accuracy value sampling runs')
        fig.show()
        
        # Print the best accuracy point with lower number of wavelenghts, printing the corresponding selected wavelengths
        best_accuracy = self.statiscs_df[self.statiscs_df['Run'] == run]['Accuracy'].max()
        # print(self.statiscs_df.sort_values(by='Iteration', ascending=False)[(self.statiscs_df['Run'] == run) & (self.statiscs_df['Accuracy'] == best_accuracy)])
        best_accuracy_row = self.statiscs_df.sort_values(by='Iteration', ascending=False)[(self.statiscs_df['Run'] == run) & (self.statiscs_df['Accuracy'] == best_accuracy)]
        wavelengths = self.survived_df[self.survived_df['Run'] == run]['Wavelengths'].values
        print(f'Best accuracy: {best_accuracy} at iteration {best_accuracy_row["Iteration"].values[0]} with {wavelengths}')
        
        # Seaborn plot
        plt.figure(figsize=(14, 8))
        
        self.statiscs_df['Accuracy_app'] = self.statiscs_df['Accuracy'].apply(lambda x: round(x, 2))
        sns.lineplot(data=self.statiscs_df[self.statiscs_df['Run'] == run], x='Iteration', y='Accuracy_app', linewidth=3, color=color_line)
        
        # Add a red point to the best accuracy
        best_accuracy = round(best_accuracy, 2) 
        plt.plot(best_accuracy_row['Iteration'].values[0], best_accuracy, 'ro')
        plt.text(best_accuracy_row['Iteration'].values[0]+1, best_accuracy, f'{best_accuracy:.2f}', verticalalignment='top', color='r')
        
        plt.xlabel('Iteration')
        plt.ylabel('Accuracy')
        
        plt.xlim(0, self.statiscs_df[self.statiscs_df['Run'] == run]['Iteration'].max())
        
        # Set y ticks
        plt.yticks(np.arange(0.75, 0.97, 0.05))
        
        plt.savefig(f'{self.path}/Accuracy_value_sampling_runs.pdf')
        plt.title('Accuracy value sampling runs')
        plt.savefig(f'{self.path}/Accuracy_value_sampling_runs_tiled.pdf')
        # plt.show()
        plt.close()

    def plot_coefficients(self, run=0):
        # Plot all the coefficients
        fig = px.line(self.coefficients_df[self.coefficients_df['Run'] == run], x='Iteration', y='Cofficients',
                      labels={'Iteration': 'Iteration', 'Cofficients': 'Cofficients'},
                      title='Coefficients sampling runs', line_group='Wavelengths', color='Wavelengths')
        fig.show()
        
        # Seaborn plot
        sns.lineplot(data=self.coefficients_df[self.coefficients_df['Run'] == run], x='Iteration', y='Cofficients', hue='Wavelengths')
        plt.xlabel('Iteration')
        plt.ylabel('Cofficients')
        plt.title('Cofficients sampling runs')
        plt.savefig(f'{self.path}/Cofficients_sampling_runs.pdf')
        # plt.show()
        plt.close()

    def plot_coefficients_ext(self, run=0):
        # Plot all the extended coefficients

        # Reshape the coefficients_df dataframe using pivot
        coefficients_ext_df = self.coefficients_df[self.coefficients_df['Run'] == run].pivot(
            index='Iteration', columns='Wavelengths', values='Cofficients').fillna(0)

        fig = px.line(coefficients_ext_df, title='Coefficients sampling runs')

        fig.show()
        
        # Seaborn plot
        sns.lineplot(data=coefficients_ext_df, legend=False)
        plt.xlabel('Iteration')
        plt.ylabel('Cofficients')
        plt.title('Cofficients sampling runs')
        plt.savefig(f'{self.path}/Cofficients_sampling_runs.pdf')
        plt.show()
        plt.close()
        

    def plot_coefficients_1(self, run=0):
        all_cofficients = self.coefficients_df[self.coefficients_df['Run'] == run].pivot_table(index=['Iteration'], columns='Wavelengths', values='Cofficients').values
        plt.plot(all_cofficients)
        plt.xlabel('Iteration')
        plt.ylabel('Cofficients')
        plt.title('Cofficients sampling runs')

        all_cofficients = np.array(all_cofficients)
        s_indices = np.where(all_cofficients[-1] != 0)[0]
        s_coef = all_cofficients[-1, s_indices]
        s_w    = self.wavelengths[s_indices]
        x      = len(all_cofficients)

        for c in range(len(s_coef)):
            plt.plot(x-1, s_coef[c], 'ko')  # This plots a x-1,  at the location (x_value, point)
            plt.text(x, s_coef[c], f'{s_coef[c]:.2f} - {s_w[c]} nm', fontsize=12)  # This adds text at the location (x_value, y_value)
        plt.show()

    def save_results(self):
        self.df_PCA.to_csv(f'{self.path}/pca.csv', index=False)
        self.pca_screen_df.to_csv(f'{self.path}/pca_screen.csv', index=False)
        self.df_original.to_csv(f'{self.path}/X_df.csv', index=True)
        self.df_test.to_csv(f'{self.path}/X_test.csv', index=True)
        self.df_original_2.to_csv(f'{self.path}/X_2_df.csv', index=True)
        self.df_test_2.to_csv(f'{self.path}/X_2_test.csv', index=True)
        np.savetxt(f'{self.path}/wavelengths.txt', self.wavelengths, fmt='%s')
        
        with open(f'{self.path}/col_class_i.txt', 'w') as f:
            f.write(f'{self.col_class_i}\n')
        
    def load_results_partial(self):
        # Read and put togheter all files from self.path_statistics
        self.statiscs_df     = pd.DataFrame(columns=['Run', 'Iteration', 'Ratio', 'Selected Variables', 'Selected Wavelengths', 'RMSECV'])
        for file in os.listdir(self.path_statistics):
            self.statiscs_df     = pd.concat([self.statiscs_df, pd.read_csv(f'{self.path_statistics}/{file}')], ignore_index=True)

        # Read and put togheter all files from self.path_coefficients
        self.coefficients_df = pd.DataFrame(columns=['Run', 'Iteration', 'Wavelengths', 'Cofficients'])
        for file in os.listdir(self.path_coefficients):
            # print(self.path_coefficients)
            self.coefficients_df = pd.concat([self.coefficients_df, pd.read_csv(f'{self.path_coefficients}/{file}')], ignore_index=True)

    def load_results(self, class_column):
        self.load_results_partial()
        self.df_PCA          = pd.read_csv(f'{self.path}/pca.csv')
        self.pca_screen_df   = pd.read_csv(f'{self.path}/pca_screen.csv')
        self.PCA_1_labels    = self.df_PCA.columns[0]
        self.PCA_2_labels    = self.df_PCA.columns[1]
        self.PCA_3_labels    = self.df_PCA.columns[2]

        self.wavelengths = []
        with open(f'{self.path}/wavelengths.txt') as f:
            for line in f:
                self.wavelengths.append(line.strip())
                
        self.col_class_i = np.loadtxt(f'{self.path}/col_class_i.txt', dtype=int)

        # trasform it in pivot table format
        self.df_original, self.X, self.y = self._read_pivot(f'{self.path}/X_df.csv')
        self.df_test, self.X_test, self.y_test = self._read_pivot(f'{self.path}/X_test.csv')
        
        self.df_original_2, self.X_2, self.y_2 = self._read_pivot(f'{self.path}/X_2_df.csv')
        self.df_test_2, self.X_test_2, self.y_test_2 = self._read_pivot(f'{self.path}/X_2_test.csv')

        self.N_SAMPLES = self.df_original.shape[0]
        self.P         = self.df_original.shape[1]

        self.class_labels = self.df_original.index.get_level_values(class_column).unique()
        self.col_class_i  = self.df_original.index.names.index(class_column)

    def compute_survived_wavelengths_n_variables(self, n_variables=2):
        #id = self.statiscs_df[self.statiscs_df['Selected Variables'] == n_variables].groupby('Run')['RMSECV'].idxmin().tolist()
        id = self.statiscs_df[self.statiscs_df['Selected Variables'] == n_variables].groupby('Run')['Accuracy'].idxmax().tolist()
        self._compute_survived_wavelengths(id)

    def compute_survived_wavelengths_best_accuracy(self):
        #id = self.statiscs_df.sort_values(by='Selected Variables').groupby('Run')['RMSECV'].idxmin().tolist()
        # id = self.statiscs_df[self.statiscs_df['Selected Variables'] < 50].sort_values(by='Selected Variables').groupby('Run')['Accuracy'].idxmax().tolist()
        id = self.statiscs_df.sort_values(by='Selected Variables').groupby('Run')['Accuracy'].idxmax().tolist()
        self._compute_survived_wavelengths(id)

    def _compute_survived_wavelengths(self, id):
        # Filter the statistics dataframe based on the given id
        survived_stats = self.statiscs_df.iloc[id][['Run', 'Iteration']]

        # Merge the filtered statistics dataframe with the coefficients dataframe
        self.survived_df = pd.merge(survived_stats, self.coefficients_df, on=['Run', 'Iteration'])[['Wavelengths', 'Run']]

        # Sort the survived dataframe by wavelength
        self.survived_df.sort_values(by=['Wavelengths', 'Run'], inplace=True)
        self.survived_df['Wavelengths'] = self.survived_df['Wavelengths'].astype(str)
        
        # Sum the occurency of each wavelength
        count_sourvived = self.survived_df['Wavelengths'].value_counts()
        
        # Add the missing wavelengths
        self.plot_survived_df = pd.merge(pd.DataFrame(self.wavelengths, columns=['Wavelengths']), count_sourvived, on='Wavelengths', how='left').fillna(0)

    def accuracy_survived_wavelenghts(self, thr=None, rdm=False, all=False, model_type='PLS', wavelengths=None, learning_curve=False, roc_plot=False, pls_plot=False, confusion_matrix=False):  
        var_combinations = []
        
        # Keep all the wavelengths from survived df that have an occurrence greater than thr
        if wavelengths is not None:
            for e in wavelengths:
                e.sort()
                wavelengths_str = [f"{w:.3f}" if isinstance(w, float) else w for w in e]
                var_combinations.append([self.wavelengths.index(w) for w in wavelengths_str])
        
        if thr is not None:
            for t in thr:
                vars_selected_i = self.extract_select_variable(t)
                vars_selected_i.sort() # Sort the selected variables
                # Create all combinations with vars_selected_i            
                for r in range(len(vars_selected_i), len(vars_selected_i) + 1):
                    var_combinations.extend(combinations(vars_selected_i, r))

        if all:
            var_combinations.append(range(self.P))

        print(f'\n{len(var_combinations)} COMBINATIONS')
        accuracy_list = []
        w_list        = []

        # # Randomize the input data
        # df = self.df_original.sample(frac=1)
        # self.X  = df.to_numpy()
        # y_label = np.array([x[0] for x in df.index])

        # # Binarize the output
        # self.y = LabelBinarizer().fit_transform(y_label).reshape(-1, 1)

        if rdm:
            self.X_2  = self.df_original_2.sample(frac=1).to_numpy()

        latex = ""
        for i in range(len(var_combinations)-1, -1, -1):
            # Shuffle input data
            # self.X, self.y = self._random_input(self.df_original)

            w_list.append([self.wavelengths[w] for w in var_combinations[i]])
            # self._components_number(var_combinations[i], range(self.N_SAMPLES))

            # Compute PLS using the selected variables
            # indices  = range(self.N_SAMPLES)
            # self.pls = self._compute_model(var_combinations[i], indices, n=len(var_combinations[i]))
            print(f'{i}) {w_list[-1]}')
            print(len(w_list[-1]))
            accuracy, recall, precision, f1, cutoff, roc, q2  = self._compute_accuracy(self.X_2, self.y_2, self.X_test_2, self.y_test_2, all=True,
                                              var_selected_i=var_combinations[i], model_type=model_type, confusion_matrix=confusion_matrix, learning_curve=learning_curve)
            accuracy_list.append(accuracy)
            print(f'Accuracy: {accuracy:.2f} - Recall: {recall:.2f}')
            # print(f'Precision: {precision:.2f} - F1: {f1:.2f} - ROC: {roc:.2f} - Q^2: {q2:.2f} - Cutoff: {cutoff:.2f}\n')
            print(f'Precision: {precision:.2f} - F1: {f1:.2f} - ROC: {roc:.2f} - Cutoff: {cutoff:.2f}\n')

            # For LATEX table
            if self.df_original.equals(self.df_original_2):
                calibration = 'False'
            else:
                calibration = 'True'
            latex+=f'{self.MAX_COMPONENTS} & {calibration} & {len(var_combinations[i])} & {accuracy:.2f} & {recall:.2f} & {precision:.2f} & {f1:.2f} & {q2:.2f} & {roc:.2f} & {cutoff:.2f} \\\\ \hline\n'
                            
            if roc_plot:
                self.plot_roc(var_combinations[i])
                
            if pls_plot:
                self.plot_PLS(self.X_2, self.y_2, var_combinations[i])
           
        print(latex) 
        return accuracy_list, w_list

    def plot_PLS(self, X, y, var_selected_i):
        X_pls = self.pls.transform(X[:, var_selected_i])
        # Fit PLS regression
        array = np.concatenate((X_pls, y), axis=1)

        columns = [f'PLS Component {i+1}' for i in range(self.MAX_COMPONENTS)]
        columns.append('Class') 
        df_pls = pd.DataFrame(array, columns=columns)
            
        for i, c in enumerate(self.class_labels):
            df_pls['Class'][df_pls['Class'] == i] = c

        sns.set_context("paper", font_scale=1.5)
        
        # Scatter plot of the first two components
        plt.figure(figsize=(8, 6))
        sns.scatterplot(df_pls, x='PLS Component 1', y='PLS Component 2', hue='Class', palette='viridis', s=70)
        # plt.title("Scatter Plot of PLS Components")
        plt.xlabel("PLS Component 1")
        plt.ylabel("PLS Component 2")
        plt.grid(True)
        plt.savefig(f'{self.path}/PLS_Components_{len(var_selected_i)}.pdf')
        plt.show()
        plt.close()
        
        # Pairplot of all PLS components (here just two)
        # Set the figure and axes
        plt.figure(figsize=(8, 10))
        g = sns.pairplot(df_pls, hue='Class', palette='viridis', corner=False)

        # Adjust subplot spacing for proper label alignment
        g.fig.subplots_adjust(left=0.2, wspace=0.1, hspace=0.1)
        
        # Align all left y-axis labels explicitly
        for ax in g.axes[:, 0]:  # Select the first column of subplots
            if ax is not None:
                ax.yaxis.set_label_coords(-0.4, 0.5)  # Adjust label position
        
        plt.savefig(f'{self.path}/PLS_Components_pp_{len(var_selected_i)}.pdf')
        plt.show()
        plt.close()
        
        sns.set_context("paper", font_scale=2)

    def extract_select_variable(self, thr):
        survived_w = self.survived_df['Wavelengths'].value_counts()
        survived_w = survived_w[survived_w>=thr].index
        vars_selected_i = [self.wavelengths.index(w) for w in survived_w]
        return vars_selected_i

    def _cross_predict(self, X, y, X_test_f, var_selected_i=None, model_type='PLS', q2_flag=False):
        # if model_type == 'PLS':
        #     pls = PLSRegression(n_components=self.OPTIMAL_N)

        # elif model_type == 'LR':
        #     pls = LogisticRegression(penalty='elasticnet', solver='saga', l1_ratio=0.1)
        #     pls = LogisticRegression(penalty='l2')
        #     # pls = LogisticRegression(penalty='elasticnet', solver='saga', l1_ratio=0.5)
        #     print('LR')

        # y_pred = cross_val_predict(pls, self.X, self.y, cv=self.CV_FOLD)
        # return y_pred

        if var_selected_i is None:
            var_selected_i = range(self.P)
        
        b_accuracy = 0
        b_y_train  = None
        b_X_train  = None

        if self.CV_FOLD > 1:
            skf = StratifiedKFold(self.CV_FOLD)

            for train_index, test_index in skf.split(X, y):
                X_train, X_test = X[train_index], X[test_index,:][:,var_selected_i]
                y_train, y_test = y[train_index], y[test_index]

                b_accuracy, b_y_train, b_X_train = self._single_fold(X, y, var_selected_i, model_type, b_accuracy, train_index, 
                                                                     test_index, X_test, y_test, y_train, b_y_train, b_X_train)
        else:
            # X_train, X_test, y_train, y_test = train_test_split(X[:,var_selected_i], y, test_size=self.TEST, random_state=42)
            train_index, test_index = train_test_split(range(len(X)), test_size=self.TEST)
            X_train, X_test = X[train_index,:][:, var_selected_i], X[test_index,:][:, var_selected_i]
            y_train, y_test = y[train_index], y[test_index]
            
            b_accuracy, b_y_train = self._single_fold(X, y, var_selected_i, model_type, b_accuracy, train_index, test_index, X_test, y_test, y_train, b_y_train)
                
            # print(y_pred[test_index][-3:], y_test[-3:],'\n')
            # print(accuracy_score(y_test, y_pred[test_index]))
            # y_pred_cutoff = (y_pred[test_index] >= 0.5).astype(int)
            # print(y_pred[test_index][-3:], y_pred_cutoff[-3:],y_test[-3:],'\n')
            # prsint(accuracy_score(y_test, y_pred_cutoff))

        
        if self.scalar is None:
            y_pred = self.pls.predict(X_test_f[:, var_selected_i])
        else:
            y_pred = self.pls.predict(self.scalar.transform(X_test_f[:, var_selected_i]))

        if q2_flag:
            return y_pred, b_y_train, b_X_train
        else:
            return y_pred

    def _single_fold(self, X_original, y, var_selected_i, model_type, b_accuracy, train_index, test_index, X_test, y_test, y_train, b_y_train, b_X_train):
        # print how many classes for each y_train and y_test fold
        # print(np.unique(y_train, return_counts=True), np.unique(y_test, return_counts=True))

        # Check if self.pls.n_components exixts
        if model_type == 'PLS':
            n = self.MAX_COMPONENTS
        elif model_type == 'LR':
            n = 0
        
        if self.scalar is not None:
            X = X.copy()
        else:
            X = X_original
        
        # Now you can fit the model on the training data and test on the testing data
        if self.scalar is not None:
            X[train_index, :][:, var_selected_i] = self.scalar.fit_transform(X[train_index, :][:, var_selected_i])
        model = self._compute_model(X, y, var_selected_i, train_index, n, model_type)

        if self.scalar is not None:
            X_test = self.scalar.transform(X_test)

        if model_type == 'PLS':
            y_pred= model.predict(X_test)
        elif model_type == 'LR':
            y_pred = model.predict(X_test).reshape(y_pred[test_index].shape)

        y_train_pred = model.predict(X[:,var_selected_i])
        accuracy, recall, precision, f1, cutoff = self._calculate_accuracy(y, y_train_pred, all=True)
        # print(f'TRAINING: {accuracy:.2f} - {recall:.2f} - {precision:.2f} - {f1:.2f} - {cutoff:.2f}')
        accuracy, recall, precision, f1, cutoff = self._calculate_accuracy(y_test, y_pred, all=True)
        # print(f'TESTING: {accuracy:.2f} - {recall:.2f} - {precision:.2f} - {f1:.2f} - {cutoff:.2f}\n')
        if b_accuracy < accuracy:
            b_accuracy = accuracy
            b_y_train  = y_train
            b_X_train  = X[train_index, :][:, var_selected_i]
            self.pls   = model
            
        return b_accuracy, b_y_train, b_X_train

    def learning_curve(self, X, y, cutoff, len_var, model_type='PLS'):
        if model_type == 'PLS':
            # For classification, instantiate PLSDAClassifier without externally providing cutoff.
            model = PLSDAClassifier(n_components=self.MAX_COMPONENTS, cutoff=cutoff)
            train_sizes, train_scores, validation_scores = learning_curve(model, X, y,
                                                                           train_sizes=np.linspace(0.1, 1.0, 30), 
                                                                           scoring='accuracy',
                                                                           random_state=42, cv=self.CV_FOLD)
        data = []
        n_train_sizes = train_sizes.shape[0]
        n_folds = train_scores.shape[1]
        for i in range(n_train_sizes):
            for j in range(n_folds):
                data.append({'Train_sizes': train_sizes[i], 'Score': train_scores[i, j], 'Set': 'Training'})
                data.append({'Train_sizes': train_sizes[i], 'Score': validation_scores[i, j], 'Set': 'Validation'})
        plot_data = pd.DataFrame(data)
        plot_data.to_csv(os.path.join(self.path, f'Learning_Curve_Data_{len_var}.csv'), index=False)
        plt.figure(figsize=(12, 8))
        sns.lineplot(data=plot_data, x='Train_sizes', y='Score', hue='Set', palette='viridis', linewidth=3, marker='o')
        plt.ylim(0.49, 1)
        plt.xlabel("Training Samples", fontsize=26)
        plt.ylabel("Accuracy", fontsize=26)
        plt.xticks(fontsize=24)
        plt.yticks(fontsize=24)
        plt.legend(loc="lower right", fontsize=26)
        title = f"{len_var} Wavelengths"
        plt.title(title, fontsize=28)
        filename = f'Learning_Curve_{len_var}_titled.pdf'
        plt.savefig(os.path.join(self.path, filename))
        plt.show()
        plt.close()


    
    def _read_pivot(self, path):
        df = pd.read_csv(path)
        index_rows = df.columns[~df.columns.isin(self.wavelengths)].to_list()
        df.set_index(index_rows, inplace=True)
        X, y = self._random_input(df)
        return df, X, y
    
    def _compute_confusion_matrix(self, y_test, y_pred, len_var, cutoff=None): 
        
        y_pred_cutoff, _ = self._compute_y_pred_cutoff(y_test, y_pred, cutoff=cutoff)
        
        cm = confusion_matrix(y_test, y_pred_cutoff)
        
        # Create a dataframe from the confusion matrix cm
        cm_df = pd.DataFrame(cm, index=self.class_labels, columns=self.class_labels)
        
        # Save this dataframe to a csv file
        cm_df.to_csv(f'{self.path}/Confusion_Matrix_{len_var}.csv')
        
        # Plot the confusion matrix
        # fig = plt.figure(figsize=(8, 6))
        sns.heatmap(cm_df, linewidths=1, annot=True, fmt='d', cmap='viridis', center=True)
        
        plt.xlabel('Predicted', fontsize=18)
        plt.ylabel('Actual', fontsize=18)
        plt.title(f'{len_var} Wavelenghts', fontsize=18)
        
        # Put the correct labels
        # plt.xticks([0, 1], self.class_labels, ha='right')
        # plt.yticks([0, 1], self.class_labels, va='baseline')
        
        plt.savefig(f'{self.path}/Confusion_Matrix_{len_var}.pdf')
        plt.show()
        plt.close()
        
        return cm

    def _compute_y_pred_cutoff(self, y_test, y_pred,cutoff=None):
        # Find cutoff
        if cutoff is None:
            cutoff = self.find_cutoff(y_test, y_pred)

        y_pred_cutoff = (y_pred >= cutoff).astype(int).flatten()
        # print(y_pred_cutoff)
        
        return y_pred_cutoff, cutoff

    def _all_metrics(self, y_test, y_pred, cutoff=None):
        y_pred_cutoff, cutoff = self._compute_y_pred_cutoff(y_test, y_pred, cutoff=cutoff)
        
        accuracy  = accuracy_score(y_test, y_pred_cutoff)
        recall    = recall_score(y_test, y_pred_cutoff)
        precision = precision_score(y_test, y_pred_cutoff)
        f1        = f1_score(y_test, y_pred_cutoff)
        
        return accuracy,recall,precision,f1,cutoff

    def find_cutoff(self, y_test, y_pred):
        """find the best value of cutoff that maximizes the accuracy"""
        # cutoffs = np.arange(min(y_pred), max(y_pred), 0.05)
        # best_accuracy = 0
        # best_cutoff = 0
        # for c in cutoffs:
        #     y_pred_cutoff = [1 if i >= c else 0 for i in y_pred]
        #     accuracy = accuracy_score(y_test, y_pred_cutoff)
        #     if accuracy > best_accuracy:
        #         best_accuracy = accuracy
        #         best_cutoff = c
        
        fpr, tpr, thresholds, _ = self._compute_roc_auc(y_test, y_pred)
        
        # Find the best threshold
        best_cutoff = thresholds[np.argmax(tpr - fpr)]
        
        # # Plot a graph of the predicted values and the cutoff
        # plt.figure(figsize=(8, 6))
        # plt.plot(y_pred, 'ro', label='Predicted')
        # plt.axhline(y=best_cutoff, color='b', linestyle='-', label='Cutoff')
        # plt.xlabel('Sample')
        # plt.ylabel('Predicted Value')
        # plt.title('Predicted Values and Cutoff')
        # plt.legend()
        # plt.grid(True)
        # # plt.savefig(f'{self.path}/Predicted_Values_and_Cutoff.pdf')
        # plt.show()
        # plt.close()
        
        return best_cutoff
    
    def permutation_test(self, wavelengths=None, N=1000, save_file=True):           
        if wavelengths is None:
            var_combinations = range(self.P)
        else:
            wavelengths_str = [f"{w:.3f}" if isinstance(w, float) else w for w in wavelengths]
            var_combinations = [self.wavelengths.index(w) for w in wavelengths_str]
        n = len(var_combinations)
        
        df = pd.concat([self.df_original_2, self.df_test_2])

        # Add first right values        
        permutation_scores = []
        accuracy, recall, precision, f1, cutoff, roc, q2  = self._compute_accuracy(self.X_2, self.y_2, self.X_test_2, self.y_test_2, var_selected_i=var_combinations, all=True)
        print(f'{accuracy:.2f}, {recall:.2f}, {precision:.2f}, {f1:.2f}, {roc:.2f}, {q2:.2f}')
        permutation_scores.append([accuracy, recall, precision, f1, roc, q2, cutoff])
        
        # y = self.y_2.copy()
        # y_test = self.y_test_2.copy()
        X, y = self._random_input(df)
        
        for _ in tqdm(range(N)):            
            # Randomize the labels
            # np.random.shuffle(y) 
            
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=self.TEST, random_state=42)
            # print how much 1 and 0 in y_train and y_test
            # print(np.unique(y_train, return_counts=True), np.unique(y_test, return_counts=True))

            np.random.shuffle(y_train)
            np.random.shuffle(y_test)
            
            # Compute all metrics
            accuracy, recall, precision, f1, _, roc, q2  = self._compute_accuracy(X_train, y_train, X_test, y_test, var_selected_i=var_combinations, all=True,cutoff=cutoff)
            
            permutation_scores.append([accuracy, recall, precision, f1, roc, q2, cutoff])
            # print(f'{accuracy:.2f}, {recall:.2f}, {precision:.2f}, {f1:.2f}, {roc:.2f}, {q2:.2f}')
            
        
        if save_file:
            # Append accuracy to a file
            file_path = f'{self.path}/permutation_test_{n}.csv'
            print(f'Saving permutation test results to {file_path}')
            with open(file_path, 'w') as f:
                f.write('Accuracy,Recall,Precision,F1,ROC,Q2,cutoff\n')
                for metrics in permutation_scores:
                    for m in metrics[:-1]:
                        f.write(f'{m:.2f},')
                    f.write(f'{metrics[-1]:.2f}\n')