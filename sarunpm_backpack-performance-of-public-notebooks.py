import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
import matplotlib.pyplot as plt 
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')  # ignore notifications
from scipy.optimize import curve_fit


# Lorentzian function
def lorentzian(x, amp, mean, gamma):
    return amp * (gamma ** 2) / ((x - mean) ** 2 + gamma ** 2)


# Determination of the Full Width at Half Maximum (FWHM) using Lorentzian function
def lFWHM(x, y, ax):
    # Initial guess for the parameters: amplitude, mean, gamma
    initial_guess = [max(y), np.mean(x), np.std(x)]

    # Fit the Lorentzian function to the data
    popt, _ = curve_fit(lorentzian, x, y, p0=initial_guess)
    amp, mean, gamma = popt

    # Calculate FWHM
    fwhm = -2 * gamma

    # Calculate the half-maximum
    half_max = amp / 2

    # Solve for x values where y = half_max
    fwhm_x1 = mean - gamma
    fwhm_x2 = mean + gamma

    # Print FWHM
    #print(f"FWHM: {fwhm:.2f}")
    #print(f"FWHM points: {fwhm_x1:.2f}, {fwhm_x2:.2f}")

    # Generate fitted Lorentzian curve
    x_fit = np.linspace(min(x), max(x), 1000)
    y_fit = lorentzian(x_fit, amp, mean, gamma)

    # Plot the KDE data (assuming it's a KDE from seaborn)
    ax.plot(x, y, label='KDE', color='blue')

    # Plot the fitted Lorentzian
    ax.plot(x_fit, y_fit, 'r--', label='Fitted Lorentzian')

    # Plot FWHM lines
    ax.axvline(x=fwhm_x1, color='g', linestyle='--', label='FWHM Points')
    ax.axvline(x=fwhm_x2, color='g', linestyle='--')

    # Annotate FWHM value
    ax.text(mean, half_max, f'FWHM ------------> {abs(fwhm):.5f}', ha='center')

    # Add legend
    ax.legend()


# Function to Plot the obtained results
def lorentzplot(df, columns, nrows=3, ncols=3):
    # Only take the first nrows * ncols columns if there are more than that
    num_cols = nrows * ncols
    columns = columns[:num_cols]
    
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 5, nrows * 4))

    for i, cname in enumerate(columns):
        if nrows == 1 and ncols == 1:
            ax = axes
        elif nrows == 1 or ncols == 1:
            ax = axes[i]
        else:
            ax = axes[i // ncols, i % ncols]
        
        # Create the KDE plot
        kde_plot = sns.histplot(data=df, x=cname, kde=True, palette='viridis', ax=ax)

        # Extract KDE data
        kde_data = kde_plot.lines[0].get_data()
        kde_x, kde_y = kde_data

        # Plot the Lorentzian fit and FWHM
        lFWHM(kde_x, kde_y, ax)
        
        ax.set_xlabel(cname)
        ax.set_ylabel('Frequency')
        ax.set_title(f'Distribution of {cname}')
        
    # Hide any unused subplots
    for j in range(num_cols, nrows * ncols):
        if nrows == 1 or ncols == 1:
            fig.delaxes(axes[j])
        else:
            fig.delaxes(axes[j // ncols, j % ncols])

    plt.tight_layout()
    plt.show()


#Loading Submission files of TOP Public Performers
df1 = pd.read_csv('/kaggle/input/first-place-single-model-lb-38-81/submission_v1.csv')
df2 = pd.read_csv('/kaggle/input/ps-s5e2-dividing-attention/submission.csv')
df3 = pd.read_csv('/kaggle/input/use-original-data-for-cv-boost-and-lb-boost/submission.csv')
df4 = pd.read_csv('/kaggle/input/backpack-nn-lgbm-cat-xgb-ydf-hgb/submission.csv')
df5 = pd.read_csv('/kaggle/input/5-2-gbm-cat-mix/submission2000.csv')
df6 = pd.read_csv('/kaggle/input/feature-engineering-with-rapids-lb-38-847/submission_v1.csv')
df7 = pd.read_csv('/kaggle/input/s5e2-sae-te-xgb/submission.csv')
df8 = pd.read_csv('/kaggle/input/backpack-price-prediction-eda-ensemble/submission.csv')
df9 = pd.read_csv('/kaggle/input/backpack-submission/submission_TE_XGB2_38.65005041648449_CV.csv')


public_top_df = pd.DataFrame({'id' : df1.id, 
                           'ChrisD_PB-38.81xxx_PR-38.62984'       : df1['Price'],                           
                           'Invicible_PB-38.83225_PR-38.64749'    : df2[' Price'],
                           'Kumaran_K_PB-38.83576_PR-38.64391'    : df3['Price'],
                           'Makhail_38.83654_PR-38.63456'         : df4['Price'],
                           'GyoGyoCat_PB-38.84017_PR-38.64778'    : df5['Price'],
                           'ChrisD_PB-38.84716_PR-38.65245'       : df6['Price'],
                           'Masaya_PB-38.84771_PR-38.63456'       : df7['Price'],
                           'Oleksii_PB-38.86125_PR-38.66913'      : df8['Price'],
                           'Sarun_PB-38.85428_PR-38.64721'        : df9['Price']
                          })


dfm1 = pd.read_csv('/kaggle/input/backpack-submission/submission_1XGB_38.88616352610392_cv.csv')
dfm2 = pd.read_csv('/kaggle/input/backpack-submission/submission_XGBv2_38.844222739676006_CVB.csv')
dfm3 = pd.read_csv('/kaggle/input/backpack-submission/submission_TE_XGB_38.651239392501964_CV.csv')
dfm4 = pd.read_csv('/kaggle/input/backpack-submission/submission_TE_CATBST_38.65080119861763_CV.csv')
dfm5 = pd.read_csv('/kaggle/input/backpack-submission/submission_TE_XGB2_38.64946945063536_CV.csv')
dfm6 = pd.read_csv('/kaggle/input/backpack-submission/submission_TE_XGB2_38.65005041648449_CV.csv')


 my_top_df = pd.DataFrame({'id'      : dfm1.id, 
                           'XGB1_PB-39.11980_PR-38.91013'    : dfm1['Price'],                           
                           'XGB2_PB-39.14623_PR-38.95295'    : dfm2['Price'],
                           'TE_XGB1_PB-38.84486_PR-38.65252' : dfm3['Price'],
                           'TE_CAT1_PB-38.84434_PR-38.65250' : dfm4['Price'],
                           'TE_XGB2_PB-38.84268_PR-38.64895' : dfm5['Price'],
                           'TE_XGB2_PB-38.84258_PR-38.64721' : dfm6['Price'],
                          })


lorentzplot(public_top_df, public_top_df.columns[1:])


lorentzplot(my_top_df, my_top_df.columns[1:],3,2)


submission = pd.DataFrame({'id' : df1.id, 
                          'Price': public_top_df.iloc[:, 1:].mean(axis=1)
                          })
submission.to_csv('submission.csv',index=False)


submission.head()


submission_my_top = pd.DataFrame({'id' : df1.id, 
                          'Price': my_top_df.iloc[:, 1:].mean(axis=1)
                          })
submission_my_top.to_csv('my_submission.csv', index=False)


submission_my_top.head()


lorentzplot(df1, df1.columns[1:],1,1)


lorentzplot(submission_my_top, submission_my_top.columns[1:],1,1)

