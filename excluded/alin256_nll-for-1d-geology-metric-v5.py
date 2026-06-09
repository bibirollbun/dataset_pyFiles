# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt
import pandas.api.types
import numpy as np


class ParticipantVisibleError(Exception):
    # If you want an error message to be shown to participants, you must raise the error as a ParticipantVisibleError
    # All other errors will only be shown to the competition host. This helps prevent unintentional leakage of solution data.
    pass

def score(solution: pd.DataFrame, submission: pd.DataFrame, row_id_column_name: str) -> float:
    """
    NLL metric for the Geology Forecast Challenge
    Detailed desctiption is the competition file
    www.kaggle.com/competitions/geology-forecast-challenge-open/overview/evaluation
    """
    # TODO: You likely want to delete the row ID column, which Kaggle's system uses to align
    # the solution and submission before passing these dataframes to score().
    del solution[row_id_column_name]
    del submission[row_id_column_name]

    # TODO: adapt or remove this check depending on what data types make sense for your metric
    for col in submission.columns:
        if not pandas.api.types.is_numeric_dtype(submission[col]):
            raise ParticipantVisibleError(f'Submission column {col} must be a number')

    # TODO: add additional checks appropriate for your metric. Common things to check include:
    # Non-negative inputs, the wrong number of columns in the submission, values that should be restricted to a range, etc.
    # The more errors you tag as participant visible, the easier it will be for participants to debug their submissions.

    # calculate your metric here. For the template, we'll just calculate a simple mean absolute error over all non-id columns.
    # By default, the metric doesn't have information about what columns to expect. You probably want to add the names of other
    # columns as arguments to score, like row_id_column_name, if you won't be processing all columns the same way.

    NEGATIVE_PART = -299
    LARGEST_CHUNK = 600
    SMALLEST_CHUNK = 350
    TOTAL_REALIZATIONS = 10
    INFLATION_SIGMA = 600

    # exponent_1 = 0.9879100831042902
    # mult_1 = 0.7557775492195064
    #
    # exponent_2 = 1.9758201662085804
    # mult_2 = 0.5711997039042433

    sigma_2 = np.ones((LARGEST_CHUNK + NEGATIVE_PART - 1))
    from_ranges = [1, 61, 245]
    to_ranges_excl = [61, 245, 301]
    log_slopes = [1.0406028049510443, 0.0, 7.835345062351012]
    log_offsets = [-6.430669850650689, -2.1617411566043896, -45.24876794412965]

    for growth_mode in range(len(from_ranges)):
        for i in range(from_ranges[growth_mode], to_ranges_excl[growth_mode]):
            sigma_2[i - 1] = np.exp(np.log(i) * log_slopes[growth_mode] + log_offsets[growth_mode])

    # trying to inflate sigma not to get errors
    sigma_2 *= INFLATION_SIGMA
    # plt.plot(sigma_2)
    # print(sigma_2)
    # plt.savefig('sigma_2.png', bbox_inches='tight')
    # plt.show()

    # sigma_2 = np.array(list(mult_2*np.power(i, exponent_2) for i in range(1, LARGEST_CHUNK + NEGATIVE_PART)))

    # Compute the inverse of the diagonal covariance matrix (element-wise inverse)
    cov_matrix_inv_diag = 1. / sigma_2  # Inverse of the diagonal elements

    # formula for multivariate gaussian
    # https://en.wikipedia.org/wiki/Multivariate_normal_distribution
    # https://en.wikipedia.org/wiki/Covariance_matrix#Covariance_matrix_as_a_parameter_of_a_distribution
    # formula for mixture density
    # https://en.wikipedia.org/wiki/Mixture_model#Multivariate_Gaussian_mixture_model

    num_rows = solution.shape[0]

    # TODO add probabilities
    # now we use all probs equal to:
    p = 1. / TOTAL_REALIZATIONS
    ps = np.full((num_rows, TOTAL_REALIZATIONS), p)

    log_p = -np.log(TOTAL_REALIZATIONS)
    inner_product_matrix = np.zeros((num_rows, TOTAL_REALIZATIONS))

    # ps = np.full((num_rows, TOTAL_REALIZATIONS), p)

    # exp_misfit = np.zeros((num_rows, TOTAL_REALIZATIONS))

    num_columns = LARGEST_CHUNK + NEGATIVE_PART - 1

    # collecting solution and submission in numpy arrays
    full_submission = np.zeros((num_rows, TOTAL_REALIZATIONS, num_columns))
    full_solution = np.zeros((num_rows, TOTAL_REALIZATIONS, num_columns))

    # Iterating through column names
    for k in range(TOTAL_REALIZATIONS):
        # compute misfits in individual columns
        # Create an array filled with zeros
        misfit = np.zeros((num_rows, num_columns))
        for i in range(num_columns):
            if k == 0:
                column_name = str(i + 1)
            else:
                column_name = f"r_{k}_pos_{i + 1}"
            # this needs to be different cov matrix
            misfit[:, i] = solution[column_name].values - submission[column_name].values
            full_submission[:, k, i] = submission[column_name].values
            full_solution[:, k, i] = solution[column_name].values
        misfit_scaled = misfit * cov_matrix_inv_diag
        inner_product = np.sum(misfit_scaled * misfit, axis=1)
        # exp_misfit_cur = np.exp(inner_product)
        # exp_misfit[:, k] = exp_misfit_cur
        inner_product_matrix[:, k] = inner_product #+ log_p

    # Find normalization constant
    # max_vals = 100*np.ones_like(inner_product_matrix[:,0]) # np.zeros_like(inner_product_matrix[:,0]) # np.max(inner_product_matrix, axis=1)
    max_vals = np.max(inner_product_matrix, axis=1)
    # add and subtract max_vals to avoid numerical instability
    product = np.exp(inner_product_matrix + log_p - max_vals[:, None])
    log_sum_exp = max_vals + np.log(np.sum(product, axis=1))

    nll = -log_sum_exp
    computed_score = nll.mean()

    # Check for infinite scores
    if np.isinf(computed_score):
        raise ParticipantVisibleError(f"Your score is {computed_score}, which means there is room for improvement.")

    return computed_score

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory


# TODO complete the code to run the metric
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


        
# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session

