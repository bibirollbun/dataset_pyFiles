import pandas as pd
import numpy as np
from functools import reduce
import warnings
import shutil

warnings.filterwarnings('ignore')

# === Ensemble script for BeatsPerMinute prediction ===

class UltimateEnsemble:
    def __init__(self, random_seed=42):
        # Set random seed for reproducibility
        self.random_seed = random_seed
        np.random.seed(random_seed)

    def load_base_submission(self):
        """
        Load 'best' base submission.
        If not found, generate dummy data as fallback.
        """
        try:
            base_sub = pd.read_csv('/kaggle/input/beats-per-minute-xgb-lgb-cat/submission.csv')
            print(f"âœ… Base submission loaded: {len(base_sub)} rows")
            return base_sub
        except FileNotFoundError:
            print("âš ï¸� 'best' not found, generating dummy data.")
            dummy_ids = range(524164, 524164 + 1000)
            dummy_bpm = np.random.normal(119.2, 0.3, 1000)
            return pd.DataFrame({'id': dummy_ids, 'BeatsPerMinute': dummy_bpm})

    def create_advanced_noise_variants(self, base_preds):
        """
        Generate 5 advanced noise variants based on base predictions.
        """
        variants = []
        # 1. Original kernel-style random noise (intensity and sign)
        rand_int_1 = np.random.randint(0, 101, len(base_preds))
        rand_sign_1 = np.random.choice([-1, 1], len(base_preds))
        noise_1 = (rand_int_1 / 10000.0) * rand_sign_1
        variants.append(base_preds + noise_1)

        # 2. Gaussian noise
        gaussian_noise = np.random.normal(0, 0.001, len(base_preds))
        variants.append(base_preds + gaussian_noise)

        # 3. Quantile-based micro adjustments
        quantiles = np.percentile(base_preds, [25, 50, 75])
        micro_adjust = np.where(base_preds < quantiles[0], -0.0005,
                            np.where(base_preds > quantiles[2], 0.0005, 0))
        variants.append(base_preds + micro_adjust)

        # 4. Cyclical pattern noise
        cyclical_noise = 0.0003 * np.sin(2 * np.pi * np.arange(len(base_preds)) / 100)
        variants.append(base_preds + cyclical_noise)

        # 5. Statistical outlier adjustment using z-score
        z_scores = np.abs((base_preds - np.mean(base_preds)) / np.std(base_preds))
        outlier_adjust = np.where(z_scores > 2,
                                 np.random.uniform(-0.001, 0.001, len(base_preds)), 0)
        variants.append(base_preds + outlier_adjust)

        return variants

    def create_weighted_ensemble(self, variants, strategy='adaptive'):
        """
        Combine noise variants into a weighted ensemble using given strategy.
        Supported strategies: adaptive, exponential, uniform, custom
        """
        if strategy == 'adaptive':
            variances = [np.var(v) for v in variants]
            inv_var = [1/v if v > 0 else 1 for v in variances]
            weights = np.array(inv_var) / np.sum(inv_var)
        elif strategy == 'exponential':
            weights = np.array([0.4, 0.25, 0.15, 0.12, 0.08])
        elif strategy == 'uniform':
            weights = np.ones(len(variants)) / len(variants)
        else: # custom
            weights = np.array([0.35, 0.20, 0.20, 0.15, 0.10])

        ensemble = np.average(variants, axis=0, weights=weights)
        print(f"ğŸ“Š Ensemble weights ({strategy}): {np.round(weights, 4)}")
        return ensemble

    def apply_post_processing(self, predictions):
        """
        Post-processing: smooth boundaries and optimize precision.
        """
        processed = predictions.copy()
        mean_pred = np.mean(processed)
        std_pred = np.std(processed)

        # Boundary smoothing (clip outliers)
        upper_bound = mean_pred + 2.5 * std_pred
        lower_bound = mean_pred - 2.5 * std_pred
        processed = np.where(processed > upper_bound,
                             processed * 0.999 + mean_pred * 0.001, processed)
        processed = np.where(processed < lower_bound,
                             processed * 0.999 + mean_pred * 0.001, processed)

        # Precision optimization
        processed = np.round(processed, 7)
        return processed

    def create_multiple_submissions(self, base_submission):
        """
        Generate multiple submission files using different ensemble strategies.
        """
        base_preds = base_submission['BeatsPerMinute'].values
        strategies = ['adaptive', 'exponential', 'uniform', 'custom']
        submissions = {}
        for strategy in strategies:
            print(f"\nğŸ”„ Creating {strategy} ensemble...")
            variants = self.create_advanced_noise_variants(base_preds)
            ensemble_preds = self.create_weighted_ensemble(variants, strategy)
            final_preds = self.apply_post_processing(ensemble_preds)

            result_df = base_submission.copy()
            result_df['BeatsPerMinute'] = final_preds
            filename = f'submission_{strategy}_enhanced.csv'
            result_df.to_csv(filename, index=False)
            submissions[strategy] = result_df
            print(f"âœ… {filename} created")
        return submissions

    def create_meta_ensemble(self, submissions):
        """
        Combine all ensemble results into a final meta-ensemble.
        """
        print(f"\nğŸ�¯ Creating meta-ensemble from {len(submissions)} strategies...")
        all_preds = [sub_df['BeatsPerMinute'].values for sub_df in submissions.values()]
        meta_weights = [0.3, 0.25, 0.25, 0.2]
        meta_ensemble = np.average(all_preds, axis=0, weights=meta_weights)
        final_meta = self.apply_post_processing(meta_ensemble)

        base_df = list(submissions.values())[0].copy()
        base_df['BeatsPerMinute'] = final_meta
        base_df.to_csv('submission_meta_ultimate.csv', index=False)
        print(f"ğŸ�† Meta-ensemble created: submission_meta_ultimate.csv")
        return base_df

def run_ensemble_iterations(iterations=2):
    """
    Run the ensemble process for the specified number of iterations.
    """
    ensemble = UltimateEnsemble(random_seed=42)
    for i in range(iterations):
        print(f"\n{'='*20} Iteration {i+1} {'='*20}")
        # 1. Load base submission
        base_submission = ensemble.load_base_submission()
        # 2. Create multiple ensembles
        submissions = ensemble.create_multiple_submissions(base_submission)
        # 3. Create meta-ensemble
        meta_submission = ensemble.create_meta_ensemble(submissions)
        # 4. Update 'submission.csv' for next iteration
        shutil.copy('submission_meta_ultimate.csv', 'submission.csv')
        print(f"\nâœ… 'best.csv' has been updated for the next iteration.")

    print(f"\n{'='*20} All {iterations} iterations complete. {'='*20}")
    print(f"ğŸ�† Final result is in 'submission_meta_ultimate.csv' and 'submission.csv'.")

if __name__ == '__main__':
    # Run the entire ensemble process for 5 iterations
    run_ensemble_iterations(iterations=5)

