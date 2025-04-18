"""
Created on Fri Sep 20 12:46:30 2024

@author: petersen.jonas
"""

import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Update model metrics.')
    parser.add_argument('--models', nargs='+', help='List of models to update (e.g., A B C)', required=True)
    args = parser.parse_args()
    import src.generate_stacked_residuals.generate_abs_mean_gradient_training_data
    if 'exp' in args.models:
        import src.generate_stacked_residuals.generate_exponential_smoothing_stacked_residuals
    if 'lgbm' in args.models: 
        import src.generate_stacked_residuals.generate_lgbm_basic_stacked_residuals
        import src.generate_stacked_residuals.generate_lgbm_darts_stacked_residuals
        import src.generate_stacked_residuals.generate_lgbm_w_sklearn_stacked_residuals
        import src.generate_stacked_residuals.generate_lgbm_feature_darts_stacked_residuals
    if 'mean' in args.models:
        import src.generate_stacked_residuals.generate_mean_profile_stacked_residuals
    if 'naive' in args.models:
        import src.generate_stacked_residuals.generate_naive_darts_stacked_residuals
        import src.generate_stacked_residuals.generate_naive_seasonal_darts_stacked_residuals
    if 'ssm' in args.models:
        import src.generate_stacked_residuals.generate_SSM_stacked_residuals
    if 'tide' in args.models:
        import src.generate_stacked_residuals.generate_tide_darts_stacked_residuals
        import src.generate_stacked_residuals.generate_tide_feature_darts_stacked_residuals
    if 'xgboost' in args.models:
        import src.generate_stacked_residuals.generate_xgboost_darts_stacked_residuals
    if 'sorcerer' in args.models:
        import src.generate_stacked_residuals.generate_sorcerer_stacked_residuals
    if 'tlp' in args.models:
        import src.generate_stacked_residuals.generate_tlp_regression_model_stacked_residuals
    if 'deepar' in args.models:
        import src.generate_stacked_residuals.generate_deepar_stacked_residuals
    if 'climatological' in args.models:
        import src.generate_stacked_residuals.generate_climatological_darts_stacked_residuals
    if 'tft' in args.models:
        import src.generate_stacked_residuals.generate_tft_darts_stacked_residuals
    if 'all' in args.models:
        import src.generate_stacked_residuals.generate_exponential_smoothing_stacked_residuals
        import src.generate_stacked_residuals.generate_lgbm_basic_stacked_residuals
        import src.generate_stacked_residuals.generate_lgbm_darts_stacked_residuals
        import src.generate_stacked_residuals.generate_lgbm_w_sklearn_stacked_residuals
        import src.generate_stacked_residuals.generate_mean_profile_stacked_residuals
        import src.generate_stacked_residuals.generate_naive_darts_stacked_residuals
        import src.generate_stacked_residuals.generate_SSM_stacked_residuals
        import src.generate_stacked_residuals.generate_tide_darts_stacked_residuals
        import src.generate_stacked_residuals.generate_xgboost_darts_stacked_residuals
        import src.generate_stacked_residuals.generate_sorcerer_stacked_residuals
        import src.generate_stacked_residuals.generate_tlp_regression_model_stacked_residuals
        import src.generate_stacked_residuals.generate_deepar_stacked_residuals
        import src.generate_stacked_residuals.generate_climatological_darts_stacked_residuals
        import src.generate_stacked_residuals.generate_lgbm_feature_darts_stacked_residuals
        import src.generate_stacked_residuals.generate_naive_seasonal_darts_stacked_residuals
        import src.generate_stacked_residuals.generate_tft_darts_stacked_residuals
        import src.generate_stacked_residuals.generate_tide_feature_darts_stacked_residuals