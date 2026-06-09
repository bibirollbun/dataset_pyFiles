!git clone https://github.com/priorlabs/tabpfn-extensions.git

!wget https://huggingface.co/Prior-Labs/TabPFN-v2-reg/resolve/main/tabpfn-v2-regressor.ckpt
!wget https://huggingface.co/Prior-Labs/TabPFN-v2-reg/resolve/main/tabpfn-v2-regressor-2noar4o2.ckpt
!wget https://huggingface.co/Prior-Labs/TabPFN-v2-reg/resolve/main/tabpfn-v2-regressor-5wof9ojf.ckpt
!wget https://huggingface.co/Prior-Labs/TabPFN-v2-reg/resolve/main/tabpfn-v2-regressor-09gpqh39.ckpt
!wget https://huggingface.co/Prior-Labs/TabPFN-v2-reg/resolve/main/tabpfn-v2-regressor-wyl4o83o.ckpt

!pip install tabpfn --no-deps --target=/kaggle/working/


# ## First include this Notebook as Input into your offline TabPFN notebook.
# ## Then uncomment and use the below two code cells to install and use TabPFN:

# ## update install of tabpfn
# !mkdir -p /root/.cache/tabpfn/
# !cp /kaggle/usr/lib/tabpfn_and_extensions/tabpfn-v2-regressor.ckpt /root/.cache/tabpfn/tabpfn-v2-regressor.ckpt

# ## update install of tabpfn_extensions
# !cp -r /kaggle/usr/lib/tabpfn_and_extensions/tabpfn-extensions/src/tabpfn_extensions/ tabpfn_extensions
# !mkdir -p tabpfn_extensions/hpo/hpo_models/
# !cp /kaggle/usr/lib/tabpfn_and_extensions/tabpfn-v2-regressor.ckpt tabpfn_extensions/hpo/hpo_models/tabpfn-v2-regressor.ckpt
# !cp /kaggle/usr/lib/tabpfn_and_extensions/tabpfn-v2-regressor-09gpqh39.ckpt tabpfn_extensions/hpo/hpo_models/tabpfn-v2-regressor-09gpqh39.ckpt
# !cp /kaggle/usr/lib/tabpfn_and_extensions/tabpfn-v2-regressor-2noar4o2.ckpt tabpfn_extensions/hpo/hpo_models/tabpfn-v2-regressor-2noar4o2.ckpt
# !cp /kaggle/usr/lib/tabpfn_and_extensions/tabpfn-v2-regressor-wyl4o83o.ckpt tabpfn_extensions/hpo/hpo_models/tabpfn-v2-regressor-wyl4o83o.ckpt
# !cp /kaggle/usr/lib/tabpfn_and_extensions/tabpfn-v2-regressor-5wof9ojf.ckpt tabpfn_extensions/hpo/hpo_models/tabpfn-v2-regressor-5wof9ojf.ckpt

# # ## OPTIONAL: override with your own modified source file(s)!
# # !cp /kaggle/input/tabpfn-override/pfn_phe.py tabpfn_extensions/post_hoc_ensembles/pfn_phe.py


# ## Example usage:

# from tabpfn import TabPFNRegressor
# from tabpfn_extensions.post_hoc_ensembles.sklearn_interface import AutoTabPFNRegressor

# ## excerpt from K-fold loop: (see TabPFN Starter for glue code: https://www.kaggle.com/code/maiernator/tabfn-starter )
#     tabpfn_kap = AutoTabPFNRegressor(random_state=42, device="cuda", max_time=60 * 3)
    
#     # Get indices of columns with a specific dtype
#     cat_cols = x_train.select_dtypes(include='object').columns
#     cat_cols = list(x_train.columns.get_indexer(cat_cols))
#     print(cat_cols)
    
#     tabpfn_kap.fit(x_train, y_train, categorical_feature_indices=cat_cols)
    
#     # INFER OOF
#     print('cv predict...')
#     oof_tabpfn_kap[test_index] = tabpfn_kap.predict(x_valid)
#     print('test predict...')
#     # INFER TEST
#     pred_tabpfn_kap += tabpfn_kap.predict(x_test)


