%%bash

pip install --upgrade 'scikit-learn<1.6.0'


from sklearn import metrics


def normalized_gini_coefficient(y_true, y_score):
    roc_auc_score = metrics.roc_auc_score(y_true, y_score)
    return 2 * roc_auc_score - 1



import pandas as pd


train_df = pd.read_csv(
    "/kaggle/input/porto-seguro-safe-driver-prediction/train.csv",
    index_col="id",
    na_values=[-1],
)


train_df.info()


import matplotlib.pyplot as plt
import numpy as np



train_features_df = train_df.drop("target", axis="columns")
train_target = train_df.loc[:, "target"]


# some of the categorical features have missing values
(
    train_features_df.filter(like="ps_ind_", axis="columns")
                     .info()
)


_ = (
    train_features_df.filter(like="ps_ind_", axis="columns")
                     .hist(density=True)
)
plt.tight_layout()


_ = (
    train_features_df.loc[:, ["ps_ind_01", "ps_ind_03", "ps_ind_14", "ps_ind_15"]]
                     .hist(density=True)
)
plt.tight_layout()


(
    train_features_df.loc[:, "ps_ind_01"]
                     .value_counts()
                     .sort_index()
)


# possibly the calendar month in which policy started?
(
    train_features_df.loc[:, "ps_ind_03"]
                     .value_counts()
                     .sort_index()
)


(
    train_features_df.loc[:, "ps_ind_14"]
                     .value_counts()
                     .sort_index()
)


(
    train_features_df.loc[:, "ps_ind_15"]
                     .value_counts()
                     .sort_index()
)


_ = (
    train_features_df.filter(like="ps_reg")
                     .hist()
)


# ps_reg_01 and ps_reg_02 could be ordinal features?
(
    train_features_df.loc[:, "ps_reg_01"]
                     .value_counts()
                     .sort_index()
)


# ps_reg_01 and ps_reg_02 could be ordinal features?
(
    train_features_df.loc[:, "ps_reg_02"]
                     .value_counts()
                     .sort_index()
)


# only ps_reg_03 has missing values
(
    train_features_df.filter(like="ps_reg_", axis="columns")
                     .info()
)


(
    train_features_df.loc[:, "ps_reg_03"]
                     .describe()
)


# some of the categorical features have missing values
(
    train_features_df.filter(like="ps_car_", axis="columns")
                     .info()
)


(
    train_features_df.filter(like="ps_car_", axis="columns")
                     .hist()
)
plt.tight_layout()


# this
(
    train_features_df.loc[:, "ps_car_11"]
                     .value_counts()
                     .sort_index()
)


# this ordinal feature appears to have been transformed using sqrt!
(
    train_features_df.loc[:, "ps_car_15"]
                     .pow(2)
                     .value_counts()
                     .sort_index()
)


_ = (
    train_features_df.loc[:, "ps_car_15"]
                     .pow(2)
                     .hist()
)


# no missing values
(
    train_features_df.filter(like="ps_calc_", axis="columns")
                     .info()
)


(
    train_features_df.filter(like="ps_calc_", axis="columns")
                     .hist()
)
plt.tight_layout()


# looks like the ps_calc_* features take small number of unique values
(
    train_features_df.loc[:, "ps_calc_03"]
                     .value_counts()
                     .sort_index()
)


train_target.value_counts(normalize=True)


import numpy as np
from sklearn import compose, impute, pipeline, preprocessing


ps_ind_features_pipeline = pipeline.make_pipeline(
    compose.make_column_transformer(
        (
            pipeline.make_pipeline(
                impute.SimpleImputer(
                    strategy="most_frequent",
                    add_indicator=False,
                )
            ),
            compose.make_column_selector(
                pattern="_bin$",
            )
        ),
        (
            pipeline.make_pipeline(
                impute.SimpleImputer(
                    strategy="most_frequent",
                    add_indicator=False,
                ),
                preprocessing.OneHotEncoder(
                    handle_unknown="infrequent_if_exist",
                    min_frequency=None,
                    sparse_output=False,
                )
            ),
            compose.make_column_selector(
                pattern="_cat$",
            )
        ),
        n_jobs=-1,
        remainder=pipeline.make_pipeline(
            impute.SimpleImputer(
                strategy="most_frequent",
                add_indicator=False,
            ),
            preprocessing.OrdinalEncoder(
                handle_unknown="use_encoded_value",
                unknown_value=-1,
                min_frequency=None
            )
        ),
        verbose_feature_names_out=False,
    )
).set_output(transform="pandas")

ps_ind_column_selector = compose.make_column_selector(
    pattern="ps_ind_"
)


ps_ind_features_pipeline


_selected_features = ps_ind_column_selector(train_features_df)
transformed_ps_ind_features_df = ps_ind_features_pipeline.fit_transform(
    train_features_df.loc[:, _selected_features]
)


transformed_ps_ind_features_df.info()


ps_reg_features_pipeline = pipeline.make_pipeline(
    compose.make_column_transformer(
        (
            pipeline.make_pipeline(
                impute.SimpleImputer(
                    strategy="median",
                ),
                preprocessing.MinMaxScaler(
                    feature_range=(-1, 1)
                )
            ),
            [
                "ps_reg_01",
                "ps_reg_02",
            ]
        ),
        (
            pipeline.make_pipeline(
                impute.SimpleImputer(
                    strategy="median",
                ),
                preprocessing.RobustScaler()
            ),
            [
                "ps_reg_03"
            ]
        ),
        n_jobs=-1,
        remainder="drop",
        verbose_feature_names_out=False,
    )
).set_output(transform="pandas")

ps_reg_column_selector = compose.make_column_selector(
    pattern="ps_reg_"
)


ps_reg_features_pipeline


_selected_features = ps_reg_column_selector(train_features_df)
transformed_ps_reg_features_df = ps_reg_features_pipeline.fit_transform(
    train_features_df.loc[:, _selected_features]
)


transformed_ps_reg_features_df.info()


ps_car_features_pipeline = pipeline.make_pipeline(
    compose.make_column_transformer(
        (
            pipeline.make_pipeline(
                impute.SimpleImputer(
                    strategy="constant",
                    fill_value=-1,
                    add_indicator=False
                ),
                preprocessing.OneHotEncoder(
                    handle_unknown="infrequent_if_exist",
                    min_frequency=None,
                    sparse_output=False
                )
            ),
            compose.make_column_selector(
                pattern="_cat$"
            )
        ),
        (
            pipeline.make_pipeline(
                impute.SimpleImputer(
                    strategy="most_frequent",
                    add_indicator=False
                ),
                preprocessing.OrdinalEncoder(
                    handle_unknown="use_encoded_value",
                    unknown_value=-1,
                    min_frequency=None
                ),
            ),
            [
                "ps_car_11",
            ]
        ),
        (
            pipeline.make_pipeline(
                impute.SimpleImputer(
                    strategy="median",
                ),
                preprocessing.RobustScaler()
            ),
            [
                "ps_car_12",
                "ps_car_13",
                "ps_car_14",
            ]
        ),
        (
            pipeline.make_pipeline(
                preprocessing.FunctionTransformer(
                    func=lambda df: df.pow(2),
                    inverse_func=lambda df: df.pow(0.5)
                ),
                impute.SimpleImputer(
                    strategy="most_frequent",
                    add_indicator=False
                ),
                preprocessing.OrdinalEncoder(
                    handle_unknown="use_encoded_value",
                    unknown_value=-1,
                    min_frequency=None
                ),
            ),
            [
                "ps_car_15",
            ]
        ),
        n_jobs=-1,
        remainder="drop",
        verbose_feature_names_out=False
    )
).set_output(transform="pandas")

ps_car_column_selector = compose.make_column_selector(
    pattern="ps_car_"
)


ps_car_features_pipeline


_selected_features = ps_car_column_selector(train_features_df)
transformed_ps_car_features_df = ps_car_features_pipeline.fit_transform(
    train_features_df.loc[:, _selected_features]
)


transformed_ps_car_features_df.info()


ps_calc_features_pipeline = pipeline.make_pipeline(
    compose.make_column_transformer(
        (
            pipeline.make_pipeline(
                impute.SimpleImputer(
                    strategy="median",
                    add_indicator=False,
                ),
                preprocessing.MinMaxScaler(
                    feature_range=(-1, 1)
                )
            ),
            [
                "ps_calc_01",
                "ps_calc_02",
                "ps_calc_03",
            ]
        ),
        (
            pipeline.make_pipeline(
                impute.SimpleImputer(
                    strategy="most_frequent",
                    add_indicator=False
                )
            ),
            compose.make_column_selector(
                pattern="_bin$",
            )
        ),
        n_jobs=-1,
        remainder=pipeline.make_pipeline(
            impute.SimpleImputer(
                strategy="most_frequent",
                add_indicator=False,
            ),
            preprocessing.OrdinalEncoder(
                handle_unknown="use_encoded_value",
                unknown_value=-1,
                min_frequency=None,
            )
        ),
        verbose_feature_names_out=False,
    )
).set_output(transform="pandas")

ps_calc_column_selector = compose.make_column_selector(
    pattern="ps_calc_"
)


ps_calc_features_pipeline


_selected_features = ps_calc_column_selector(train_features_df)
transformed_ps_calc_features_df = ps_calc_features_pipeline.fit_transform(
    train_features_df.loc[:, _selected_features]
)


transformed_ps_calc_features_df.info()


features_preprocessing_pipeline = pipeline.make_pipeline(
    compose.make_column_transformer(
        (
            ps_ind_features_pipeline,
            ps_ind_column_selector
        ),
        (
            ps_reg_features_pipeline,
            ps_reg_column_selector
        ),
        (
            ps_car_features_pipeline,
            ps_car_column_selector
        ),
        (
            ps_calc_features_pipeline,
            ps_calc_column_selector
        ),
        n_jobs=-1,
        remainder="drop"
    )
)


features_preprocessing_pipeline


prepared_features = (
    features_preprocessing_pipeline.fit_transform(train_features_df)
)


prepared_features.shape


from sklearn import dummy, linear_model, ensemble


dummy_pipeline = pipeline.make_pipeline(
    features_preprocessing_pipeline,
    dummy.DummyClassifier()
)

_ = dummy_pipeline.fit(train_features_df, train_target)


linear_model_pipeline = pipeline.make_pipeline(
    features_preprocessing_pipeline,
    linear_model.SGDClassifier(
        class_weight="balanced",
        loss="log_loss",
    )
)

_ = linear_model_pipeline.fit(train_features_df, train_target)


random_forest_pipeline = pipeline.make_pipeline(
    features_preprocessing_pipeline,
    ensemble.RandomForestClassifier(
        class_weight="balanced",  # important when working with imbalanced classes!
        n_jobs=-1,
    )
)

_ = random_forest_pipeline.fit(train_features_df, train_target)


from sklearn import metrics, model_selection


dummy_train_predictions = dummy_pipeline.predict(
    train_features_df
)

linear_model_train_predictions = linear_model_pipeline.predict(
    train_features_df
)

random_forest_train_predictions = random_forest_pipeline.predict(
    train_features_df
)


dummy_cv_predictions = model_selection.cross_val_predict(
    dummy_pipeline,
    train_features_df,
    train_target,
    cv=3,
    method="predict",
    n_jobs=-1,
    verbose=1,
)


linear_model_cv_predictions = model_selection.cross_val_predict(
    linear_model_pipeline,
    train_features_df,
    train_target,
    cv=3,
    method="predict",
    n_jobs=-1,
    verbose=1,
)


random_forest_cv_predictions = model_selection.cross_val_predict(
    random_forest_pipeline,
    train_features_df,
    train_target,
    cv=3,
    method="predict",
    n_jobs=-1,
    verbose=1,
)


metrics.accuracy_score?


_train_accuracy = metrics.accuracy_score(
    train_target,
    dummy_train_predictions
)

_cv_accuracy = metrics.accuracy_score(
    train_target,
    dummy_cv_predictions
)

print(f"Dummy Pipeline Train Accuracy {_train_accuracy}")
print(f"Dummy Pipeline CV Accuracy {_cv_accuracy}")


_train_accuracy = metrics.accuracy_score(
    train_target,
    linear_model_train_predictions
)

_cv_accuracy = metrics.accuracy_score(
    train_target,
    linear_model_cv_predictions
)

print(f"Linear Model Pipeline Train Accuracy {_train_accuracy}")
print(f"Linear Model Pipeline CV Accuracy {_cv_accuracy}")


_train_accuracy = metrics.accuracy_score(
    train_target,
    random_forest_train_predictions
)

_cv_accuracy = metrics.accuracy_score(
    train_target,
    random_forest_cv_predictions
)

print(f"Random Forest Pipeline Train Accuracy {_train_accuracy}")
print(f"Random Forest Pipeline CV Accuracy {_cv_accuracy}")


metrics.balanced_accuracy_score?


_train_accuracy = metrics.balanced_accuracy_score(
    train_target,
    linear_model_train_predictions
)

_cv_accuracy = metrics.balanced_accuracy_score(
    train_target,
    linear_model_cv_predictions
)

print(f"Linear Model Pipeline Train Balanced Accuracy {_train_accuracy}")
print(f"Linear Model Pipeline CV Balanced Accuracy {_cv_accuracy}")


_train_accuracy = metrics.balanced_accuracy_score(
    train_target,
    random_forest_train_predictions
)

_cv_accuracy = metrics.balanced_accuracy_score(
    train_target,
    random_forest_cv_predictions
)

print(f"Random Forest Pipeline Train Balanced Accuracy {_train_accuracy}")
print(f"Random Forest Pipeline CV Balanced Accuracy {_cv_accuracy}")


metrics.classification_report?


_report = metrics.classification_report(
    train_target,
    linear_model_cv_predictions
)

print(_report)


_report = metrics.classification_report(
    train_target,
    random_forest_cv_predictions
)

print(_report)


_matrix = metrics.confusion_matrix(
    train_target,
    linear_model_cv_predictions
)

print(_matrix)


_matrix = metrics.confusion_matrix(
    train_target,
    random_forest_cv_predictions
)

print(_matrix)


metrics.make_scorer?


normalized_gini_coefficient_scoring = metrics.make_scorer(
    normalized_gini_coefficient,
    greater_is_better=True,
    response_method="predict_proba",
)


_train_predict_probas = linear_model_pipeline.predict_proba(train_features_df)

normalized_gini_coefficient(
    train_target,
    _train_predict_probas[:, 1]
)


linear_model_cv_scores = model_selection.cross_val_score(
    linear_model_pipeline,
    train_features_df,
    train_target,
    cv=3,
    n_jobs=-1,
    scoring=normalized_gini_coefficient_scoring,
    verbose=True,
)


linear_model_cv_scores


linear_model_cv_scores.mean()


_train_predict_probas = random_forest_pipeline.predict_proba(train_features_df)

normalized_gini_coefficient(
    train_target,
    _train_predict_probas[:, 1]
)


random_forest_cv_scores = model_selection.cross_val_score(
    random_forest_pipeline,
    train_features_df,
    train_target,
    cv=3,
    n_jobs=-1,
    scoring=normalized_gini_coefficient_scoring,
    verbose=True,
)


random_forest_cv_scores


random_forest_cv_scores.mean()


model_selection.FixedThresholdClassifier?


base_classifier = linear_model.SGDClassifier(
    class_weight="balanced",
    loss="log_loss"
)

fixed_threshold_classifier_pipeline = pipeline.make_pipeline(
    features_preprocessing_pipeline,
    model_selection.FixedThresholdClassifier(
        base_classifier,
        threshold=0.10,
        response_method="predict_proba",
    )
)


model_selection.TunedThresholdClassifierCV?


base_classifier = linear_model.SGDClassifier(
    class_weight="balanced",
    loss="log_loss"
)

tuned_threshold_classifier_pipeline = pipeline.make_pipeline(
    features_preprocessing_pipeline,
    model_selection.TunedThresholdClassifierCV(
        base_classifier,
        thresholds=100,
        response_method="predict_proba",
        cv=3,
        n_jobs=-1,
        refit=True
    )
)


_ = tuned_threshold_classifier_pipeline.fit(train_features_df, train_target)


# check out the best threshold!
tuned_threshold_classifier_pipeline[-1].best_threshold_


test_features_df = pd.read_csv(
    "/kaggle/input/porto-seguro-safe-driver-prediction/test.csv",
    index_col="id",
    na_values=[-1],
)

test_predict_probas = tuned_threshold_classifier_pipeline.predict_proba(
    test_features_df
)


_ = (
    pd.read_csv(
        "/kaggle/input/porto-seguro-safe-driver-prediction/sample_submission.csv",
        index_col="id",
    ).assign(
        target=test_predict_probas[:, 1]
    ).to_csv("submission.csv", index=True)
)


%%bash

cat submission.csv | head -n 5




