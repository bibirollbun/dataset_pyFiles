# Load modules
import sys
sys.path.append('/kaggle/input/my-ariel2025-library')
import kaggle_support as kgs
import ariel_model
import ariel_gp
import copy


train_data = kgs.load_all_train_data()
test_data = kgs.load_all_test_data()
fast_mode = False
if fast_mode and not kgs.is_submission:
    train_data = train_data[:20]
if not kgs.is_submission:
    # Use train data as test data if not submitting
    test_data = copy.deepcopy(train_data)


model_visualization = ariel_gp.PredictionModel() # the core Bayesian model; this is not the model we will use for submission    
model_visualization.plot_final = True # make diagnostic plots
model_visualization.train(train_data) # does nothing for this particular model, but mandatory
model_visualization.infer(train_data[0:1]);


model = ariel_model.baseline_model()

# If you want to try any changes to the model, make them here!

model.train(train_data)


inferred_data = model.infer(test_data)
if not kgs.is_submission:
    print(kgs.score_metric(inferred_data, test_data))
df = kgs.make_submission_dataframe(inferred_data)
kgs.write_submission_csv(df)

