


exec(open(f"/kaggle/input/jigsaw2025-public-baseline-v2/myimports.py", "r").read())


%%writefile bertV1_1.py

PrintColor(
    f"\n---> BERTV1_1 CV = 0.83689320 inferencing\n"
)

ip_path   = f"/kaggle/input/jigsaw2025publicmodelsv1/BERTV1_1"
MAX_LEN   = 512
n_classes = 2

def tokenize(batch):
    return tokenizer(
        batch["text"],
        padding     = "longest",
        truncation  = True,
        max_length  = MAX_LEN
    )

def format_input(row):
    return f"""
    Comment: {row['body']}
    Rule: {row['rule']}
    Subreddit: {row['subreddit']}
    Positive Examples: {row['positive_example_1']} || {row['positive_example_2']}
    Negative Examples: {row['negative_example_1']} || {row['negative_example_2']}
    """

Xtest  = pd.read_csv(f"/kaggle/input/jigsaw-agile-community-rules/test.csv", index_col = ["row_id"])
sub_fl = pd.read_csv(f"/kaggle/input/jigsaw-agile-community-rules/sample_submission.csv", index_col = ["row_id"])
Xtest['text']  = Xtest.apply(format_input, axis=1)


tokenizer = AutoTokenizer.from_pretrained(ip_path)
model = AutoModelForSequenceClassification.from_pretrained(
    ip_path,
    num_labels = n_classes
)

training_args = TrainingArguments(
    output_dir                  = "Model",
    do_train                    = False,
    per_device_eval_batch_size  = 64,
    report_to                   = "none",
    dataloader_pin_memory       = False,
    logging_strategy            = "no",
    fp16                        = True,
    torch_compile               = True,
    seed                        = 42,
    data_seed                   = 42,
)

trainer = Trainer(
    model               = model,
    args                = training_args,
    processing_class    = tokenizer,
    data_collator       = DataCollatorWithPadding(tokenizer=tokenizer),
)

Xt         = Dataset.from_pandas( Xtest[[ "text" ]]).map(tokenize, batched=True)
test_preds = trainer.predict(Xt)
test_preds = torch.nn.functional.softmax(torch.tensor(test_preds.predictions), dim=1).numpy()[:,1]

sub_fl["rule_violation"] = test_preds
sub_fl.to_csv(f"submission.csv", index = True)



%%time 

exec(open( f"bertV1_1.py", "r").read())
!head submission.csv
print()

shutil.rmtree("Model")
!ls 
os.rename("submission.csv", "submission_1.csv")

print()


%%time 

exec(open( f"/kaggle/input/jigsaw2025-public-baseline-v3/ettinV1_1.py", "r").read())
!head submission.csv
print()

shutil.rmtree("Model")
!ls 
os.rename("submission.csv", "submission_2.csv")

print()


%%time 

exec(open( f"/kaggle/input/jigsaw2025-public-baseline-v2/bertV1_2.py", "r").read())
!head submission.csv
print()

shutil.rmtree("BERTV1_2")
!ls 
os.rename("submission.csv", "submission_3.csv")

print()


%%writefile debertasmall_public.py

PrintColor(
    f"\n---> DEBERTA Public CV = 0.87163996 inferencing\n"
)

ip_path      = "/kaggle/input/jigsaw-deberta-small-cv-0-702"
n_classes    = 1
SEED         = 42

os.environ["PYTHONHASHSEED"] = str(SEED)
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"

test = pd.read_csv("/kaggle/input/jigsaw-agile-community-rules/test.csv")

def make_prompt(row):
    return f"""[RULE]: {row['rule']}
[SUBREDDIT]: {row['subreddit']}

[COMMENT]: {row['body']}

[POSITIVE EXAMPLES]:
1. {row['positive_example_1']}
2. {row['positive_example_2']}

[NEGATIVE EXAMPLES]:
1. {row['negative_example_1']}
2. {row['negative_example_2']}

[QUESTION]: Does the comment violate the rule?
[ANSWER]:"""

features_cols = ['text','label']
columns       = ['input_ids', 'attention_mask', 'label']
test['text']  = test.apply(make_prompt,axis=1)
Xt            = Dataset.from_pandas(test[['text']]).map(tokenize, batched=True)

tokenizer     = AutoTokenizer.from_pretrained(ip_path)
model         = AutoModelForSequenceClassification.from_pretrained(
    ip_path,
    num_labels = n_classes,
)

training_args = TrainingArguments(
    output_dir                  = "Model",
    do_train                    = False,
    per_device_eval_batch_size  = 64,
    report_to                   = "none",
    dataloader_pin_memory       = False,
    logging_strategy            = "no",
    fp16                        = True,
    torch_compile               = True,
    seed                        = 42,
    data_seed                   = 42,
)

trainer = Trainer(
    model               = model,
    args                = training_args,
    processing_class    = tokenizer,
    data_collator       = DataCollatorWithPadding(tokenizer=tokenizer),
)

predictions = trainer.predict(Xt)
test_preds  = torch.sigmoid(torch.tensor(predictions.predictions)).numpy().flatten()

sub_fl["rule_violation"] = test_preds
sub_fl.to_csv(f"submission.csv", index = True)


%%time 

exec(open( f"debertasmall_public.py", "r").read())
!head submission.csv
print()

shutil.rmtree("Model")
!ls 
os.rename("submission.csv", "submission_4.csv")


%%time 

from sklearn.preprocessing import MinMaxScaler

mdl_preds = []
for myfile in [
    "submission_1.csv", 
    "submission_2.csv", 
    "submission_3.csv", 
    "submission_4.csv",
]:
    
    print(f"{myfile} processing now")
    scl = MinMaxScaler()
    df  = pd.read_csv(f"{myfile}", index_col = "row_id")[["rule_violation"]]
    df  = scl.fit_transform(df).flatten()
    mdl_preds.append(df)

sub_fl = pd.read_csv(
    f"/kaggle/input/jigsaw-agile-community-rules/sample_submission.csv",
    index_col = "row_id",
)

sub_fl["rule_violation"] = \
np.average(
    np.stack(mdl_preds, axis=1), 
    axis=1, 
    weights = [0.20, 0.25, 0.50, 0.75]
)

sub_fl.to_csv("submission.csv", index = True)
print()
!ls

print()
!head submission.csv

