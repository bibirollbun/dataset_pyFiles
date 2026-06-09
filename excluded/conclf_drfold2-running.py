!cp -r /kaggle/input/drfold2/pytorch/default/3/DRfold2 ./
# !cd DRfold2/Arena && make Arena


# !cd DRfold2 && python3 DRfold_infer.py test/seq.fasta test/output/


import os
import pandas as pd
import subprocess



def run_inference_on_fasta(fasta_file_path, output_path):
    subprocess.run(
        ["python3", "-u", "DRfold2/DRfold_infer.py", fasta_file_path, output_path],
        check=True
    )
    os.system(f"rm -r {output_path}/rets_dir")


# Assume that the atoms are listed in order with respect to the sequence
def convert_pdb_to_c1_coords(pdb_path):
    x = []
    y = []
    z = []

    with open(pdb_path, "r") as pdb_file:
        for line in pdb_file:

            desc = line.strip().split()
            if desc[0] == "ATOM" and desc[2] == "C1'":
                x.append(float(desc[6]))
                y.append(float(desc[7]))
                z.append(float(desc[8]))
    
    return {"x": x, "y": y, "z": z}

def create_fasta_file(sequence, output_path, file_name="seq.fasta",seq_id=""):
    
    os.makedirs(output_path, exist_ok=True)
    
    with open(os.path.join(output_path, file_name), "w") as file:
        file.write(f">{seq_id}\n")
        file.write(sequence + "\n")



test_df = pd.read_csv("/kaggle/input/stanford-rna-3d-folding/test_sequences.csv")
train_df = pd.read_csv("/kaggle/input/stanford-rna-3d-folding/train_sequences.csv")


def create_submission(df, start_id=None):

    submission = pd.DataFrame({"ID":[], "resname":[], "resid":[],
                               "x_1":[], "y_1":[], "z_1":[], 
                               "x_2":[], "y_2":[], "z_2":[],
                               "x_3":[], "y_3":[], "z_3":[]
                              })
    
    if start_id is not None:
        start = df["target_id"].tolist().index(start_id)
        df = df.iloc[(start + 1):]
    
    for idx, row in df.iterrows():
        
        sequence = row.sequence 
        target_id = row.target_id

        print("-----------------")
        print(f"Folding sequence: {target_id}\nSequence length: {len(sequence)}")
        print("-----------------")

        try:
            create_fasta_file(sequence, f"./{target_id}", seq_id=target_id)
            run_inference_on_fasta(os.path.join(f"./{target_id}", "seq.fasta"), f"./{target_id}")
            coords = convert_pdb_to_c1_coords(os.path.join(f"./{target_id}", "relax", "model_1.pdb"))
        except Exception as e:
            print(e)
            # bad prediction
            coords = {
                "x": [0.] * len(sequence),
                "y": [0.] * len(sequence),
                "z": [0.] * len(sequence)
            }
        
        
        rows = []            # list of dicts
        for resid in range(1, len(sequence)+1):
            rows.append({
                "ID"    : f"{target_id}_{resid}",
                "resname": sequence[resid-1],
                "resid" : resid,
                "x_1"   : coords["x"][resid-1],
                "y_1"   : coords["y"][resid-1],
                "z_1"   : coords["z"][resid-1],
                # add x_2 … z_3 when you have them (see §4)
        })
        submission = pd.concat([submission, pd.DataFrame.from_records(rows)], ignore_index=True)
    submission = submission.replace(np.nan, 0.0)
    submission.to_csv("submission.csv", index=False)


create_submission(test_df) # actual test

