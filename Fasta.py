import pandas as pd
donor1 = pd.read_csv("/home/students/s.nag/Desktop/alphabetaepi/epitcr/epitcr_data/beta_pairs_10x_donor1.csv")
donor2 = pd.read_csv("/home/students/s.nag/Desktop/alphabetaepi/epitcr/epitcr_data/beta_pairs_10x_donor2.csv")
donor3 = pd.read_csv("/home/students/s.nag/Desktop/alphabetaepi/epitcr/epitcr_data/beta_pairs_10x_donor3.csv")
donor4 = pd.read_csv("/home/students/s.nag/Desktop/alphabetaepi/epitcr/epitcr_data/beta_pairs_10x_donor4.csv")
frames = [donor1, donor2, donor3, donor4]
dataset = pd.concat(frames)
dataset = dataset.loc[dataset["Binder"]==0]
dataset = dataset.drop_duplicates()
len(dataset)
dataset_1 = dataset.sample(149571)
dataset_1.to_csv("10x_1_neg_sampled.csv")
#dataset1 = pd.read_csv("/home/students/s.nag/Desktop/alphabetaepi/epitcr/epitcr_data/10x_neg_1_1_sampled.csv")
negative_beta_tcr = dataset_1['CDR3b']
with open('10x_1_cdr3b_neg_sam.fasta', 'w') as f:
    for k, v in negative_beta_tcr.items():
        f.write(f'>{k}\n{v}\n')
negative_beta_epi = dataset_1['Epitope']
with open('10x_1_epi_neg_sam.fasta', 'w') as f:
    for k, v in negative_beta_epi.items():
        f.write(f'>{k}\n{v}\n')
