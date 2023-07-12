import pandas as pd
dataset1 = pd.read_csv("/home/students/s.nag/Desktop/alphabetaepi/epitcr/epitcr_data/10x_neg_1_1.csv")
negative_beta_tcr = df_negative['CDR3b']
with open('10x_1_cdr3b_neg.fasta', 'w') as f:
    for k, v in negative_beta_tcr.items():
        f.write(f'>{k}\n{v}\n')
negative_beta_epi = df_negative['Epitope']
with open('10x_1_epi_neg.fasta', 'w') as f:
    for k, v in negative_beta_epi.items():
        f.write(f'>{k}\n{v}\n')
