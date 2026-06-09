import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')


train_labels = pd.read_csv('/kaggle/input/matrix-lmn/train_labels_perfect.csv')
train_labels.head(3)


residue_pair = ['GG', 'GC', 'GA', 'CG', 'CC', 'AG', 'AA', 'GU', 'UG', 'UC', 'CU', 'AC', 'CA', 'UU', 'AU', 'UA']

for i in range(len(residue_pair)):
    df = train_labels[train_labels.res2==residue_pair[i]]
    plt.figure(figsize=(10, 5))
    sns.histplot(df['l_1'], bins=50, kde=True)
    plt.xlabel("Bond distance")
    plt.ylabel("Count")
    plt.title(f"Bond Distance Distribution for Bound {residue_pair[i]} Pair")
    plt.show()



residue_triplet = ['GGG', 'GGU', 'GUG', 'UGC', 'GCU', 'CUC', 'UCA', 'CAG', 'AGU', 'GUA', 'UAC', 'ACG', 'CGA', 'GAG', 'AGA', 'AGG', 'GGA', 'GAA', 'AAC', 'ACC', 'CCG', 'CGC', 'GCA', 'CAC', 'CCC', 'GGC', 'GCG', 'UGG', 'CUA', 'UAG', 'AGC', 'GCC', 'CCA', 'ACU', 'CAA', 'AAA', 'AAG', 'CAU', 'GAC', 'CUG', 'UGA', 'GAU', 'AUC', 'GUC', 'UCU', 'UAU', 'AUA', 'UAA', 'CUU', 'UUC', 'UCG', 'CGG', 'GUU', 'UUG', 'UGU', 'UCC', 'CCU', 'AUG', 'CGU', 'UUA', 'UUU', 'AUU', 'ACA', 'AAU']

for i in range(len(residue_triplet)):
    df = train_labels[train_labels.res3==residue_triplet[i]]
    plt.figure(figsize=(10, 5))
    sns.histplot(df['m_1'], bins=50, kde=True)
    plt.xlabel("Bond angle [degree]")
    plt.ylabel("Count")
    plt.title(f"Bond Angle Distribution for {residue_triplet[i]}")
    plt.show()



residue_quartet = ['GGGU', 'GGUG', 'GUGC', 'UGCU', 'GCUC', 'CUCA', 'UCAG', 'CAGU', 'AGUA', 'GUAC', 'UACG', 'ACGA', 'CGAG', 'GAGA', 'AGAG', 'GAGG', 'AGGA', 'GGAA', 'GAAC', 'AACC', 'ACCG', 'CCGC', 'CGCA', 'GCAC', 'CACC', 'ACCC', 'GGCG', 'GCGC', 'GCAG', 'AGUG', 'GUGG', 'UGGG', 'GGGC', 'GGCU', 'GCUA', 'CUAG', 'UAGC', 'AGCG', 'CGCC', 'GCCA', 'CCAC', 'CACU', 'ACUC', 'UCAA', 'CAAA', 'AAAA', 'AAAG', 'AAGG', 'AGGC', 'GGCC', 'GCCC', 'CCCA', 'CCAU', 'GGGA', 'GGAC', 'GACU', 'ACUG', 'CUGA', 'UGAC', 'GACG', 'CGAU', 'GAUC', 'AUCA', 'UCAC', 'CACG', 'ACGC', 'AGUC', 'GUCU', 'UCUA', 'CUAU', 'GGAU', 'GAUA', 'AUAA', 'UAAC', 'AACU', 'ACUU', 'CUUC', 'UUCG', 'UCGG', 'CGGU', 'GGUU', 'GUUG', 'UUGU', 'UGUC', 'GUCC', 'UCCC', 'GCGA', 'CGAC', 'GACC', 'CCCU', 'CCUG', 'UGAU', 'GAUG', 'AUGA', 'UGAG', 'GCCG', 'CCGA', 'CGAA', 'GAAA', 'AAAC', 'CCGU', 'CGCU', 'GCUU', 'CUUG', 'UUGC', 'UGCG', 'GCGU', 'CGUC', 'CUCG', 'UCGU', 'CGUA', 'GUAA', 'UAAG', 'AAGA', 'GAGU', 'GUCA', 'ACCA', 'AAGC', 'AGCC', 'CCCG', 'UUAC', 'UACC', 'CCAA', 'CAAG', 'AAGU', 'AGUU', 'GUUU', 'UUUG', 'UUGA', 'AGGU', 'GGUA', 'CGUG', 'GUGU', 'UGUA', 'GUAG', 'AGCU', 'UCAU', 'CAUU', 'AUUA', 'UUAG', 'CUCC', 'UCCG', 'GAGC', 'GGCA', 'CAGA', 'AGAU', 'AUCU', 'UCUG', 'GCCU', 'CUGG', 'GGAG', 'CUCU', 'UCUC', 'CUGC', 'UGCC', 'GCAA', 'GGUC', 'CAGC', 'GCUG', 'ACGG', 'UACA', 'ACAG', 'CAGG', 'GGGG', 'UCUU', 'CGGA', 'UCCA', 'UGUG', 'GUGA', 'UGAA', 'AACA', 'ACAC', 'CGGC', 'GCGG', 'UGGA', 'UACU', 'AGAA', 'CUGU', 'UGUU', 'GUUC', 'UUCC', 'CCAG', 'AGAC', 'GACA', 'ACCU', 'CCUC', 'UCCU', 'UCGC', 'CGCG', 'CCUA', 'CUAA', 'GUUA', 'UUAU', 'UAUG', 'AUGG', 'UGGC', 'UUCA', 'CAAC', 'UUGG', 'GAAG', 'ACGU', 'CGUU', 'UUUC', 'CCUU', 'CGGG', 'ACAU', 'AUUG', 'UGCA', 'ACAA', 'CCCC', 'CUUU', 'UUUU', 'AGGG', 'CAUC', 'AUCG', 'UGGU', 'UAGU', 'CAUG', 'AUGC', 'UAGG', 'GUCG', 'UCGA', 'CCGG', 'AUAU', 'UAUC', 'ACUA', 'GUAU', 'UAUU', 'CAUA', 'AUAC', 'AUAG', 'AGCA', 'AUGU', 'UAAA', 'AAAU', 'AAUC', 'UAUA', 'GAUU', 'AUUC', 'AUCC', 'AACG', 'CAAU', 'AAUG', 'UUUA', 'UUAA', 'UAAU', 'CACA', 'UAGA', 'GCAU', 'UUCU', 'GAAU', 'CUUA', 'AAUA', 'CUAC', 'AAUU', 'AUUU']

for i in range(len(residue_quartet)):
    df = train_labels[train_labels.res4==residue_quartet[i]]
    plt.figure(figsize=(10, 5))
    sns.histplot(df['n_1'], bins=50, kde=True)
    plt.xlabel("Torsion angle [degree]")
    plt.ylabel("Count")
    plt.title(f"Torsion Angle Distribution for {residue_quartet[i]}")
    plt.show()





