!pip install biopython scikit-learn xgboost numpy pandas joblib matplotlib seaborn

import os
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import traceback
from collections import Counter
from Bio import SeqIO
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score
from sklearn.metrics import classification_report
from xgboost import XGBClassifier


# ConfiguraÃ§Ãµes de visualizaÃ§Ã£o
plt.rcParams['figure.figsize'] = (10, 6)

# Lista aminoÃ¡cidos
AMINO_ACIDS = list('ACDEFGHIKLMNPQRSTVWY')

def read_fasta_to_df(fasta_path, id_col='id', seq_col='sequence'):
    """LÃª um arquivo FASTA e retorna DataFrame com colunas id e sequence."""
    records = list(SeqIO.parse(fasta_path, 'fasta'))
    data = [(rec.id, str(rec.seq).upper()) for rec in records]
    return pd.DataFrame(data, columns=[id_col, seq_col])


# Feature: ComposiÃ§Ã£o de AminoÃ�cidos
def aa_composition(seq):
    seq = seq.upper()
    L = len(seq)
    counts = Counter(seq)
    comps = [counts.get(aa, 0)/L if L>0 else 0.0 for aa in AMINO_ACIDS]
    return comps


def aa_composition_df(df, seq_col='sequence'):
    arr = [aa_composition(s) for s in df[seq_col].astype(str)]
    cols = [f'AAC_{aa}' for aa in AMINO_ACIDS]
    return pd.DataFrame(arr, columns=cols)


# Feature: Descritores comprimento e hidrofobia
def simple_seq_features(df, seq_col='sequence'):
    seqs = df[seq_col].astype(str)
    lengths = seqs.apply(len)
    hydrophobic = seqs.apply(lambda s: sum(s.count(ch) for ch in 'AILMFVWP')/len(s) if len(s)>0 else 0.0)
    return pd.DataFrame({'length': lengths, 'hydrophobic_frac': hydrophobic})


# Feature: K-mers
def seq_to_kmers(seq, k=3):
    """Gera lista de k-mers para uma sequÃªncia."""
    return [seq[i:i+k] for i in range(len(seq)-k+1)]


def kmer_corpus(df, seq_col='sequence', k=3):
    """Cria corpus de k-mers separados por espaÃ§o (TF-IDF)."""
    docs = []
    for s in df[seq_col].astype(str):
        kmers = seq_to_kmers(s, k=k)
        docs.append(' '.join(kmers))
    return docs


def exploratory_analysis(train_df, test_df, target_col='label', seq_col='sequence'):
    """Realiza anÃ¡lise exploratÃ³ria completa dos dados."""
    
    print("=" * 60)
    print(" ANÃ�LISE EXPLORATÃ“RIA DOS DADOS")
    print("=" * 60)
    
    print(f" DimensÃµes dos dados:")
    print(f"   Treino: {train_df.shape[0]} sequÃªncias, {train_df.shape[1]} colunas")
    print(f"   Teste:  {test_df.shape[0]} sequÃªncias, {test_df.shape[1]} colunas")
    
    print(f"\n DistribuiÃ§Ã£o de classes no treino:")
    class_dist = train_df[target_col].value_counts().sort_index()
    for cls, count in class_dist.items():
        print(f"   Classe {cls}: {count} sequÃªncias ({count/len(train_df)*100:.1f}%)")
    
    train_df['seq_length'] = train_df[seq_col].apply(len)
    test_df['seq_length'] = test_df[seq_col].apply(len)
    
    print(f"\n EstatÃ­sticas de comprimento das sequÃªncias:")
    print(f"   Treino - MÃ©dia: {train_df['seq_length'].mean():.1f}, "
          f"Min: {train_df['seq_length'].min()}, Max: {train_df['seq_length'].max()}")
    print(f"   Teste  - MÃ©dia: {test_df['seq_length'].mean():.1f}, "
          f"Min: {test_df['seq_length'].min()}, Max: {test_df['seq_length'].max()}")
    
    short_seqs_train = (train_df['seq_length'] < 10).sum()
    short_seqs_test = (test_df['seq_length'] < 10).sum()
    
    if short_seqs_train > 0 or short_seqs_test > 0:
        print(f"\n  SequÃªncias muito curtas (<10 aa):")
        print(f"   Treino: {short_seqs_train}, Teste: {short_seqs_test}")
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # DistribuiÃ§Ã£o de comprimentos - Treino
    axes[0, 0].hist(train_df['seq_length'], bins=50, alpha=0.7, color='blue', label='Treino')
    axes[0, 0].set_xlabel('Comprimento da SequÃªncia')
    axes[0, 0].set_ylabel('FrequÃªncia')
    axes[0, 0].set_title('DistribuiÃ§Ã£o de Comprimentos - Treino')
    axes[0, 0].legend()
    
    # DistribuiÃ§Ã£o de comprimentos - Teste
    axes[0, 1].hist(test_df['seq_length'], bins=50, alpha=0.7, color='red', label='Teste')
    axes[0, 1].set_xlabel('Comprimento da SequÃªncia')
    axes[0, 1].set_ylabel('FrequÃªncia')
    axes[0, 1].set_title('DistribuiÃ§Ã£o de Comprimentos - Teste')
    axes[0, 1].legend()
    
    # DistribuiÃ§Ã£o de classes
    class_dist.plot(kind='bar', ax=axes[1, 0], color=['skyblue', 'lightcoral'])
    axes[1, 0].set_xlabel('Classe')
    axes[1, 0].set_ylabel('NÃºmero de SequÃªncias')
    axes[1, 0].set_title('DistribuiÃ§Ã£o de Classes no Treino')
    
    # Boxplot de comprimentos por classe
    if target_col in train_df.columns:
        train_df.boxplot(column='seq_length', by=target_col, ax=axes[1, 1])
        axes[1, 1].set_title('Comprimento por Classe')
        axes[1, 1].set_ylabel('Comprimento')
    
    plt.tight_layout()
    plt.show()
    
    return train_df, test_df


def build_feature_matrix(df, k=3, use_tfidf_kmers=True, kmer_vectorizer=None, kmer_max_features=200):
    """Retorna X dataframe e o vetor TF-IDF."""
    parts = []

    # AAC
    aac_df = aa_composition_df(df)
    parts.append(aac_df)

    # Features simples
    simple_df = simple_seq_features(df)
    parts.append(simple_df)

    # K-mers
    if use_tfidf_kmers:
        corpus = kmer_corpus(df, k=k)
        if kmer_vectorizer is None:
            kmer_vectorizer = TfidfVectorizer(
                analyzer='word', 
                token_pattern=r"(?u)\b\w+\b", 
                max_features=kmer_max_features
            )
            X_kmer = kmer_vectorizer.fit_transform(corpus)
        else:
            X_kmer = kmer_vectorizer.transform(corpus)
        
        kmer_df = pd.DataFrame(
            X_kmer.toarray(), 
            columns=[f'kmer_{t}' for t in kmer_vectorizer.get_feature_names_out()]
        )
        parts.append(kmer_df)

    X = pd.concat(parts, axis=1)
    return X, kmer_vectorizer


def evaluate_models(train_df, target_col, test_df=None, id_col='id', seq_col='sequence'):
    """FunÃ§Ã£o principal com mÃºltiplas mÃ©tricas de avaliaÃ§Ã£o"""
    
    print("Construindo features de treino...")
    X_train, kmer_vectorizer = build_feature_matrix(
        train_df, k=3, use_tfidf_kmers=True, kmer_vectorizer=None, kmer_max_features=200
    )
    y_train = train_df[target_col].values
    
    print(f"DimensÃµes da matriz de features: {X_train.shape}")
    print(f"DistribuiÃ§Ã£o de classes: {Counter(y_train)}")
    
    # Modelos classificadores
    models = {
        'logreg': LogisticRegression(max_iter=200, class_weight='balanced', random_state=42),
        'rf': RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1),
        'svc': SVC(probability=True, class_weight='balanced', random_state=42)
    }
    models['xgb'] = XGBClassifier(
        eval_metric='logloss', 
        random_state=42, 
        n_jobs=-1
    )
    
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    results = []

    print("\n Iniciando validaÃ§Ã£o cruzada...")
    
    for name, model in models.items():
        print(f"\n Avaliando {name}...")
        
        try:
            pipeline = Pipeline([
                ('scaler', StandardScaler()),
                ('model', model)
            ])
            
            # AcurÃ¡cia
            ba_scores = cross_val_score(pipeline, X_train, y_train, cv=skf, scoring='balanced_accuracy')
            ba_mean, ba_std = ba_scores.mean(), ba_scores.std()
            
            # PrecisÃ£o
            precision_scores = cross_val_score(pipeline, X_train, y_train, cv=skf, scoring='precision_macro')
            precision_mean, precision_std = precision_scores.mean(), precision_scores.std()
            
            # Recall
            recall_scores = cross_val_score(pipeline, X_train, y_train, cv=skf, scoring='recall_macro')
            recall_mean, recall_std = recall_scores.mean(), recall_scores.std()
            
            # F1 Score
            f1_scores = cross_val_score(pipeline, X_train, y_train, cv=skf, scoring='f1_macro')
            f1_mean, f1_std = f1_scores.mean(), f1_scores.std()
            
            result_row = {
                'model': name,
                'balanced_accuracy_mean': ba_mean,
                'balanced_accuracy_std': ba_std,
                'precision_mean': precision_mean,  
                'recall_mean': recall_mean,        
                'f1_mean': f1_mean                
            }
            
            results.append(result_row)
            
            print(f"   ğŸ“Š Balanced Accuracy: {ba_mean:.4f} Â± {ba_std:.4f}")
            print(f"   ğŸ“Š Precision:         {precision_mean:.4f} Â± {precision_std:.4f}")
            print(f"   ğŸ“Š Recall:            {recall_mean:.4f} Â± {recall_std:.4f}")
            print(f"   ğŸ“Š F1-Score:          {f1_mean:.4f} Â± {f1_std:.4f}")
            
        except Exception as e:
            print(f"   â�Œ Falha ao avaliar {name}: {e}")
            traceback.print_exc()
            results.append({
                'model': name,
                'balanced_accuracy_mean': 0,
                'balanced_accuracy_std': 0,
                'precision_mean': 0,
                'recall_mean': 0,
                'f1_mean': 0
            })
    
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values('balanced_accuracy_mean', ascending=False)
    
    print("\n" + "="*70)
    print("ğŸ�† COMPARAÃ‡ÃƒO DETALHADA DOS MODELOS")
    print("="*70)
    
    display_columns = ['model', 'balanced_accuracy_mean', 'precision_mean', 'recall_mean', 'f1_mean']
    display_df = results_df[display_columns].copy()
    display_df.columns = ['Modelo', 'Balanced Acc', 'Precision', 'Recall', 'F1-Score']
    
    for col in display_df.columns[1:]:
        display_df[col] = display_df[col].apply(lambda x: f"{x:.4f}")
    
    print(display_df.to_string(index=False))
    
    print("\n GRÃ�FICO COMPARATIVO DAS MÃ‰TRICAS")
    plot_model_comparison(results_df)
    
    best_name = results_df.iloc[0]['model']
    best_model = models[best_name]
    
    print(f"\n Treinando modelo final: {best_name}")
    
    final_pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('model', best_model)
    ])
    
    final_pipeline.fit(X_train, y_train)
    
    y_pred = final_pipeline.predict(X_train)
    
    print(f"\n RELATÃ“RIO DE CLASSIFICAÃ‡ÃƒO NO TREINO COMPLETO:")
    print(classification_report(y_train, y_pred, target_names=['Classe 0', 'Classe 1']))
    
    preds = None
    if test_df is not None:
        print("ğŸ”® Gerando prediÃ§Ãµes para o conjunto de teste...")
        X_test, _ = build_feature_matrix(
            test_df, k=3, use_tfidf_kmers=True, 
            kmer_vectorizer=kmer_vectorizer,
            kmer_max_features=200
        )
        preds = final_pipeline.predict(X_test)
        
        print(f"ğŸ“Š EstatÃ­sticas das prediÃ§Ãµes no teste: {Counter(preds)}")
    
    return results_df, final_pipeline, kmer_vectorizer, preds


def plot_model_comparison(results_df):
    """GrÃ¡fico comparativo das mÃ©tricas dos modelos (SEM accuracy)"""
    models = results_df['model'].values
    metrics = ['balanced_accuracy_mean', 'precision_mean', 'recall_mean', 'f1_mean']
    metric_names = ['Balanced Acc', 'Precision', 'Recall', 'F1-Score']
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    x = np.arange(len(models))
    width = 0.2
    
    for i, (metric, name) in enumerate(zip(metrics, metric_names)):
        values = results_df[metric].values
        ax.bar(x + i*width, values, width, label=name, alpha=0.8)
    
    ax.set_xlabel('Modelos')
    ax.set_ylabel('Score')
    ax.set_title('ComparaÃ§Ã£o de Modelos por MÃ©tricas')
    ax.set_xticks(x + width*1.5)
    ax.set_xticklabels(models, rotation=45)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    for i, model in enumerate(models):
        for j, metric in enumerate(metrics):
            value = results_df[results_df['model'] == model][metric].values[0]
            ax.text(i + j*width, value + 0.01, f'{value:.3f}', 
                   ha='center', va='bottom', fontsize=8)
    
    plt.tight_layout()
    plt.show()


def save_results(results_df, pipeline, kmer_vectorizer, submission_df, model_name='best_pipeline'):
    """Salva todos os resultados e artefatos do modelo."""
    
    joblib.dump(pipeline, f'{model_name}.joblib')
    joblib.dump(kmer_vectorizer, f'{model_name}_vectorizer.joblib')
    
    results_df.to_csv('cross_validation_results.csv', index=False)
    
    submission_df.to_csv('submission.csv', index=False)
    
    with open('model_report.txt', 'w') as f:
        f.write("RELATÃ“RIO DO MODELO - DESAFIO BIOINFORMÃ�TICA\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Melhor modelo: {results_df.iloc[0]['model']}\n")
        f.write(f"Balanced Accuracy: {results_df.iloc[0]['balanced_accuracy_mean']:.4f}\n")
        f.write(f"Precision: {results_df.iloc[0]['precision_mean']:.4f}\n")
        f.write(f"Recall: {results_df.iloc[0]['recall_mean']:.4f}\n")
        f.write(f"F1-Score: {results_df.iloc[0]['f1_mean']:.4f}\n")
        f.write(f"SequÃªncias no teste: {len(submission_df)}\n")
        f.write(f"DistribuiÃ§Ã£o das prediÃ§Ãµes: {dict(Counter(submission_df['label']))}\n")
    
    print(f" Modelo salvo em: {model_name}.joblib")
    print(f" Vectorizer salvo em: {model_name}_vectorizer.joblib") 
    print(f" Resultados da validaÃ§Ã£o salvos em: cross_validation_results.csv")
    print(f" Scores detalhados salvos em: detailed_cv_scores.joblib")
    print(f" RelatÃ³rio salvo em: model_report.txt")
    print(f" Arquivo de submissÃ£o salvo em: submission.csv")


DATA_DIR = '/kaggle/input/desafio-cd-na-bioinformatica-inteligente'

NEG_FASTA = os.path.join(DATA_DIR, 'Neg_train_fasta.txt')
POS_FASTA = os.path.join(DATA_DIR, 'Pos_train_fasta.txt')
TEST_FASTA = os.path.join(DATA_DIR, 'seqs_test.txt')

# Verificar se os arquivos existem
if not all(os.path.exists(p) for p in [NEG_FASTA, POS_FASTA, TEST_FASTA]):
    print(f"â�Œ ERRO: Certifique-se de que os arquivos FASTA estÃ£o na pasta '{DATA_DIR}/'.")
    print("   Arquivos necessÃ¡rios:")
    print(f"   - {NEG_FASTA}")
    print(f"   - {POS_FASTA}") 
    print(f"   - {TEST_FASTA}")
else:
    print("âœ… Todos os arquivos encontrados!")
    
    print("\n Lendo arquivos FASTA...")
    neg_df = read_fasta_to_df(NEG_FASTA)
    pos_df = read_fasta_to_df(POS_FASTA)

    neg_df['label'] = 0
    pos_df['label'] = 1

    train_df = pd.concat([neg_df, pos_df], ignore_index=True)
    test_df = read_fasta_to_df(TEST_FASTA)
    
    train_df, test_df = exploratory_analysis(train_df, test_df)
    
    print("\n" + "="*60)
    print("TREINAMENTO E AVALIAÃ‡ÃƒO DE MODELOS")
    print("="*60)
    
    results_df, pipeline, kmer_vectorizer, preds = evaluate_models(
        train_df, target_col='label', test_df=test_df
    )
    
    if preds is not None:
        submission = pd.DataFrame({
            'ID': test_df['id'],
            'label': preds
        })
        
        # Salvar todos os resultados
        save_results(results_df, pipeline, kmer_vectorizer, submission)
        
        print("\n PIPELINE COMPLETADO COM SUCESSO!")
        print("Resumo da execuÃ§Ã£o:")
        best_model = results_df.iloc[0]
        print(f"   - Melhor modelo: {best_model['model']}")
        print(f"   - Balanced Accuracy: {best_model['balanced_accuracy_mean']:.4f}")
        print(f"   - Precision: {best_model['precision_mean']:.4f}")
        print(f"   - Recall: {best_model['recall_mean']:.4f}")
        print(f"   - F1-Score: {best_model['f1_mean']:.4f}")
        print(f"   - Qtd sequÃªncias classificadas: {len(preds)}")
    else:
        print("â�Œ Nenhuma prediÃ§Ã£o foi gerada.")




