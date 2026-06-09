import pandas as pd
import numpy as np # Útil para operações numéricas
import matplotlib.pyplot as plt
import seaborn as sns


train = pd.read_csv('/kaggle/input/playground-series-s5e4/train.csv')
test = pd.read_csv('/kaggle/input/playground-series-s5e4/test.csv')


display(train.head())
display(train.describe())
display(train.info())

display(test.head())
display(test.describe())
display(test.info())


# Obter os nomes das colunas
print("\nNomes das Colunas:")
print(train.columns.tolist())

# --- 3. Verificação de Dados Faltantes (Missing Values) ---
print("\n--- Dados Faltantes (train) ---")
print("Contagem de valores nulos por coluna:")
missing_values = train.isnull().sum()
print(missing_values[missing_values > 0]) # Mostra apenas colunas com valores faltantes

print("\n--- Dados Faltantes (test) ---")
print("Contagem de valores nulos por coluna:")
missing_values = test.isnull().sum()
print(missing_values[missing_values > 0]) # Mostra apenas colunas com valores faltantes


train['Episode_Title'].unique()


import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted, check_array, check_X_y

class SingleStrategyImputer(BaseEstimator, TransformerMixin):
    """
    Imputa valores faltantes em todas as colunas apropriadas usando uma única
    estratégia global ('mean', 'median', 'most_frequent') ou um valor constante.

    A estratégia é aplicada às colunas com base no seu tipo de dado:
    - 'mean', 'median': Aplicado apenas a colunas numéricas.
    - 'most_frequent': Aplicado a colunas numéricas ou categóricas/objeto.
    - Valor Constante: Aplicado a todas as colunas com valores faltantes.

    Colunas onde a estratégia não é aplicável (ex: 'mean' em coluna de string)
    ou que não possuem valores faltantes no conjunto de treino não serão imputadas.

    Parâmetros
    ----------
    strategy : str or int or float
        A estratégia de imputação a ser usada. Pode ser uma das strings
        'mean', 'median', 'most_frequent', ou um valor constante
        (int, float, str) a ser usado para preenchimento.
    """
    def __init__(self, strategy):
        # Validação básica da estratégia
        allowed_strategies = ['mean', 'median', 'most_frequent']
        if not isinstance(strategy, (str, int, float)):
             raise ValueError("A estratégia deve ser 'mean', 'median', 'most_frequent', ou um valor constante (str, int, float).")
        if isinstance(strategy, str) and strategy not in allowed_strategies:
             # Permite strings que não são palavras-chave como valores constantes
             pass
             # print(f"Aviso: Estratégia '{strategy}' não é uma palavra-chave padrão. Será tratada como valor constante.")

        self.strategy = strategy
        self.imputation_values_ = {} # Armazena os valores calculados no fit por coluna
        self.fitted_columns_ = []   # Armazena as colunas que realmente foram ajustadas

    def fit(self, X, y=None):
        """
        Calcula os valores de imputação para cada coluna com NaNs no DataFrame X,
        baseado na estratégia global e no tipo de dado da coluna.

        Parâmetros
        ----------
        X : pd.DataFrame
            O DataFrame de entrada para ajustar o imputer.
        y : None
            Ignorado. Presente por compatibilidade com o Scikit-learn.

        Retorna
        -------
        self : object
            Retorna a instância ajustada do imputer.
        """
        # Validação básica de entrada
        if not isinstance(X, pd.DataFrame):
             raise ValueError("X deve ser um pandas DataFrame.")

        self.imputation_values_ = {} # Reinicia a cada fit
        self.fitted_columns_ = []   # Reinicia a cada fit

        is_constant_strategy = not isinstance(self.strategy, str) or self.strategy not in ['mean', 'median', 'most_frequent']
        constant_value = self.strategy if is_constant_strategy else None

        for column in X.columns:
            # Pula colunas sem valores faltantes no conjunto de treino
            if not X[column].isnull().any():
                continue

            imputation_val = None # Valor a ser usado para esta coluna

            # --- Aplica Estratégia Constante (se definida) ---
            if is_constant_strategy:
                imputation_val = constant_value
                # Nenhuma verificação de tipo necessária para constante, aplica a todas

            # --- Aplica Estratégias por Palavra-Chave ---
            else:
                if pd.api.types.is_numeric_dtype(X[column]):
                    if self.strategy == 'mean':
                        imputation_val = X[column].mean()
                    elif self.strategy == 'median':
                        imputation_val = X[column].median()
                    elif self.strategy == 'most_frequent':
                        valid_modes = X[column].dropna().mode()
                        if not valid_modes.empty:
                            imputation_val = valid_modes[0]
                        else: # Coluna numérica só com NaNs
                            imputation_val = 0 # Ou outro fallback numérico
                            print(f"Aviso: Coluna numérica '{column}' continha apenas NaNs. Imputando com {imputation_val}.")

                elif pd.api.types.is_object_dtype(X[column]) or pd.api.types.is_categorical_dtype(X[column]):
                    if self.strategy == 'most_frequent':
                        valid_modes = X[column].dropna().mode()
                        if not valid_modes.empty:
                            imputation_val = valid_modes[0]
                        else: # Coluna objeto/categórica só com NaNs
                             imputation_val = "Missing" # Ou outro fallback string
                             print(f"Aviso: Coluna categórica/objeto '{column}' continha apenas NaNs. Imputando com '{imputation_val}'.")
                    # else: mean/median não são válidos aqui, imputation_val continua None

                else:
                     print(f"Aviso: Tipo de dado não suportado para estratégias padrão na coluna '{column}': {X[column].dtype}. Pulando imputação para esta coluna (a menos que seja estratégia constante).")
                     # imputation_val continua None

            # --- Armazena o valor calculado (se válido) ---
            if imputation_val is not None:
                 # Verifica se o valor calculado resultou em NaN (ex: mean de coluna só com NaN)
                 if pd.isna(imputation_val):
                      is_numeric = pd.api.types.is_numeric_dtype(X[column])
                      fallback = 0 if is_numeric else "Unknown"
                      print(f"Aviso: O valor calculado para imputar a coluna '{column}' com a estratégia '{self.strategy}' resultou em NaN. "
                            f"Usando fallback: {fallback}.")
                      imputation_val = fallback

                 self.imputation_values_[column] = imputation_val
                 self.fitted_columns_.append(column) # Adiciona à lista de colunas ajustadas

            # Se imputation_val for None (ex: pediu 'mean' para coluna de string),
            # a coluna não será adicionada a fitted_columns_ e não será transformada.

        if not self.fitted_columns_:
             print("Aviso: Nenhuma coluna encontrada com valores faltantes que pudesse ser imputada com a estratégia fornecida durante o fit.")

        self.n_features_in_ = X.shape[1]
        self.feature_names_in_ = np.array(X.columns, dtype=object)
        self._is_fitted = True # Marca que o fit foi feito
        return self

    def transform(self, X):
        """
        Preenche os valores faltantes em X usando os valores calculados durante o fit.

        Parâmetros
        ----------
        X : pd.DataFrame
            O DataFrame a ser transformado. Deve ter as mesmas colunas que
            foram vistas durante o fit.

        Retorna
        -------
        pd.DataFrame
            O DataFrame com os valores faltantes imputados nas colunas apropriadas.
        """
        check_is_fitted(self, '_is_fitted') # Verifica se o fit foi chamado

        if not isinstance(X, pd.DataFrame):
             raise ValueError("X deve ser um pandas DataFrame.")

        # Verifica se as colunas ajustadas estão presentes
        missing_cols = [col for col in self.fitted_columns_ if col not in X.columns]
        if missing_cols:
            raise ValueError(f"Colunas ajustadas no fit não encontradas no X do transform: {missing_cols}")

        X_transformed = X.copy() # Cria cópia para não modificar o original

        # Itera APENAS sobre as colunas que foram ajustadas no fit
        for column in self.fitted_columns_:
             if X_transformed[column].isnull().any(): # Só imputa se houver nulos
                 # Usa o valor armazenado para aquela coluna específica
                 value_to_impute = self.imputation_values_[column]
                 X_transformed[column] = X_transformed[column].fillna(value_to_impute)

        return X_transformed

    def get_feature_names_out(self, input_features=None):
        """Retorna os nomes das features (não altera os nomes)."""
        check_is_fitted(self)
        if input_features is None:
            return self.feature_names_in_
        else:
            # Retorna os mesmos nomes que entraram
            return np.asarray(input_features, dtype=object)


import pandas as pd # Importa o pandas padrão
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted
import warnings

class PandasCompatEncoder(BaseEstimator, TransformerMixin):
    """
    Aplica encoding em colunas categóricas E numéricas usando pd.get_dummies
    (para onehot) e pd.Series.astype('category').cat.codes (para label),
    utilizando exclusivamente a biblioteca pandas padrão.

    Permite aplicar OneHotEncoding a colunas numéricas, mas emite um aviso.

    Parâmetros
    ----------
    encoding_map : dict
        Dicionário onde chaves são nomes de colunas e valores são
        'onehot' ou 'label'.
    """
    def __init__(self, encoding_map):
        if not isinstance(encoding_map, dict):
            raise ValueError("encoding_map deve ser um dicionário.")
        self.encoding_map = encoding_map
        self.label_maps_ = {} # {col: list_of_ordered_categories}
        self.output_features_ = None # Nomes das colunas após transform
        self._is_fitted = False
        self._feature_names_in = None # Nomes das colunas vistas no fit

    def fit(self, X, y=None):
        """
        Identifica colunas e aprende as categorias para LabelEncoding usando pandas.
        """
        if not isinstance(X, pd.DataFrame):
             try:
                 # Tenta converter para DataFrame pandas se não for
                 X = pd.DataFrame(X)
                 warnings.warn("Input X não era um DataFrame pandas, foi convertido.", UserWarning)
             except Exception as e:
                  raise ValueError(f"X deve ser um DataFrame pandas ou conversível para um. Erro: {e}")

        self._feature_names_in = np.array(X.columns, dtype=object)
        self.label_maps_ = {}
        # Começa com todas as colunas originais
        output_feature_names_list = list(X.columns)

        print("Iniciando fit do PandasCompatEncoder (usando pandas)...")

        columns_to_process = [col for col in self.encoding_map if col in X.columns]
        if not columns_to_process:
             print("Aviso (fit): Nenhuma coluna do encoding_map encontrada em X.")
             self._is_fitted = True
             # Se nada processado, as features de saída são as de entrada
             self.output_features_ = self._feature_names_in
             return self

        # --- Processa colunas que existem ---
        for column in columns_to_process:
            method = self.encoding_map.get(column)
            print(f"  Preparando encoder '{method}' para a coluna '{column}'...")

            # Remove a coluna original da lista de nomes de saída *apenas se for processada*
            if column in output_feature_names_list:
                 output_feature_names_list.remove(column)

            if method == 'onehot':
                 # O fit do OneHot (get_dummies) não requer armazenamento de estado,
                 # mas precisamos saber que a coluna será transformada.
                 pass # A lógica dos nomes fica para get_feature_names_out/transform

            elif method == 'label':
                 try:
                    # Aprende categorias únicas ordenadas, ignorando NaN
                    unique_cats = X[column].dropna().unique()
                    # Garante ordem consistente (importante!) e converte para lista
                    learned_categories = sorted(list(unique_cats))

                    if not learned_categories:
                         print(f"Aviso (fit): Coluna '{column}' para LabelEncoding não possui categorias (apenas NaN?). Ignorando.")
                         if column not in output_feature_names_list: output_feature_names_list.append(column) # Readiciona
                         continue # Pula para a próxima coluna

                    self.label_maps_[column] = learned_categories
                    # Adiciona o nome da nova coluna label encoded
                    output_feature_names_list.append(f"{column}_encoded")

                 except Exception as e:
                    print(f"Erro ao aprender categorias para LabelEncoding de '{column}': {e}. Ignorando coluna.")
                    if column not in output_feature_names_list: output_feature_names_list.append(column)
                    if column in self.label_maps_: del self.label_maps_[column]
                    continue
            else:
                 print(f"Aviso (fit): Método de encoding '{method}' inválido para '{column}'. Ignorando.")
                 # Se inválido, a coluna original permanece na lista (se não foi removida antes)
                 if column not in output_feature_names_list: output_feature_names_list.append(column)


        # Armazena nomes de saída *inferidos* no fit (OHE ainda não detalhado)
        # self.output_features_ = np.array(output_feature_names_list, dtype=object)
        # É melhor deixar output_features_ como None até o transform ser chamado
        self._is_fitted = True
        print("Fit do PandasCompatEncoder concluído.")
        return self

    def transform(self, X):
        """
        Aplica pd.get_dummies e label encoding usando pandas.
        """
        check_is_fitted(self, '_is_fitted')

        if not isinstance(X, pd.DataFrame):
             try:
                 X = pd.DataFrame(X)
                 warnings.warn("Input X não era um DataFrame pandas, foi convertido.", UserWarning)
             except Exception as e:
                  raise ValueError(f"X deve ser um DataFrame pandas ou conversível para um. Erro: {e}")

        # Faz cópia para não modificar o DataFrame original
        X_transformed = X.copy()
        print("Iniciando transform do PandasCompatEncoder (usando pandas)...")

        onehot_cols_to_process = []
        label_cols_to_process = []
        cols_to_drop = []

        # Identifica colunas a serem processadas que existem em X
        for column in self.encoding_map:
             method = self.encoding_map.get(column)
             if column in X.columns:
                 if method == 'onehot':
                     # Avisa se for numérica, mas processa de qualquer forma
                     if pd.api.types.is_numeric_dtype(X[column]):
                          warnings.warn(f"Aplicando OneHotEncoding (pd.get_dummies) à coluna numérica '{column}'.", UserWarning)
                     onehot_cols_to_process.append(column)
                     cols_to_drop.append(column) # Marcar original para remoção
                 elif method == 'label' and column in self.label_maps_: # Só processa se foi fitado
                     label_cols_to_process.append(column)
                     cols_to_drop.append(column) # Marcar original para remoção
                 elif method == 'label':
                      print(f"Aviso (transform): LabelEncoding para '{column}' não foi ajustado no fit ou não tem categorias. Ignorando.")
             else:
                 print(f"Aviso (transform): Coluna '{column}' do encoding_map não encontrada em X. Ignorando.")


        # --- Aplica OneHotEncoding ---
        ohe_df = None
        if onehot_cols_to_process:
            print(f"  Aplicando OneHotEncoding (pd.get_dummies) para: {onehot_cols_to_process}")
            try:
                ohe_df = pd.get_dummies(X_transformed[onehot_cols_to_process],
                                          columns=onehot_cols_to_process, # Especifica quais colunas do subset fazer OHE
                                          prefix=onehot_cols_to_process, # Usa nome da coluna como prefixo
                                          prefix_sep='_',
                                          dummy_na=False, # Não cria coluna extra para NaN
                                          dtype=np.int8) # Usa int8 para economizar memória
                print(f"    Colunas OHE geradas: {ohe_df.columns.to_list()}")
            except Exception as e:
                print(f"Erro durante pd.get_dummies para {onehot_cols_to_process}: {e}. Pulando OHE.")
                # Tira da lista de drop se falhou
                for col in onehot_cols_to_process:
                    if col in cols_to_drop: cols_to_drop.remove(col)
                ohe_df = None

        # --- Aplica LabelEncoding ---
        # Modifica X_transformed diretamente para as colunas label encoded
        if label_cols_to_process:
            print(f"  Aplicando LabelEncoding (astype('category').cat.codes) para: {label_cols_to_process}")
            for column in label_cols_to_process:
                new_col_name = f"{column}_encoded"
                try:
                    # Pega as categorias aprendidas no fit (já ordenadas)
                    learned_categories = self.label_maps_[column]
                    # Cria o tipo categórico com as categorias EXATAS aprendidas
                    cat_type = pd.api.types.CategoricalDtype(categories=learned_categories, ordered=False)
                    # Converte a coluna para este tipo categórico
                    # Valores em X que não estão nas categorias aprendidas viram NaN no cat_series
                    cat_series = X_transformed[column].astype(cat_type)
                    # .cat.codes mapeia categorias para 0, 1, 2... e NaN para -1
                    X_transformed[new_col_name] = cat_series.cat.codes.astype(np.int32) # Usar int32 para caber -1
                    print(f"    Coluna LabelEncoded gerada: {new_col_name}")
                except Exception as e:
                    print(f"Erro durante LabelEncoding para '{column}': {e}. Pulando coluna.")
                    # Tira da lista de drop se falhou
                    if column in cols_to_drop: cols_to_drop.remove(col)
                    # Remove a coluna nova se foi criada parcialmente antes do erro
                    if new_col_name in X_transformed.columns:
                         X_transformed = X_transformed.drop(columns=[new_col_name])
                    continue

        # --- Remove colunas originais processadas ---
        # Garante que só remove colunas que ainda existem e foram marcadas
        cols_to_drop_final = list(set(cols_to_drop) & set(X_transformed.columns))
        if cols_to_drop_final:
            print(f"  Removendo colunas originais processadas: {cols_to_drop_final}")
            X_transformed = X_transformed.drop(columns=cols_to_drop_final)

        # --- Concatena resultados do OHE (se houver) ---
        if ohe_df is not None and not ohe_df.empty:
             print("  Concatenando resultados OHE...")
             # Concatena as colunas restantes de X_transformed com as novas colunas OHE
             X_final = pd.concat([X_transformed, ohe_df], axis=1)
        else:
            # Se não houve OHE ou OHE falhou, o resultado é X_transformed modificado
             X_final = X_transformed

        print(f"Transform do PandasCompatEncoder concluído. Shape final: {X_final.shape}")

        # Garante que a saída é sempre um DataFrame
        if not isinstance(X_final, pd.DataFrame):
             print("Aviso: Resultado final não era DataFrame, convertendo...")
             X_final = pd.DataFrame(X_final)

        # Armazena os nomes das colunas *reais* após a transformação bem-sucedida
        self.output_features_ = np.array(X_final.columns, dtype=object)

        return X_final

    def get_feature_names_out(self, input_features=None):
        """Retorna os nomes das features após a transformação."""
        check_is_fitted(self, '_is_fitted')

        # Prioriza retornar os nomes reais do último transform bem-sucedido
        if hasattr(self, 'output_features_') and self.output_features_ is not None:
             return self.output_features_

        # --- Fallback se transform não foi chamado ---
        # Tenta inferir a partir do fit (será menos preciso para OHE)
        warnings.warn("Nomes de features de saída sendo inferidos do estado do fit."
                      " Chame transform primeiro para nomes precisos, "
                      "especialmente para OneHotEncoding.", UserWarning)
        if self._feature_names_in is None:
             # Se nem fit foi chamado ou X não tinha colunas
             raise ValueError("Não é possível obter nomes de features: fit não foi chamado ou X não tinha colunas.")

        inferred_output_names = []
        original_cols = list(self._feature_names_in)
        processed_in_fit_map = {} # {col: method}

        # Identifica quais colunas do map foram vistas no fit
        for col in self.encoding_map:
            if col in original_cols:
                 processed_in_fit_map[col] = self.encoding_map[col]

        # 1. Adiciona colunas que não foram processadas
        for col in original_cols:
            if col not in processed_in_fit_map:
                inferred_output_names.append(col)

        # 2. Adiciona nomes para colunas processadas (inferido)
        for column, method in processed_in_fit_map.items():
            if method == 'label' and column in self.label_maps_:
                inferred_output_names.append(f"{column}_encoded")
            elif method == 'onehot':
                # Não sabemos as categorias exatas, só podemos dar um placeholder
                inferred_output_names.append(f"{column}_ohe_placeholder*")
            # Ignora 'label' sem map ou métodos inválidos

        return np.array(inferred_output_names, dtype=object)

    def get_params(self, deep=True):
        """Obtém os parâmetros para este estimador."""
        return {"encoding_map": self.encoding_map}

    def set_params(self, **params):
        """Define os parâmetros deste estimador."""
        for key, value in params.items():
            # Poderia adicionar validação aqui se necessário
            setattr(self, key, value)
        # Validação específica para encoding_map
        if 'encoding_map' in params:
             if not isinstance(self.encoding_map, dict):
                  raise ValueError("encoding_map deve ser um dicionário.")
        # Resetar estado fittado se parâmetros relevantes mudam
        self._is_fitted = False
        self.label_maps_ = {}
        self.output_features_ = None
        self._feature_names_in = None
        return self


class ExtractEpisodeNumber(BaseEstimator, TransformerMixin):
    """
    Extrai o primeiro número encontrado em uma coluna de string especificada,
    converte-o para inteiro e substitui a coluna original pela nova coluna numérica.

    Projetado para colunas como 'Episode_Title' no formato 'Texto X',
    onde X é o número a ser extraído.

    Parâmetros
    ----------
    column_name : str
        O nome da coluna string da qual extrair o número.

    new_column_name : str, default='Episode_Number'
         O nome a ser dado à nova coluna que conterá os números inteiros extraídos.

    fillna_value : int, default=0
        Valor a ser usado para preencher casos onde nenhum número é encontrado
        na string da coluna original.

    Attributes
    ----------
    n_features_in_ : int
        Número de features vistas durante o `fit`.

    feature_names_in_ : ndarray of shape (n_features_in_,)
        Nomes das features vistas durante o `fit`.
    """
    def __init__(self, column_name, new_column_name='Episode_Number', fillna_value=0):
        self.column_name = column_name
        self.new_column_name = new_column_name
        self.fillna_value = fillna_value
        if not isinstance(fillna_value, int):
             # Poderia permitir float se a saída fosse float, mas o pedido é int
             raise ValueError("fillna_value deve ser um inteiro.")

    def fit(self, X, y=None):
        if not isinstance(X, pd.DataFrame):
            raise ValueError("X deve ser um pandas DataFrame.")
        if self.column_name not in X.columns:
            raise ValueError(f"Coluna '{self.column_name}' não encontrada em X.")
        # Validação opcional do tipo da coluna
        if not pd.api.types.is_string_dtype(X[self.column_name]) and \
           not pd.api.types.is_object_dtype(X[self.column_name]):
             print(f"Aviso: Coluna '{self.column_name}' não é do tipo string/object. A extração pode falhar.")

        self.n_features_in_ = X.shape[1]
        self.feature_names_in_ = np.array(X.columns, dtype=object)
        self._is_fitted = True
        return self

    def transform(self, X):
        check_is_fitted(self)
        if not isinstance(X, pd.DataFrame):
            raise ValueError("X deve ser um pandas DataFrame.")
        if self.column_name not in X.columns:
             raise ValueError(f"Coluna '{self.column_name}' (vista no fit) não encontrada no X do transform.")
        # Validação opcional de consistência de features
        if X.shape[1] != self.n_features_in_:
             raise ValueError(f"Número de features da entrada ({X.shape[1]}) é diferente do esperado ({self.n_features_in_}).")


        X_transformed = X.copy()

        # Expressão regular para encontrar a primeira sequência de um ou mais dígitos (\d+)
        # .str.extract retorna um DataFrame; pegamos a primeira coluna ([0])
        #astype(str) é adicionado por segurança, caso hajam não-strings na coluna
        extracted_numbers = X_transformed[self.column_name].astype(str).str.extract(r'(\d+)', expand=False)

        # Converter para numérico (float primeiro para lidar com NaN do extract)
        # e preencher NaNs (casos onde não encontrou número) com fillna_value
        numeric_col = pd.to_numeric(extracted_numbers, errors='coerce').fillna(self.fillna_value)

        # Converter para inteiro
        X_transformed[self.new_column_name] = numeric_col.astype(int)

        # Remover a coluna original de string
        X_transformed = X_transformed.drop(columns=[self.column_name])

        return X_transformed

    def get_feature_names_out(self, input_features=None):
        """Retorna os nomes das features após a transformação."""
        check_is_fitted(self)

        if input_features is None:
             # Usa os nomes armazenados durante o fit
             input_features_ = self.feature_names_in_
        else:
             # Valida os nomes fornecidos
             input_features_ = np.asarray(input_features, dtype=object)
             if len(input_features_) != self.n_features_in_:
                  raise ValueError(f"Número de features em input_features ({len(input_features_)}) é diferente do esperado ({self.n_features_in_}).")
             if self.column_name not in input_features_:
                  raise ValueError(f"A coluna original '{self.column_name}' não está presente em input_features.")


        # Cria a lista de nomes de saída
        output_features = list(input_features_)
        try:
            # Encontra o índice da coluna antiga e a substitui pela nova
            idx = output_features.index(self.column_name)
            output_features[idx] = self.new_column_name
        except ValueError:
             # Isso não deveria acontecer se as validações anteriores passaram, mas é uma segurança
             raise RuntimeError(f"Não foi possível encontrar '{self.column_name}' na lista de features para substituição.")

        return np.array(output_features, dtype=object)


class StrategicFeatureEngineer(BaseEstimator, TransformerMixin):
    """
    Aplica engenharia de features estratégica em dados de podcast.
    Assume que o número do episódio já foi extraído em uma etapa anterior.

    Cria features como densidade de anúncios, popularidade combinada,
    indicadores de fim de semana, features cíclicas de tempo e duração,
    e processa outras colunas existentes.

    Parâmetros: [Mesmos parâmetros de antes...]
    """
    def __init__(self, fillna_numeric=0.0, fillna_ad_density=0.0,
                 fillna_episode_length=60.0, fillna_ads=0, clip_ads_upper=3,
                 sentiment_map={'Negative': 0, 'Neutral': 1, 'Positive': 2},
                 weekday_map=None, time_map=None):

        self.fillna_numeric = fillna_numeric
        self.fillna_ad_density = fillna_ad_density
        self.fillna_episode_length = fillna_episode_length
        self.fillna_ads = fillna_ads
        self.clip_ads_upper = clip_ads_upper
        self.sentiment_map = sentiment_map
        self.weekday_map = weekday_map or {
            'Monday': 0, 'Tuesday': 1, 'Wednesday': 2, 'Thursday': 3,
            'Friday': 4, 'Saturday': 5, 'Sunday': 6
        }
        self.time_map = time_map or {
            'Morning': 0, 'Afternoon': 1, 'Evening': 2, 'Night': 3
        }

        # Nomes das colunas que esta classe *adiciona*
        # REMOVIDO: Episode_Title (modificado -> numérico pela outra classe)
        self._added_feature_names = [
            'Has_Guest', 'Ad_Density', 'Popularity_Product', 'Is_Weekend',
            'Day_Time_Combo', 'Length_x_HostPop', 'Sentiment_Numeric',
            'SinWeekday', 'CosWeekday', 'SinTime', 'CosTime',
            'SinEpLen', 'CosEpLen', 'ELen_Int', 'ELen_Dec'
            # Note: Number_of_Ads e Episode_Length_minutes são *modificadas*.
        ]
        # Nomes das colunas que esta classe *remove*
        self._removed_feature_names = [
            'Episode_Sentiment', 'Publication_Day', 'Publication_Time'
            # Colunas intermediárias como 'Weekday', 'Time' também são removidas
        ]

    def fit(self, X, y=None):
        if not isinstance(X, pd.DataFrame):
            raise ValueError("X deve ser um pandas DataFrame.")

        # Colunas necessárias para as transformações
        # REMOVIDO: 'Episode_Title'
        required_cols = [
            'Guest_Popularity_percentage', 'Number_of_Ads',
            'Episode_Length_minutes', 'Host_Popularity_percentage',
            'Publication_Day', 'Publication_Time', 'Episode_Sentiment'
            # 'Episode_Number' (criado antes) não é estritamente necessário *aqui*,
            # mas é esperado que exista no fluxo geral.
        ]
        missing_cols = [col for col in required_cols if col not in X.columns]
        if missing_cols:
            # O erro original ocorreu aqui!
            raise ValueError(f"Colunas necessárias não encontradas em X: {missing_cols}")

        # Verificações de mapeamento (Publication_Day/Time) - mantidas
        if not all(day in self.weekday_map for day in X['Publication_Day'].unique() if pd.notna(day)):
             unknown_days = [day for day in X['Publication_Day'].unique() if pd.notna(day) and day not in self.weekday_map]
             print(f"Aviso: Dias desconhecidos encontrados em 'Publication_Day': {unknown_days}. Serão mapeados para NaN.")
        if not all(time in self.time_map for time in X['Publication_Time'].unique() if pd.notna(time)):
             unknown_times = [time for time in X['Publication_Time'].unique() if pd.notna(time) and time not in self.time_map]
             print(f"Aviso: Horários desconhecidos encontrados em 'Publication_Time': {unknown_times}. Serão mapeados para NaN.")

        self.n_features_in_ = X.shape[1]
        self.feature_names_in_ = np.array(X.columns, dtype=object)
        self._is_fitted = True
        return self

    def transform(self, X):
        check_is_fitted(self)
        if not isinstance(X, pd.DataFrame):
            raise ValueError("X deve ser um pandas DataFrame.")

        X_eng = X.copy()
        cols_to_drop = []

        # --- Pré-processamento / Modificações ---

        # 0. Modificar Number_of_Ads
        X_eng['Number_of_Ads'] = X_eng['Number_of_Ads'].fillna(self.fillna_ads).clip(lower=0, upper=self.clip_ads_upper).astype(np.uint8)

        # 1. Modificar Episode_Length_minutes
        X_eng['Episode_Length_minutes'] = X_eng['Episode_Length_minutes'].fillna(self.fillna_episode_length)

        # 3. Has Guest ...
        X_eng['Has_Guest'] = X_eng['Guest_Popularity_percentage'].notna().astype(int)
        # 4. Ad Density ...
        num_ads = X_eng['Number_of_Ads']
        den_len = X_eng['Episode_Length_minutes']
        X_eng['Ad_Density'] = np.where(den_len > 0, num_ads / den_len, self.fillna_ad_density).astype(np.float32)
        # 5. Popularity Product ...
        host_pop = X_eng['Host_Popularity_percentage'].fillna(self.fillna_numeric)
        guest_pop = X_eng['Guest_Popularity_percentage'].fillna(self.fillna_numeric)
        X_eng['Popularity_Product'] = (host_pop * guest_pop).astype(np.float32)
        # 6. Is Weekend ...
        weekend_days = [day for day, num in self.weekday_map.items() if num >= 5]
        X_eng['Is_Weekend'] = X_eng['Publication_Day'].isin(weekend_days).astype(int)
        cols_to_drop.append('Publication_Day')
        # 7. Day-Time Combo ...
        day_str = X_eng['Publication_Day'].astype(str).fillna('NaN')
        time_str = X_eng['Publication_Time'].astype(str).fillna('NaN')
        X_eng['Day_Time_Combo'] = day_str + '_' + time_str
        cols_to_drop.append('Publication_Time')
        # 8. Length x Host Popularity ...
        length = X_eng['Episode_Length_minutes']
        X_eng['Length_x_HostPop'] = (length * host_pop).astype(np.float32)
        # 9. Sentiment Numeric ...
        X_eng['Sentiment_Numeric'] = X_eng['Episode_Sentiment'].map(self.sentiment_map).fillna(self.fillna_numeric).astype(int)
        cols_to_drop.append('Episode_Sentiment')
        # 10. Cyclical Weekday Features ...
        X_eng['Weekday'] = X_eng['Publication_Day'].map(self.weekday_map)
        X_eng['SinWeekday'] = np.sin(2 * np.pi * X_eng['Weekday'] / 7).astype(np.float32)
        X_eng['CosWeekday'] = np.cos(2 * np.pi * X_eng['Weekday'] / 7).astype(np.float32)
        X_eng['SinWeekday'] = X_eng['SinWeekday'].fillna(np.sin(2 * np.pi * self.fillna_numeric / 7))
        X_eng['CosWeekday'] = X_eng['CosWeekday'].fillna(np.cos(2 * np.pi * self.fillna_numeric / 7))
        cols_to_drop.append('Weekday')
        # 11. Cyclical Time Features ...
        X_eng['Time'] = X_eng['Publication_Time'].map(self.time_map)
        X_eng['SinTime'] = np.sin(2 * np.pi * X_eng['Time'] / 4).astype(np.float32)
        X_eng['CosTime'] = np.cos(2 * np.pi * X_eng['Time'] / 4).astype(np.float32)
        X_eng['SinTime'] = X_eng['SinTime'].fillna(np.sin(2 * np.pi * self.fillna_numeric / 4))
        X_eng['CosTime'] = X_eng['CosTime'].fillna(np.cos(2 * np.pi * self.fillna_numeric / 4))
        cols_to_drop.append('Time')
        # 12. Cyclical Episode Length Features ...
        X_eng['SinEpLen'] = np.sin(2 * np.pi * X_eng['Episode_Length_minutes'] / 60).astype(np.float32)
        X_eng['CosEpLen'] = np.cos(2 * np.pi * X_eng['Episode_Length_minutes'] / 60).astype(np.float32)
        # 13. Episode Length Decomposition ...
        X_eng['ELen_Int'] = np.floor(X_eng['Episode_Length_minutes']).astype(np.int32)
        X_eng['ELen_Dec'] = (X_eng['Episode_Length_minutes'] - X_eng['ELen_Int']).astype(np.float32)


        # --- Final Cleanup ---
        final_cols_to_drop = list(set(cols_to_drop).intersection(set(X_eng.columns)))
        X_eng = X_eng.drop(columns=final_cols_to_drop)

        return X_eng

    def get_feature_names_out(self, input_features=None):
        """Retorna os nomes das features após a transformação."""
        check_is_fitted(self)
        if input_features is None:
            input_features_ = list(self.feature_names_in_)
        else:
            input_features_ = list(np.asarray(input_features, dtype=object))

        output_features = input_features_
        # Remove as colunas que esta classe explicitamente remove
        output_features = [col for col in output_features if col not in self._removed_feature_names]
        # Adiciona as novas features criadas por esta classe
        for new_col in self._added_feature_names:
             if new_col not in output_features:
                  output_features.append(new_col)

        # Como 'Episode_Title' não é mais processada aqui, não precisamos nos preocupar
        # em removê-la ou adicioná-la nesta função. A classe `ExtractEpisodeNumber`
        # lida com a substituição de 'Episode_Title' por 'Episode_Number' em seu
        # próprio get_feature_names_out.

        return np.array(output_features, dtype=object)


from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error

target = 'Listening_Time_minutes'
features = ['Podcast_Name', 'Episode_Title', 'Episode_Length_minutes', 'Genre', 'Host_Popularity_percentage', 'Publication_Day', 'Publication_Time', 'Guest_Popularity_percentage', 'Number_of_Ads', 'Episode_Sentiment']


import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.base import clone, is_regressor, BaseEstimator
from sklearn.metrics import mean_squared_error # Ou sua métrica de score
import time

# --- Defina sua função de scoring aqui (ex: rmse) ---
def rmse(y_true, y_pred):
    """Calcula o Root Mean Squared Error."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    if np.any(np.isnan(y_pred)) or np.any(np.isinf(y_pred)):
        # print("Aviso: NaNs ou Infs encontrados nas previsões. Retornando infinito para RMSE.")
        return np.inf # Retorna infinito se houver inválidos
    return np.sqrt(mean_squared_error(y_true, y_pred))
# ----------------------------------------------------

def cross_validate_and_predict_simple(
    model: BaseEstimator,
    X: pd.DataFrame,
    y: pd.Series,
    X_test: pd.DataFrame, # << ADICIONADO: Precisa do conjunto de teste
    cv: KFold,
    scoring_func=rmse,    # << ADICIONADO: Função de score
    verbose: bool = True
) -> tuple:
    """
    Realiza validação cruzada KFold simples, sem pré-processamento extra.
    Retorna previsões Out-of-Fold (OOF), previsões médias no conjunto de teste
    e os scores de cada fold.

    Args:
        model: Instância do modelo a ser treinado e avaliado.
        X: DataFrame de treino (features).
        y: Series de treino (target).
        X_test: DataFrame de teste (features). Deve ter as mesmas colunas que X.
        cv: Instância de KFold configurada.
        scoring_func: Função que recebe (y_true, y_pred) e retorna um score
                      (padrão: rmse).
        verbose: Se True, imprime detalhes.

    Returns:
        tuple: Contendo:
            - oof_preds (pd.Series): Previsões OOF para o conjunto de treino,
                                     com o índice original de X.
            - avg_test_preds (np.ndarray): Previsões médias para o conjunto de teste.
            - scores (list): Lista de scores (um por fold). Retorna np.inf se
                             ocorrer erro no fold.
    """
    if not isinstance(X, pd.DataFrame): X = pd.DataFrame(X)
    if not isinstance(y, pd.Series): y = pd.Series(y)
    if not isinstance(X_test, pd.DataFrame): X_test = pd.DataFrame(X_test)

    # Guarda o índice original de X para alinhar OOF
    original_index = X.index
    # Reseta índices para usar iloc nos folds
    X = X.reset_index(drop=True)
    y = y.reset_index(drop=True)
    # Garante que X_test tenha as mesmas colunas que X (importante se X foi modificado)
    X_test = X_test[X.columns].copy()

    oof_preds = np.zeros(len(X)) * np.nan # Inicializa OOF com NaN
    test_preds_list = [] # Armazena previsões de teste de cada fold
    scores = []

    if verbose: print(f"  Iniciando CV Simples ({cv.get_n_splits()} folds) com Predições...")

    for fold, (train_idx, valid_idx) in enumerate(cv.split(X, y)):
        start_fold_time = time.time()
        if verbose: print(f"    --- Fold {fold + 1}/{cv.get_n_splits()} ---")

        # --- Divisão ---
        X_train_fold, X_valid_fold = X.iloc[train_idx], X.iloc[valid_idx]
        y_train_fold, y_valid_fold = y.iloc[train_idx], y.iloc[valid_idx]

        # --- Treinamento ---
        model_fold = clone(model)
        try:
            if verbose: print("      Treinando modelo...")
            model_fold.fit(X_train_fold, y_train_fold)

            # --- Previsão OOF ---
            if verbose: print("      Prevendo no conjunto de validação (OOF)...")
            fold_oof_preds = model_fold.predict(X_valid_fold)
            oof_preds[valid_idx] = fold_oof_preds

            # --- Previsão Teste ---
            if verbose: print("      Prevendo no conjunto de teste...")
            fold_test_preds = model_fold.predict(X_test)
            test_preds_list.append(fold_test_preds)

            # --- Score ---
            if verbose: print("      Calculando score...")
            score = scoring_func(y_valid_fold, fold_oof_preds)
            scores.append(score)
            fold_duration = time.time() - start_fold_time
            if verbose: print(f"      Score Fold {fold + 1}: {score:.6f} ({fold_duration:.2f}s)")

        except Exception as e:
            print(f"      ERRO no Fold {fold + 1}/{cv.get_n_splits()}: {e}")
            # Não armazena OOF/Test preds deste fold se o fit/predict falhar
            # Adiciona score infinito para penalizar
            scores.append(np.inf)
            # Garante que OOF permaneça NaN para os índices deste fold
            # oof_preds[valid_idx] = np.nan # Já inicializado com NaN
            # Não adiciona nada a test_preds_list neste caso

    # --- Finalização ---
    # Calcula média das previsões de teste (apenas dos folds que concluíram)
    if test_preds_list: # Só calcula se houver previsões válidas
         # Garante que todos os elementos sejam numpy arrays antes de empilhar
         test_preds_arrays = [np.asarray(preds) for preds in test_preds_list]
         avg_test_preds = np.mean(np.stack(test_preds_arrays, axis=0), axis=0)
    else:
         print("Aviso: Nenhuma previsão de teste válida foi gerada.")
         # Retorna array de NaNs com o formato correto
         avg_test_preds = np.zeros(len(X_test)) * np.nan

    # Cria Series OOF com o índice original
    oof_preds_series = pd.Series(oof_preds, index=original_index, name='oof_predictions')

    if verbose:
        mean_cv_score = np.nanmean(scores) # Ignora inf/nan no cálculo da média
        print(f"  Score CV Médio Final: {mean_cv_score:.6f}")
        print("-" * 20)

    return oof_preds_series, avg_test_preds, scores


encoding_strategy_map = {
    # Colunas originais que passam direto e são categóricas
    'Podcast_Name': 'label',
    'Genre': 'onehot',
    'Day_Time_Combo': 'label',   # Combinação string, alta cardinalidade
    'ELen_Int': 'label',         # Parte inteira da duração tratada como categoria
}

preprocessing_pipeline = Pipeline([
    ('extract_episode_num', ExtractEpisodeNumber(column_name='Episode_Title', new_column_name='Episode_Number')), # A coluna 'Episode_Title' será removida aqui
    ('feature_eng', StrategicFeatureEngineer(fillna_numeric=0.0)), # fillna_numeric=0 aqui é um fallback
    ('imputer', SingleStrategyImputer(strategy=0)),
    ('encoder', PandasCompatEncoder(encoding_map=encoding_strategy_map)),
])

X_fe = preprocessing_pipeline.fit_transform(train[features])
y = train[target]
test_fe = preprocessing_pipeline.transform(test[features])


%%time
from xgboost import XGBRegressor
import warnings
warnings.simplefilter("ignore")

N_SPLITS = 12
kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=42)

model_xgb = XGBRegressor(
    random_state=42,
    device='cuda',
)

oof_predictions_xgb, test_predictions_xgb, fold_scores = cross_validate_and_predict_simple(
    model=model_xgb,
    X=X_fe,
    y=y,
    X_test=test_fe,
    cv=kf,
    scoring_func=rmse # Passa a função RMSE
)

print("\n--- Resultados Finais ---")
print("Shape das previsões OOF:", oof_predictions_xgb.shape)
print("Primeiras 5 previsões OOF:\n", oof_predictions_xgb.head())
print("\nShape das previsões de teste:", test_predictions_xgb.shape)
print("Primeiras 5 previsões de teste:\n", test_predictions_xgb[:5])
print("\nScores por fold:", fold_scores)
print(f"Score médio CV (RMSE): {np.mean(fold_scores):.5f}")


#----------------------------------------------------
# Executar o Estudo Optuna
#----------------------------------------------------

#Iniciando otimização com 50 trials...
#   [I 2025-04-04 15:58:17,067] Trial 0 finished with value: 12.760721915339325 and parameters: {'learning_rate': 0.09278627379252685, 'max_depth': 15, 'subsample': 0.65, 'colsample_bytree': 0.8500000000000001, 'reg_alpha': 1.3063486569235974e-08, 'reg_lambda': 3.283900620326155e-08, 'min_child_weight': 8}. Best is trial 0 with value: 12.760721915339325.
#   Trial 0: Score=12.760722, Avg Best N_Estimators=110
#   [I 2025-04-04 16:20:32,605] Trial 1 finished with value: 12.635357444851552 and parameters: {'learning_rate': 0.014930798362566076, 'max_depth': 9, 'subsample': 0.7, 'colsample_bytree': 0.6, 'reg_alpha': 1.9041872678668332, 'reg_lambda': 0.19732692848036315, 'min_child_weight': 19}. Best is trial 1 with value: 12.635357444851552.
#   Trial 1: Score=12.635357, Avg Best N_Estimators=6381
#   [I 2025-04-04 16:29:03,306] Trial 2 finished with value: 12.795535963679844 and parameters: {'learning_rate': 0.04524856770206565, 'max_depth': 6, 'subsample': 0.65, 'colsample_bytree': 0.5, 'reg_alpha': 1.2108259173992199e-05, 'reg_lambda': 5.6115849352769415e-06, 'min_child_weight': 14}. Best is trial 1 with value: 12.635357444851552.
#   Trial 2: Score=12.795536, Avg Best N_Estimators=4325
"""
# --- Configurações ---
N_TRIALS = 50  # Número de tentativas do Optuna (aumente para uma busca mais completa)
N_SPLITS = 12   # Número de folds para validação cruzada
RANDOM_STATE = 42 # Para reprodutibilidade do KFold

# Cria ou carrega o estudo
study = optuna.create_study(
    direction='minimize'  # Queremos minimizar o RMSE
)

print(f"Sampler is: {study.sampler.__class__.__name__}")
print(f"Iniciando otimização com {N_TRIALS} trials...")

start_time = time.time()
# Executa a otimização
# Adiciona um timeout opcional (em segundos) para limitar a busca
study.optimize(objective, n_trials=N_TRIALS, timeout=None, gc_after_trial=True) # gc_after_trial libera memória
end_time = time.time()

print(f"Otimização concluída em {end_time - start_time:.2f} segundos.")"""


"""
[I 2025-04-04 17:07:54,547] Trial 0 finished with value: 12.625044398536103 and parameters: {'learning_rate': 0.015362014245019574, 'max_depth': 10, 'subsample': 0.7, 'colsample_bytree': 0.8500000000000001, 'reg_alpha': 0.00023141176293072876, 'reg_lambda': 0.11973330959304687, 'min_child_weight': 20}. Best is trial 0 with value: 12.625044398536103.
  Trial 0: Score=12.625044, Avg Best N_Estimators=4278
"""


xgb_params = {
    'learning_rate': 0.015362014245019574,
    'max_depth': 10,
    'subsample': 0.7, 
    'colsample_bytree': 0.8500000000000001,
    'reg_alpha': 0.00023141176293072876,
    'reg_lambda': 0.11973330959304687,
    'min_child_weight': 20
}
N_SPLITS = 50
kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=42)

oof_predictions_xgb, test_predictions_xgb, fold_scores = cross_validate_and_predict_simple(
    model=XGBRegressor(
        **xgb_params,
        random_state=42,
        n_estimators=4278
    ),
    X=X_fe,
    y=y,
    X_test=test_fe,
    cv=kf,
    scoring_func=rmse # Passa a função RMSE
)

print("\n--- Resultados Finais ---")
print("Shape das previsões OOF:", oof_predictions_xgb.shape)
print("Primeiras 5 previsões OOF:\n", oof_predictions_xgb.head())
print("\nShape das previsões de teste:", test_predictions_xgb.shape)
print("Primeiras 5 previsões de teste:\n", test_predictions_xgb[:5])
print("\nScores por fold:", fold_scores)
print(f"Score médio CV (RMSE): {np.mean(fold_scores):.5f}")


submission = pd.read_csv("/kaggle/input/playground-series-s5e4/sample_submission.csv")

submission[target] = test_predictions_xgb

display(submission.head())
display(submission.drop("id", axis=1).hist(bins=50))

submission.to_csv("submission_ch02.csv", index=False)


train_processed = X_fe[final_features_to_keep]
train_processed[target] = train[target]
train_processed['id'] = train.id
test_processed = test_fe[final_features_to_keep]
test_processed['id'] = test.id


train_processed.to_parquet("train_fe.parquet", index=False)
test_processed.to_parquet("test_fe.parquet", index=False)

