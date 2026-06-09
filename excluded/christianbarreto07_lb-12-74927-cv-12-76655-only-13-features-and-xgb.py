%load_ext cudf.pandas

import pandas as pd
import numpy as np

train = pd.read_csv("/kaggle/input/playground-series-s5e4/train.csv")#.sample(n=250_000, random_state=42)
test = pd.read_csv("/kaggle/input/playground-series-s5e4/test.csv")


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


from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted, check_array, check_X_y

class CustomImputer(BaseEstimator, TransformerMixin):
    """
    Imputa valores faltantes em colunas específicas usando estratégias definidas
    individualmente por coluna (mean, median, most_frequent ou valor constante).

    Parâmetros
    ----------
    imputation_map : dict
        Dicionário onde as chaves são os nomes das colunas a serem imputadas
        e os valores são a estratégia de imputação ('mean', 'median',
        'most_frequent') ou um valor constante para preencher os NaNs.
        Exemplo: {'coluna_A': 'mean', 'coluna_B': 'most_frequent', 'coluna_C': 0}
    """
    def __init__(self, imputation_map):
        if not isinstance(imputation_map, dict):
            raise ValueError("imputation_map deve ser um dicionário.")
        self.imputation_map = imputation_map
        self.imputation_values_ = {} # Armazena os valores calculados no fit

    def fit(self, X, y=None):
        # Validação básica de entrada
        if not isinstance(X, pd.DataFrame):
             raise ValueError("X deve ser um pandas DataFrame.")

        self.imputation_values_ = {} # Reinicia a cada fit
        for column, strategy in self.imputation_map.items():
            if column not in X.columns:
                raise ValueError(f"Coluna '{column}' especificada no imputation_map não encontrada em X.")

            if pd.api.types.is_numeric_dtype(X[column]):
                if strategy == 'mean':
                    self.imputation_values_[column] = X[column].mean()
                elif strategy == 'median':
                    self.imputation_values_[column] = X[column].median()
                elif strategy == 'most_frequent':
                    self.imputation_values_[column] = X[column].mode()[0] # Pega o primeiro modo se houver empate
                elif isinstance(strategy, (int, float)): # Se for um valor constante numérico
                    self.imputation_values_[column] = strategy
                elif strategy not in ['mean', 'median', 'most_frequent']:
                     raise ValueError(f"Estratégia '{strategy}' inválida para a coluna numérica '{column}'. Use 'mean', 'median', 'most_frequent' ou um valor numérico constante.")

            elif pd.api.types.is_object_dtype(X[column]) or pd.api.types.is_categorical_dtype(X[column]):
                 if strategy == 'most_frequent':
                    # Garante que não há NaN no cálculo da moda se possível
                    # (embora mode() geralmente ignore NaNs por padrão)
                    valid_modes = X[column].dropna().mode()
                    if not valid_modes.empty:
                       self.imputation_values_[column] = valid_modes[0]
                    else:
                        # Caso raro: coluna só tem NaNs no treino, define um placeholder
                        # Você pode querer tratar isso de forma diferente (ex: erro ou string vazia)
                        self.imputation_values_[column] = "Missing"
                        print(f"Aviso: Coluna '{column}' continha apenas valores nulos no fit. Imputando com 'Missing'.")

                 elif isinstance(strategy, str): # Se for um valor constante string
                     self.imputation_values_[column] = strategy
                 elif strategy in ['mean', 'median']:
                     raise ValueError(f"Estratégia '{strategy}' não pode ser usada na coluna não numérica '{column}'. Use 'most_frequent' ou um valor constante.")
                 else: # Caso seja um valor constante não string (e coluna não numérica)
                     self.imputation_values_[column] = strategy # Permite imputar com números se desejado, mas cuidado

            else:
                 raise TypeError(f"Tipo de dado não suportado para imputação na coluna '{column}': {X[column].dtype}")


            if column in self.imputation_values_ and pd.isna(self.imputation_values_[column]):
                 print(f"Aviso: O valor calculado para imputar a coluna '{column}' com a estratégia '{strategy}' é NaN. "
                       f"Isso pode acontecer se a coluna inteira for NaN no conjunto de treino. Verifique seus dados.")
                 # Decide um fallback, por exemplo, 0 para numérico, "Unknown" para categórico
                 fallback = 0 if pd.api.types.is_numeric_dtype(X[column]) else "Unknown"
                 self.imputation_values_[column] = fallback
                 print(f"Usando fallback: {fallback} para a coluna '{column}'")


        self._is_fitted = True # Marca que o fit foi feito
        return self

    def transform(self, X):
        check_is_fitted(self, '_is_fitted') # Verifica se o fit foi chamado

        # Validação básica de entrada
        if not isinstance(X, pd.DataFrame):
             raise ValueError("X deve ser um pandas DataFrame.")

        X_transformed = X.copy() # Cria cópia para não modificar o original

        for column, value in self.imputation_values_.items():
             if column not in X_transformed.columns:
                  raise ValueError(f"Coluna '{column}' (ajustada no fit) não encontrada no X do transform.")

             if X_transformed[column].isnull().any(): # Só imputa se houver nulos
                 X_transformed[column] = X_transformed[column].fillna(value)

        return X_transformed


class PandasCompatEncoder(BaseEstimator, TransformerMixin):
    """
    Aplica encoding em colunas categóricas usando diferentes métodos,
    otimizado para ambientes Pandas e cuDF.pandas.

    Métodos Suportados por Coluna (via encoding_map):
    - 'onehot': Aplica One-Hot Encoding usando pd.get_dummies.
    - 'label': Aplica Label Encoding (mapeia para inteiros 0, 1, 2...)
               aprendendo as categorias no fit. Valores não vistos/NaN
               são mapeados para -1.
    - dict: Aplica um mapeamento customizado fornecido diretamente.
            Ex: {'col_status': {'ativo': 1, 'inativo': 0, 'pendente': 2}}
            Valores na coluna não presentes nas chaves do dicionário de
            mapeamento serão convertidos para NaN.

    Parâmetros
    ----------
    encoding_map : dict
        Dicionário onde chaves são nomes de colunas e valores são
        o método de encoding ('onehot', 'label') ou um dicionário
        de mapeamento customizado.
    """
    def __init__(self, encoding_map):
        if not isinstance(encoding_map, dict):
            raise ValueError("encoding_map deve ser um dicionário.")
        self.encoding_map = encoding_map
        # Armazena {col: list_of_categories} para label encoding aprendido
        self.label_maps_ = {}
        # Armazena {col: dict_map} para mapeamento customizado fornecido
        self.custom_maps_ = {}
        self.output_features_ = None
        self._is_fitted = False
        self._feature_names_in = None # Guarda nomes das colunas vistas no fit

    def fit(self, X, y=None):
        """
        Identifica colunas, aprende categorias para LabelEncoding e valida
        mapeamentos customizados.
        """
        if not hasattr(X, 'columns'):
             raise ValueError("X deve ser um DataFrame com interface Pandas.")

        self._feature_names_in = np.array(X.columns, dtype=object)
        self.label_maps_ = {}
        self.custom_maps_ = {}
        # Começa com todas as colunas originais
        output_feature_names = list(X.columns)

        columns_to_process = [col for col in self.encoding_map if col in X.columns]
        if not columns_to_process:
             print("Aviso (fit): Nenhuma coluna do encoding_map encontrada em X.")
             self._is_fitted = True
             # Se não há colunas para processar, os nomes de saída são os de entrada
             self.output_features_ = np.array(output_feature_names, dtype=object)
             return self

        # --- Processa colunas que existem ---
        temp_output_names = [] # Lista temporária para novos nomes de saída
        cols_to_remove_from_output = [] # Colunas originais que serão substituídas

        for column in columns_to_process:
            method = self.encoding_map.get(column)

            cols_to_remove_from_output.append(column) # Marcar original para remoção da lista final

            if method == 'onehot':
                 # Para pd.get_dummies, não há estado para 'fitar', apenas marcamos.
                 # Os nomes OHE serão adicionados em get_feature_names_out (placeholder)
                 # ou inferidos no transform.
                 pass # Apenas para estrutura lógica

            elif method == 'label':
                 try:
                    # Aprende as categorias únicas, tratando NaN como uma categoria implícita se necessário
                    learned_categories = X[column].astype('category').cat.categories
                    # Converte para lista CPU se estiver no GPU para serialização/compatibilidade
                    if _has_cudf and isinstance(learned_categories, cudf.Index):
                         self.label_maps_[column] = learned_categories.to_list()
                    else:
                         self.label_maps_[column] = learned_categories.to_list()

                    temp_output_names.append(f"{column}_encoded")
                 except Exception as e:
                    print(f"Erro ao aprender categorias para LabelEncoding de '{column}': {e}. Ignorando coluna.")
                    # Se falhou, não remove a coluna original da lista final
                    if column in cols_to_remove_from_output:
                         cols_to_remove_from_output.remove(column)
                    continue # Pula para próxima coluna

            elif isinstance(method, dict):
                # É um mapeamento customizado
                if not isinstance(method, dict):
                     print(f"Erro (fit): Mapeamento para '{column}' não é um dicionário. Ignorando.")
                     if column in cols_to_remove_from_output:
                          cols_to_remove_from_output.remove(column)
                     continue
                # Valida se o dicionário não está vazio (opcional, mas bom)
                if not method:
                     print(f"Aviso (fit): Mapeamento customizado para '{column}' está vazio. Ignorando.")
                     if column in cols_to_remove_from_output:
                          cols_to_remove_from_output.remove(column)
                     continue

                self.custom_maps_[column] = method
                temp_output_names.append(f"{column}_mapped")

            else:
                 # Método inválido
                 print(f"Aviso (fit): Método de encoding '{method}' inválido para '{column}'. Ignorando.")
                 if column in cols_to_remove_from_output:
                      cols_to_remove_from_output.remove(column)

        # Constrói a lista final de nomes de features de saída
        final_output_names = [col for col in output_feature_names if col not in cols_to_remove_from_output]
        # Adiciona os nomes das colunas geradas (label, mapped)
        final_output_names.extend(temp_output_names)
        # Placeholder para OHE será tratado em get_feature_names_out

        self.output_features_ = np.array(final_output_names, dtype=object) # Armazena nomes (aproximados para OHE)
        self._is_fitted = True
        return self

    def transform(self, X):
        """
        Aplica os encodings definidos (onehot, label, custom map).
        """
        check_is_fitted(self, '_is_fitted')
        if not hasattr(X, 'columns'):
             raise ValueError("X deve ser um DataFrame com interface Pandas.")

        # Garante cópia e tipo correto (Pandas ou cuDF.pandas)
        X_transformed = X.copy()

        onehot_cols_to_process = []
        label_cols_to_process = []
        custom_map_cols_to_process = []
        cols_to_drop = [] # Colunas originais que serão substituídas com sucesso

        # Identifica colunas a serem processadas que existem em X e foram fitadas
        for column in self.encoding_map:
             method = self.encoding_map[column]
             if column in X.columns:
                 if method == 'onehot':
                     # Verifica se a coluna não é numérica (get_dummies funciona melhor com obj/cat)
                     if not pd.api.types.is_numeric_dtype(X[column]):
                          onehot_cols_to_process.append(column)
                          # Marcamos para dropar, mas só dropamos se OHE funcionar
                     else:
                          print(f"Aviso (transform): Coluna '{column}' é numérica, pulando OneHotEncoding.")
                 elif method == 'label' and column in self.label_maps_: # Verifica se foi fitado
                     label_cols_to_process.append(column)
                     cols_to_drop.append(column) # Marcar para dropar
                 elif isinstance(method, dict) and column in self.custom_maps_: # Verifica se foi fitado
                     custom_map_cols_to_process.append(column)
                     cols_to_drop.append(column) # Marcar para dropar
                 elif method == 'label' or isinstance(method, dict):
                      print(f"Aviso (transform): Encoding para '{column}' não foi ajustado corretamente no fit ou é inválido. Ignorando.")
             else:
                 print(f"Aviso (transform): Coluna '{column}' do encoding_map não encontrada em X. Ignorando.")

        # --- Aplica OneHotEncoding de forma vetorizada ---
        ohe_df = None
        processed_ohe_cols = [] # Guarda colunas que realmente passaram pelo OHE
        if onehot_cols_to_process:
            try:
                # Usa pd.get_dummies que será cudf.get_dummies se cudf.pandas carregado
                ohe_df = pd.get_dummies(X_transformed[onehot_cols_to_process],
                                          columns=onehot_cols_to_process,
                                          prefix=onehot_cols_to_process,
                                          prefix_sep='_ohe_',
                                          dummy_na=False, # Não cria coluna para NaN
                                          dtype=np.int8) # Especifica dtype para OHE
                processed_ohe_cols = onehot_cols_to_process # Marcar como processadas com sucesso
            except Exception as e:
                print(f"Erro durante pd.get_dummies para {onehot_cols_to_process}: {e}. Pulando OHE.")
                ohe_df = None

        # Adiciona colunas OHE processadas com sucesso à lista de drop
        cols_to_drop.extend(processed_ohe_cols)

        # --- Aplica LabelEncoding de forma vetorizada ---
        if label_cols_to_process:
            for column in label_cols_to_process:
                new_col_name = f"{column}_encoded"
                try:
                    # Pega as categorias aprendidas no fit
                    learned_categories = self.label_maps_[column]

                    # Converte a coluna para o tipo categórico COM as categorias aprendidas
                    # Usamos pd.Categorical que funciona com Pandas e cuDF.pandas
                    cat_series = pd.Categorical(X_transformed[column], categories=learned_categories, ordered=False)

                    # Obtém os códigos (-1 para NaN/não vistos)
                    # Define o tipo de dados explicitamente
                    codes = cat_series.codes
                    if _has_cudf and isinstance(codes, cudf.Series):
                         X_transformed[new_col_name] = codes.astype(np.int32)
                    else:
                         X_transformed[new_col_name] = codes.astype(np.int32) # Usa int32 para o -1

                except Exception as e:
                    print(f"Erro durante LabelEncoding para '{column}': {e}. Pulando coluna.")
                    # Remove coluna da lista de drop se falhou
                    if column in cols_to_drop: cols_to_drop.remove(column)
                    if new_col_name in X_transformed.columns: # Remove a coluna criada se deu erro
                         X_transformed = X_transformed.drop(columns=[new_col_name])
                    continue

        # --- Aplica Mapeamento Customizado ---
        if custom_map_cols_to_process:
            for column in custom_map_cols_to_process:
                 new_col_name = f"{column}_mapped"
                 try:
                      custom_map = self.custom_maps_[column]
                      # Aplica o mapeamento. Valores não no mapa se tornarão NaN.
                      X_transformed[new_col_name] = X_transformed[column].map(custom_map)
                      # Opcional: converter tipo de dados aqui se souber o esperado (ex: int, float)
                      # X_transformed[new_col_name] = X_transformed[new_col_name].astype(np.float32) # Exemplo

                 except Exception as e:
                      print(f"Erro durante Mapeamento Customizado para '{column}': {e}. Pulando coluna.")
                      # Remove coluna da lista de drop se falhou
                      if column in cols_to_drop: cols_to_drop.remove(column)
                      if new_col_name in X_transformed.columns: # Remove coluna criada se erro
                          X_transformed = X_transformed.drop(columns=[new_col_name])
                      continue

        # --- Remove colunas originais processadas com sucesso ---
        # Remove duplicatas da lista cols_to_drop (caso OHE e outro método se apliquem por erro)
        final_cols_to_drop = sorted(list(set(cols_to_drop)))
        if final_cols_to_drop:
            # Garante que só tenta dropar colunas que ainda existem
            cols_actually_in_df = [col for col in final_cols_to_drop if col in X_transformed.columns]
            if cols_actually_in_df:
                X_transformed = X_transformed.drop(columns=cols_actually_in_df)
            else:
                print("    Nenhuma das colunas marcadas para drop existe mais.")

        # --- Concatena OHE se existir ---
        if ohe_df is not None and not ohe_df.empty:
             # Usa pd.concat que será cudf.concat se aplicável
             # Garante que índices estão alinhados se possível (reset_index pode ser necessário em casos complexos)
             # Mas geralmente concat alinha automaticamente se os índices originais são os mesmos
             X_final = pd.concat([X_transformed, ohe_df], axis=1)
        else:
             X_final = X_transformed # Se não houve OHE, o resultado já está em X_transformed

        print(f"Transform do PandasCompatEncoder concluído. Shape final: {X_final.shape}")
        # Verifica se o resultado ainda é um DataFrame
        if not hasattr(X_final, 'columns'):
             print("Aviso: Resultado final não é DataFrame, convertendo...")
             X_final = pd.DataFrame(X_final) # Converte se virou Series

        # Opcional: Reordenar colunas para uma ordem mais previsível
        # current_cols = X_final.columns.tolist()
        # desired_order = self.get_feature_names_out() # Pode precisar ajustar get_feature_names_out
        # X_final = X_final[desired_order]

        return X_final

    def get_feature_names_out(self, input_features=None):
        """Retorna os nomes das features após a transformação."""
        check_is_fitted(self, '_is_fitted')

        if input_features is None:
            if self._feature_names_in is not None:
                input_features = self._feature_names_in
            else:
                # Tenta inferir do encoding_map como último recurso
                # Nota: Isso não inclui colunas não mapeadas do X original.
                # É melhor que fit() sempre capture _feature_names_in.
                print("Aviso (get_feature_names_out): Usando chaves de encoding_map como fallback para input_features.")
                input_features = list(self.encoding_map.keys())

        original_cols = list(input_features)
        output_feature_names = []

        # Colunas que foram efetivamente processadas (e substituídas) no fit
        processed_cols_label = set(self.label_maps_.keys())
        processed_cols_custom = set(self.custom_maps_.keys())
        # Para OHE, precisamos saber quais colunas foram *tentadas*
        processed_cols_ohe_attempted = {
            col for col, method in self.encoding_map.items()
            if method == 'onehot' and col in original_cols
        }

        # 1. Adiciona colunas que NÃO foram processadas
        for col in original_cols:
            if col not in processed_cols_label and \
               col not in processed_cols_custom and \
               col not in processed_cols_ohe_attempted:
                output_feature_names.append(col)

        # 2. Adiciona nomes das colunas processadas (Label e Custom Map)
        for column in original_cols:
             method = self.encoding_map.get(column)
             if method == 'label' and column in processed_cols_label:
                 output_feature_names.append(f"{column}_encoded")
             elif isinstance(method, dict) and column in processed_cols_custom:
                 output_feature_names.append(f"{column}_mapped")
             # OHE é tratado separadamente abaixo porque os nomes são dinâmicos

        # 3. Adiciona nomes das colunas OHE (Placeholder ou real se pudéssemos saber)
        # NOTA: pd.get_dummies() determina os nomes exatos no `transform`.
        # get_feature_names_out idealmente refletiria isso, mas é difícil sem
        # re-executar parte da lógica ou ter X disponível.
        # Scikit-learn geralmente espera que `get_feature_names_out` seja preciso.
        # Uma solução robusta pode envolver chamar `get_dummies` em uma amostra
        # ou armazenar os nomes OHE gerados no `transform`.
        # Por simplicidade aqui, adicionaremos placeholders ou os nomes reais se
        # tivéssemos armazenado no transform (o que não fizemos).
        # Vamos adicionar placeholders por enquanto.
        for column in processed_cols_ohe_attempted:
             # Idealmente, pegaríamos os nomes de self._ohe_feature_names[column]
             # se tivéssemos armazenado no transform.
             output_feature_names.append(f"{column}_ohe_placeholder") # Placeholder

        # Retorna como array numpy tipo objeto, como esperado pelo scikit-learn
        return np.array(output_feature_names, dtype=object)


    def get_params(self, deep=True):
        # Retorna os parâmetros que podem ser setados
        return {"encoding_map": self.encoding_map}

    def set_params(self, **params):
        # Define os parâmetros. Importante para GridSearchCV, etc.
        for key, value in params.items():
            # Validação específica para encoding_map se ele for setado
            if key == 'encoding_map':
                if not isinstance(value, dict):
                    raise ValueError("encoding_map deve ser um dicionário.")
            setattr(self, key, value)
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
    Aplica engenharia de features estratégica em dados de podcast,
    operando principalmente antes do OneHotEncoding.

    Assume que features como 'Episode_Number' podem ter sido extraídas antes.

    Cria interações categóricas (string-based), features numéricas
    derivadas, processa valores nulos, aplica clipping e mapeamentos.

    Features Originais Esperadas:
    'Podcast_Name', 'Episode_Title' (opcional, se ainda não processado),
    'Episode_Length_minutes', 'Genre', 'Host_Popularity_percentage',
    'Publication_Day', 'Publication_Time', 'Guest_Popularity_percentage',
    'Number_of_Ads', 'Episode_Sentiment'
    """
    def __init__(self,
                 # --- Parâmetros de Preenchimento ---
                 fillna_numeric=0.0,
                 fillna_guest_pop=-1.0, # Valor distinto para indicar ausência vs. popularidade zero
                 fillna_episode_length=60.0, # Mediana/Média pode ser melhor
                 fillna_ads=0,
                 fillna_sentiment='Neutral', # Ou a moda

                 # --- Parâmetros de Processamento ---
                 clip_ads_upper=5, # Ajustar baseado na distribuição
                 clip_length_upper=180.0, # Ex: 3 horas, ajustar
                 min_length_for_density=1.0, # Evitar divisão por zero/pequeno

                 # --- Mapeamentos (pode ajustar ou passar do exterior) ---
                 sentiment_map=None, # {'Negative': -1, 'Neutral': 0, 'Positive': 1}
                 weekday_map=None,   # {'Monday': 0, ..., 'Sunday': 6}
                 time_map=None,      # {'Morning': 0, ..., 'Night': 3}

                 # --- Controle de Interações (True para criar) ---
                 create_genre_day_interaction=True,
                 create_genre_time_interaction=True,
                 create_genre_sentiment_interaction=True,
                 create_day_time_interaction=True, # Já existia como Day_Time_Combo
                 create_podcast_day_interaction=False, # Cuidado: alta cardinalidade
                 create_podcast_time_interaction=False, # Cuidado: alta cardinalidade
                 create_host_pop_x_length=True, # Já existia como Length_x_HostPop
                 create_guest_pop_x_length=True,
                 create_ads_x_length=True,
                 create_ads_x_host_pop=True,
                 create_ads_x_guest_pop=True,
                 create_hasguest_x_genre=True,
                 create_hasguest_x_day=True,
                 ):

        # Preenchimento
        self.fillna_numeric = fillna_numeric
        self.fillna_guest_pop = fillna_guest_pop
        self.fillna_episode_length = fillna_episode_length
        self.fillna_ads = fillna_ads
        self.fillna_sentiment = fillna_sentiment

        # Processamento
        self.clip_ads_upper = clip_ads_upper
        self.clip_length_upper = clip_length_upper
        self.min_length_for_density = min_length_for_density

        # Mapeamentos
        self.sentiment_map = sentiment_map or {'Negative': -1, 'Neutral': 0, 'Positive': 1}
        self.weekday_map = weekday_map or {
            'Monday': 0, 'Tuesday': 1, 'Wednesday': 2, 'Thursday': 3,
            'Friday': 4, 'Saturday': 5, 'Sunday': 6
        }
        self.time_map = time_map or {
            'Morning': 0, 'Afternoon': 1, 'Evening': 2, 'Night': 3
        }

        # Controle de Criação
        self.create_genre_day_interaction = create_genre_day_interaction
        self.create_genre_time_interaction = create_genre_time_interaction
        self.create_genre_sentiment_interaction = create_genre_sentiment_interaction
        self.create_day_time_interaction = create_day_time_interaction
        self.create_podcast_day_interaction = create_podcast_day_interaction
        self.create_podcast_time_interaction = create_podcast_time_interaction
        self.create_host_pop_x_length = create_host_pop_x_length
        self.create_guest_pop_x_length = create_guest_pop_x_length
        self.create_ads_x_length = create_ads_x_length
        self.create_ads_x_host_pop = create_ads_x_host_pop
        self.create_ads_x_guest_pop = create_ads_x_guest_pop
        self.create_hasguest_x_genre = create_hasguest_x_genre
        self.create_hasguest_x_day = create_hasguest_x_day

        # --- Nomes das features gerenciadas ---
        # Colunas *adicionadas* ou *significativamente modificadas* por esta classe
        self._managed_feature_names = [
            # Modificadas
            'Number_of_Ads', 'Episode_Length_minutes',
            'Guest_Popularity_percentage', # Modificado por fillna
            'Host_Popularity_percentage', # Modificado por fillna
            'Episode_Sentiment', # Modificado por fillna e mapeamento

            # Novas
            'Has_Guest', 'Ad_Density', 'Popularity_Sum', 'Popularity_Diff',
            'Is_Weekend', 'Sentiment_Numeric',
            'SinWeekday', 'CosWeekday', 'SinTime', 'CosTime',
             # Features de Interação (nomes podem mudar ligeiramente na implementação)
            'Genre_Day', 'Genre_Time', 'Genre_Sentiment', 'Day_Time_Combo',
            'Podcast_Day', 'Podcast_Time',
            'HostPop_x_Length', 'GuestPop_x_Length', 'Ads_x_Length',
            'Ads_x_HostPop', 'Ads_x_GuestPop',
            'HasGuest_x_Genre', 'HasGuest_x_Day'
             # Adicionar mais se criar (ex: 'Popularity_Ratio', 'Length_Bins', etc.)
        ]
        # Colunas originais que são *removidas* ou *substituídas*
        self._removed_feature_names = [
            # Removidas porque foram mapeadas/combinadas
            #'Publication_Day', 'Publication_Time',
            # 'Episode_Sentiment' é mantida mas modificada (mapeada para numérico)
            # Colunas intermediárias como 'Weekday', 'Time' também são removidas internamente
        ]
        # Inicializa a lista de features adicionadas dinamicamente
        self._added_feature_names_dynamic = []

    def fit(self, X, y=None):
        if not isinstance(X, pd.DataFrame):
            raise ValueError("X deve ser um pandas DataFrame.")

        # Colunas necessárias para as transformações
        required_cols = [
            'Episode_Length_minutes', 'Genre', 'Host_Popularity_percentage',
            'Publication_Day', 'Publication_Time', 'Guest_Popularity_percentage',
            'Number_of_Ads', 'Episode_Sentiment', 'Podcast_Name' # Adicionado Podcast_Name
        ]
        missing_cols = [col for col in required_cols if col not in X.columns]
        if missing_cols:
            raise ValueError(f"Colunas necessárias não encontradas em X: {missing_cols}")

        # --- Verificações de Mapeamento (Importante!) ---
        self._check_mapping_coverage(X, 'Publication_Day', self.weekday_map, 'Dias')
        self._check_mapping_coverage(X, 'Publication_Time', self.time_map, 'Horários')
        self._check_mapping_coverage(X, 'Episode_Sentiment', self.sentiment_map, 'Sentimentos')
        # Poderia adicionar verificações para Genre, Podcast_Name se houvesse mapeamentos

        self.n_features_in_ = X.shape[1]
        self.feature_names_in_ = np.array(X.columns, dtype=object)
        self._is_fitted = True
        return self

    def _check_mapping_coverage(self, df, col_name, mapping, entity_name):
        """Verifica se todos os valores únicos na coluna estão no mapeamento."""
        if col_name not in df.columns: return # Coluna pode não existir
        unique_values = df[col_name].dropna().unique()
        unknown_values = [v for v in unique_values if v not in mapping]
        if unknown_values:
            print(f"Aviso: {entity_name} desconhecidos encontrados em '{col_name}': {unknown_values}. "
                  f"Serão mapeados para NaN ou valor de fillna durante a transformação.")


    def transform(self, X):
        check_is_fitted(self)
        if not isinstance(X, pd.DataFrame):
            raise ValueError("X deve ser um pandas DataFrame.")

        X_eng = X.copy()
        self._added_feature_names_dynamic = [] # Resetar a cada transform()
        cols_to_drop_internal = [] # Colunas intermediárias a remover no final

        # --- 0. Limpeza e Preenchimento Inicial ---
        X_eng['Guest_Popularity_percentage'] = X_eng['Guest_Popularity_percentage'].fillna(self.fillna_guest_pop).astype(np.float32)
        X_eng['Host_Popularity_percentage'] = X_eng['Host_Popularity_percentage'].fillna(self.fillna_numeric).astype(np.float32)
        X_eng['Episode_Length_minutes'] = X_eng['Episode_Length_minutes'].fillna(self.fillna_episode_length)
        X_eng['Number_of_Ads'] = X_eng['Number_of_Ads'].fillna(self.fillna_ads)
        X_eng['Episode_Sentiment'] = X_eng['Episode_Sentiment'].fillna(self.fillna_sentiment)
        # Preencher outras categóricas se necessário (Genre, Podcast_Name, Day, Time)
        for col in ['Genre', 'Podcast_Name', 'Publication_Day', 'Publication_Time']:
             if col in X_eng.columns:
                   X_eng[col] = X_eng[col].fillna(f'Unknown_{col}') # Ou outra estratégia

        # --- 1. Clipping ---
        X_eng['Number_of_Ads'] = X_eng['Number_of_Ads'].clip(lower=0, upper=self.clip_ads_upper).astype(np.int16)
        X_eng['Episode_Length_minutes'] = X_eng['Episode_Length_minutes'].clip(lower=0, upper=self.clip_length_upper).astype(np.float32)

        # --- 2. Features Base Derivadas ---
        # Has_Guest: Indica presença real de convidado (considerando fillna_guest_pop)
        X_eng['Has_Guest'] = (X_eng['Guest_Popularity_percentage'] != self.fillna_guest_pop).astype(int)
        self._added_feature_names_dynamic.append('Has_Guest')

        # Ad_Density: Densidade de anúncios
        length_safe = X_eng['Episode_Length_minutes'].clip(lower=self.min_length_for_density)
        X_eng['Ad_Density'] = (X_eng['Number_of_Ads'] / length_safe).astype(np.float32)
        self._added_feature_names_dynamic.append('Ad_Density')

        # Popularity Sum/Diff
        host_pop = X_eng['Host_Popularity_percentage']
        # Use 0 para convidado ausente nos cálculos de soma/diferença se fillna_guest_pop for < 0
        guest_pop_eff = np.where(X_eng['Has_Guest'] == 1, X_eng['Guest_Popularity_percentage'], 0.0)
        X_eng['Popularity_Sum'] = (host_pop + guest_pop_eff).astype(np.float32)
        X_eng['Popularity_Diff'] = (host_pop - guest_pop_eff).astype(np.float32)
        self._added_feature_names_dynamic.extend(['Popularity_Sum', 'Popularity_Diff'])
        # Popularity_Product (opcional, pode ser redundante com Sum/Diff)
        # X_eng['Popularity_Product'] = (host_pop * guest_pop_eff).astype(np.float32)
        # self._added_feature_names_dynamic.append('Popularity_Product')

        # --- 3. Mapeamentos Categóricos e Features Cíclicas ---
        # Dia da Semana
        X_eng['Weekday'] = X_eng['Publication_Day'].map(self.weekday_map).fillna(self.fillna_numeric) # Preenche NaNs pós-map
        X_eng['Is_Weekend'] = X_eng['Weekday'].isin([self.weekday_map.get('Saturday', 5), self.weekday_map.get('Sunday', 6)]).astype(int)
        X_eng['SinWeekday'] = np.sin(2 * np.pi * X_eng['Weekday'] / 7).astype(np.float32)
        X_eng['CosWeekday'] = np.cos(2 * np.pi * X_eng['Weekday'] / 7).astype(np.float32)
        cols_to_drop_internal.append('Weekday')
        self._added_feature_names_dynamic.extend(['Is_Weekend', 'SinWeekday', 'CosWeekday'])

        # Hora do Dia
        X_eng['Time'] = X_eng['Publication_Time'].map(self.time_map).fillna(self.fillna_numeric)
        X_eng['SinTime'] = np.sin(2 * np.pi * X_eng['Time'] / 4).astype(np.float32)
        X_eng['CosTime'] = np.cos(2 * np.pi * X_eng['Time'] / 4).astype(np.float32)
        cols_to_drop_internal.append('Time')
        self._added_feature_names_dynamic.extend(['SinTime', 'CosTime'])

        # Sentimento
        X_eng['Sentiment_Numeric'] = X_eng['Episode_Sentiment'].map(self.sentiment_map).fillna(self.sentiment_map.get(self.fillna_sentiment, 0)).astype(int)
        # Mantém 'Episode_Sentiment' original para interações de string, mas 'Sentiment_Numeric' pode ser usada diretamente
        self._added_feature_names_dynamic.append('Sentiment_Numeric')

        # --- 4. Interações Categóricas (String-based) ---
        # Garante que são strings antes de concatenar
        genre_str = X_eng['Genre'].astype(str)
        day_str = X_eng['Publication_Day'].astype(str)
        time_str = X_eng['Publication_Time'].astype(str)
        sentiment_str = X_eng['Episode_Sentiment'].astype(str)
        podcast_str = X_eng['Podcast_Name'].astype(str)
        hasguest_str = 'Guest' + X_eng['Has_Guest'].astype(str) # 'Guest1' ou 'Guest0'

        if self.create_genre_day_interaction:
            X_eng['Genre_Day'] = genre_str + '_' + day_str
            self._added_feature_names_dynamic.append('Genre_Day')
        if self.create_genre_time_interaction:
            X_eng['Genre_Time'] = genre_str + '_' + time_str
            self._added_feature_names_dynamic.append('Genre_Time')
        if self.create_genre_sentiment_interaction:
            X_eng['Genre_Sentiment'] = genre_str + '_' + sentiment_str
            self._added_feature_names_dynamic.append('Genre_Sentiment')
        if self.create_day_time_interaction:
            X_eng['Day_Time_Combo'] = day_str + '_' + time_str
            self._added_feature_names_dynamic.append('Day_Time_Combo')
        if self.create_podcast_day_interaction:
            X_eng['Podcast_Day'] = podcast_str + '_' + day_str
            self._added_feature_names_dynamic.append('Podcast_Day')
        if self.create_podcast_time_interaction:
            X_eng['Podcast_Time'] = podcast_str + '_' + time_str
            self._added_feature_names_dynamic.append('Podcast_Time')
        if self.create_hasguest_x_genre:
            X_eng['HasGuest_x_Genre'] = hasguest_str + '_' + genre_str
            self._added_feature_names_dynamic.append('HasGuest_x_Genre')
        if self.create_hasguest_x_day:
             X_eng['HasGuest_x_Day'] = hasguest_str + '_' + day_str
             self._added_feature_names_dynamic.append('HasGuest_x_Day')

        # --- 5. Interações Numérico x Numérico / Numérico x Binário ---
        length = X_eng['Episode_Length_minutes']
        num_ads = X_eng['Number_of_Ads']
        # host_pop, guest_pop_eff já definidos

        if self.create_host_pop_x_length:
             X_eng['HostPop_x_Length'] = (host_pop * length).astype(np.float32)
             self._added_feature_names_dynamic.append('HostPop_x_Length')
        if self.create_guest_pop_x_length:
             X_eng['GuestPop_x_Length'] = (guest_pop_eff * length).astype(np.float32)
             self._added_feature_names_dynamic.append('GuestPop_x_Length')
        if self.create_ads_x_length:
             X_eng['Ads_x_Length'] = (num_ads * length).astype(np.float32)
             self._added_feature_names_dynamic.append('Ads_x_Length')
        if self.create_ads_x_host_pop:
             X_eng['Ads_x_HostPop'] = (num_ads * host_pop).astype(np.float32)
             self._added_feature_names_dynamic.append('Ads_x_HostPop')
        if self.create_ads_x_guest_pop:
             X_eng['Ads_x_GuestPop'] = (num_ads * guest_pop_eff).astype(np.float32)
             self._added_feature_names_dynamic.append('Ads_x_GuestPop')

        # --- 6. Outras Features Potenciais (Exemplos) ---
        # - Binning de Features Numéricas (ex: Duração)
        #   X_eng['Length_Bin'] = pd.cut(length, bins=[0, 30, 60, 90, self.clip_length_upper], labels=['Curto', 'Medio', 'Longo', 'MuitoLongo'], include_lowest=True)
        #   self._added_feature_names_dynamic.append('Length_Bin') # Seria OHE'd depois

        # - Ratios (ex: Popularity Ratio)
        #   X_eng['Popularity_Ratio'] = np.where(host_pop > 0, guest_pop_eff / host_pop, 0).astype(np.float32)
        #   self._added_feature_names_dynamic.append('Popularity_Ratio')

        # - Features Polinomiais (com cautela, podem explodir)
        #   X_eng['Length_Sq'] = (length ** 2).astype(np.float32)
        #   self._added_feature_names_dynamic.append('Length_Sq')


        # --- Final Cleanup ---
        # Remove colunas intermediárias
        actual_cols_to_drop_internal = [col for col in cols_to_drop_internal if col in X_eng.columns]
        X_eng = X_eng.drop(columns=actual_cols_to_drop_internal)

        # Remove colunas originais que foram substituídas ou não são mais necessárias
        actual_cols_to_drop_original = [col for col in self._removed_feature_names if col in X_eng.columns]
        X_eng = X_eng.drop(columns=actual_cols_to_drop_original)

        # Remove duplicatas da lista de features adicionadas (caso alguma interação tenha o mesmo nome de uma existente)
        self._added_feature_names_dynamic = sorted(list(set(self._added_feature_names_dynamic)))

        return X_eng

    def get_feature_names_out(self, input_features=None):
        """Retorna os nomes das features após a transformação."""
        check_is_fitted(self)
        if input_features is None:
            # Usa as features vistas no fit se nenhum input for dado
            input_features_ = list(self.feature_names_in_)
        else:
            # Converte para lista para manipulação
            input_features_ = list(np.asarray(input_features, dtype=object))

        # Começa com as features de entrada
        output_features = input_features_

        # Remove as colunas que esta classe explicitamente remove/substitui
        output_features = [col for col in output_features if col not in self._removed_feature_names]

        # Adiciona as novas features criadas dinamicamente no último transform()
        # É crucial que transform() tenha sido chamado para popular esta lista
        for new_col in self._added_feature_names_dynamic:
             if new_col not in output_features:
                  output_features.append(new_col)

        # Garante que features que foram apenas modificadas (e não removidas) permaneçam
        # Ex: 'Number_of_Ads', 'Episode_Length_minutes', 'Episode_Sentiment', 'Guest/Host_Popularity'
        # Elas já devem estar em `output_features` se estavam em `input_features_`
        # e não foram removidas.

        # Remove duplicatas e garante ordem consistente (opcional, mas bom)
        # Usando dict.fromkeys para manter a ordem de inserção enquanto remove duplicatas
        output_features = list(dict.fromkeys(output_features))

        return np.array(output_features, dtype=object)


from itertools import combinations, permutations
from sklearn.base import BaseEstimator, TransformerMixin
import warnings

# Ignorar avisos de RuntimeWarning que podem ocorrer com divisões/logs/sqrts de zeros ou NaNs
warnings.filterwarnings("ignore", category=RuntimeWarning)

class FeatureCombiner(BaseEstimator, TransformerMixin):
    """
    Cria novas features combinando um conjunto específico de colunas numéricas.

    Gera combinações aritméticas (soma, diferença, produto, razão) e
    interações polinomiais das colunas fornecidas, adicionando-as
    ao DataFrame original.

    Parâmetros:
    ----------
    columns_to_combine : list de str
        Lista contendo exatamente os 4 nomes das colunas a serem usadas
        para gerar as novas features. Ex: ['col1', 'col2', 'col3', 'col4']

    epsilon : float, default=1e-6
        Valor pequeno adicionado aos denominadores para evitar divisão por zero.

    Attributes:
    -----------
    feature_names_in_ : ndarray de nomes de features vistos durante o fit.
    n_features_in_ : int, número de features vistas durante o fit.
    columns_to_combine_ : list de str, cópia das colunas a combinar.
    new_feature_names_ : list de str, nomes das features geradas.
    feature_names_out_ : list de str, nomes de todas as features após transform.
    """
    def __init__(self, columns_to_combine=['Episode_Length_minutes',
                                           'Host_Popularity_percentage',
                                           'Guest_Popularity_percentage',
                                           'Episode_Number'],
                 epsilon=1e-6):
        if not isinstance(columns_to_combine, list) or len(columns_to_combine) != 4:
            raise ValueError("`columns_to_combine` deve ser uma lista com exatamente 4 nomes de colunas.")
        self.columns_to_combine = columns_to_combine
        self.epsilon = epsilon

    def fit(self, X, y=None):
        """
        Verifica as colunas de entrada e armazena metadados.
        Nenhuma lógica de 'fit' real é necessária, pois as transformações
        são baseadas apenas nos nomes das colunas.

        Parâmetros:
        ----------
        X : pd.DataFrame
            DataFrame de entrada.
        y : Ignorado.

        Retorna:
        -------
        self : object
            Instância ajustada do transformador.
        """
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X) # Tenta converter se não for DataFrame

        # Verifica se as colunas existem
        missing_cols = [col for col in self.columns_to_combine if col not in X.columns]
        if missing_cols:
            raise ValueError(f"As seguintes colunas especificadas em `columns_to_combine` não foram encontradas em X: {missing_cols}")

        self.feature_names_in_ = np.array(X.columns, dtype=object)
        self.n_features_in_ = X.shape[1]
        self.columns_to_combine_ = self.columns_to_combine[:] # Guarda uma cópia
        # Define um atributo para indicar que o fit foi chamado
        self._is_fitted = True
        return self

    def transform(self, X):
        """
        Aplica as combinações de features ao DataFrame X.

        Parâmetros:
        ----------
        X : pd.DataFrame
            DataFrame para transformar. Deve ter as colunas especificadas
            durante o `fit`.

        Retorna:
        -------
        X_transformed : pd.DataFrame
            DataFrame original com as novas features combinadas adicionadas.
            Valores infinitos resultantes das operações são substituídos por NaN.
        """
        if not hasattr(self, '_is_fitted') or not self._is_fitted:
            raise RuntimeError("Este estimador FeatureCombiner não foi ajustado ('fitted'). Chame 'fit' primeiro.")

        if not isinstance(X, pd.DataFrame):
             X = pd.DataFrame(X) # Tenta converter

        # Verifica se as colunas necessárias estão presentes
        missing_cols = [col for col in self.columns_to_combine_ if col not in X.columns]
        if missing_cols:
            raise ValueError(f"As seguintes colunas necessárias para `transform` não foram encontradas em X: {missing_cols}")

        # Verifica se as colunas de entrada correspondem às do fit (opcional, mas bom)
        # if list(X.columns) != list(self.feature_names_in_):
        #    warnings.warn("As colunas do DataFrame de entrada para `transform` não correspondem exatamente às vistas durante `fit`.")


        df_orig = X.copy() # Trabalha numa cópia
        cols = self.columns_to_combine_
        epsilon = self.epsilon
        df_new = pd.DataFrame(index=df_orig.index) # DataFrame para novas features
        self.new_feature_names_ = [] # Reinicia a lista de novas features

        # --- Geração Sistemática de Features ---

        # 1. Features Polinomiais Simples (Grau 2, 3, 4)
        for col in cols:
            for degree in [2, 3, 4]:
                new_col_name = f'{col}_pow{degree}'
                if new_col_name not in df_new.columns:
                    df_new[new_col_name] = df_orig[col] ** degree
                    self.new_feature_names_.append(new_col_name)
     
        # 2. Interações Par-a-Par (Combinações de 2 colunas)
        for c1, c2 in combinations(cols, 2):
            # Soma
            new_col_name = f'{c1}_plus_{c2}'
            if new_col_name not in df_new.columns:
                df_new[new_col_name] = df_orig[c1] + df_orig[c2]
                self.new_feature_names_.append(new_col_name)
            # Diferença (ambas as direções)
            new_col_name = f'{c1}_minus_{c2}'
            if new_col_name not in df_new.columns:
                df_new[new_col_name] = df_orig[c1] - df_orig[c2]
                self.new_feature_names_.append(new_col_name)
            new_col_name = f'{c2}_minus_{c1}'
            if new_col_name not in df_new.columns:
                df_new[new_col_name] = df_orig[c2] - df_orig[c1]
                self.new_feature_names_.append(new_col_name)
            # Produto
            new_col_name = f'{c1}_x_{c2}'
            if new_col_name not in df_new.columns:
                df_new[new_col_name] = df_orig[c1] * df_orig[c2]
                self.new_feature_names_.append(new_col_name)
            # Razão (ambas as direções)
            new_col_name = f'{c1}_div_{c2}'
            if new_col_name not in df_new.columns:
                df_new[new_col_name] = df_orig[c1] / (df_orig[c2] + epsilon)
                self.new_feature_names_.append(new_col_name)
            new_col_name = f'{c2}_div_{c1}'
            if new_col_name not in df_new.columns:
                df_new[new_col_name] = df_orig[c2] / (df_orig[c1] + epsilon)
                self.new_feature_names_.append(new_col_name)

        # 3. Interações Triplas (Combinações de 3 colunas)
        for c1, c2, c3 in combinations(cols, 3):
            # Soma
            new_col_name = f'{c1}_{c2}_{c3}_sum'
            if new_col_name not in df_new.columns:
                df_new[new_col_name] = df_orig[c1] + df_orig[c2] + df_orig[c3]
                self.new_feature_names_.append(new_col_name)
            # Produto
            new_col_name = f'{c1}_{c2}_{c3}_prod'
            if new_col_name not in df_new.columns:
                df_new[new_col_name] = df_orig[c1] * df_orig[c2] * df_orig[c3]
                self.new_feature_names_.append(new_col_name)

            # Razões Mistas (permutações para cobrir todas as variações)
            # Tipo 1: (X+Y)/Z
            for x, y, z in permutations([c1, c2, c3], 3):
                new_col_name = f'({x}_plus_{y})_div_{z}'
                if new_col_name not in df_new.columns:
                    df_new[new_col_name] = (df_orig[x] + df_orig[y]) / (df_orig[z] + epsilon)
                    self.new_feature_names_.append(new_col_name)
            # Tipo 2: (X*Y)/Z
            for x, y, z in permutations([c1, c2, c3], 3):
                new_col_name = f'({x}_x_{y})_div_{z}'
                if new_col_name not in df_new.columns:
                    df_new[new_col_name] = (df_orig[x] * df_orig[y]) / (df_orig[z] + epsilon)
                    self.new_feature_names_.append(new_col_name)
            # Tipo 3: X / (Y+Z)
            # Usar combinations para o denominador para evitar nomes duplicados como X/(Y+Z) e X/(Z+Y)
            for x in [c1, c2, c3]:
                den_cols = sorted([col for col in [c1, c2, c3] if col != x])
                y, z = den_cols[0], den_cols[1]
                new_col_name = f'{x}_div_({y}_plus_{z})'
                if new_col_name not in df_new.columns:
                    df_new[new_col_name] = df_orig[x] / (df_orig[y] + df_orig[z] + epsilon)
                    self.new_feature_names_.append(new_col_name)
            # Tipo 4: X / (Y*Z)
            for x in [c1, c2, c3]:
                den_cols = sorted([col for col in [c1, c2, c3] if col != x])
                y, z = den_cols[0], den_cols[1]
                new_col_name = f'{x}_div_({y}_x_{z})'
                if new_col_name not in df_new.columns:
                     # Adicionar epsilon ao produto também pode ser útil se Y ou Z puder ser 0
                    df_new[new_col_name] = df_orig[x] / (df_orig[y] * df_orig[z] + epsilon)
                    self.new_feature_names_.append(new_col_name)


        # 4. Interações Quádruplas Simples
        c1, c2, c3, c4 = cols
        # Soma
        new_col_name = f'all4_sum'
        if new_col_name not in df_new.columns:
            df_new[new_col_name] = df_orig[c1] + df_orig[c2] + df_orig[c3] + df_orig[c4]
            self.new_feature_names_.append(new_col_name)
        # Produto
        new_col_name = f'all4_prod'
        if new_col_name not in df_new.columns:
            df_new[new_col_name] = df_orig[c1] * df_orig[c2] * df_orig[c3] * df_orig[c4]
            self.new_feature_names_.append(new_col_name)

        # 5. Interações Polinomiais Par-a-Par (Graus mais altos)
        for c1, c2 in combinations(cols, 2):
            for p1 in [1, 2, 3]: # Grau para c1
                for p2 in [1, 2, 3]: # Grau para c2
                    if p1 == 1 and p2 == 1: continue # Já coberto em interações par-a-par simples

                    # Produto Potencializado
                    new_col_name = f'{c1}pow{p1}_x_{c2}pow{p2}'
                    if new_col_name not in df_new.columns:
                        df_new[new_col_name] = (df_orig[c1]**p1) * (df_orig[c2]**p2)
                        self.new_feature_names_.append(new_col_name)

                    # Razões Potencializadas (ambas direções)
                    new_col_name = f'{c1}pow{p1}_div_{c2}pow{p2}'
                    if new_col_name not in df_new.columns:
                        df_new[new_col_name] = (df_orig[c1]**p1) / (df_orig[c2]**p2 + epsilon)
                        self.new_feature_names_.append(new_col_name)

                    # Evita duplicar a razão se p1=p2 (ex: c1^2/c2^2 vs c2^2/c1^2) - já tratado pelo nome
                    new_col_name = f'{c2}pow{p2}_div_{c1}pow{p1}'
                    # Verifica se o nome reverso já existe (importante se p1=p2)
                    rev_name_check = f'{c1}pow{p1}_div_{c2}pow{p2}' if p1==p2 else "N/A"
                    if new_col_name not in df_new.columns and new_col_name != rev_name_check:
                         df_new[new_col_name] = (df_orig[c2]**p2) / (df_orig[c1]**p1 + epsilon)
                         self.new_feature_names_.append(new_col_name)

        # 6. Interações Quádruplas (Partições 2 a 2)
        # Existem 3 maneiras de particionar 4 itens em 2 pares: ((1,2),(3,4)), ((1,3),(2,4)), ((1,4),(2,3))
        partitions = [
            ((cols[0], cols[1]), (cols[2], cols[3])),
            ((cols[0], cols[2]), (cols[1], cols[3])),
            ((cols[0], cols[3]), (cols[1], cols[2]))
        ]
        for (c1, c2), (c3, c4) in partitions:
            term1_sum = df_orig[c1] + df_orig[c2]
            term2_sum = df_orig[c3] + df_orig[c4]
            term1_prod = df_orig[c1] * df_orig[c2]
            term2_prod = df_orig[c3] * df_orig[c4]

            ops = {'sum': (term1_sum, term2_sum), 'prod': (term1_prod, term2_prod)}
            op_names = {'sum': f'({c1}+{c2})', 'prod': f'({c1}x{c2})'}
            op_names2 = {'sum': f'({c3}+{c4})', 'prod': f'({c3}x{c4})'}

            for op1_key, (term1_val, term2_val) in ops.items(): # Iterar sobre sum/prod para o primeiro par
                op1_name = op_names[op1_key]
                for op2_key, (termA_val, termB_val) in ops.items(): # Iterar sobre sum/prod para o segundo par
                     op2_name = op_names2[op2_key]

                     # Razão Direta: Op1 / Op2
                     new_col_name = f'{op1_name}_div_{op2_name}'
                     if new_col_name not in df_new.columns:
                         df_new[new_col_name] = term1_val / (termB_val + epsilon)
                         self.new_feature_names_.append(new_col_name)
                     # Razão Inversa: Op2 / Op1
                     new_col_name = f'{op2_name}_div_{op1_name}'
                     if new_col_name not in df_new.columns:
                         df_new[new_col_name] = termB_val / (term1_val + epsilon)
                         self.new_feature_names_.append(new_col_name)


        # 7. Interações Triplas Polinomiais (Ex: c1^2 * c2 * c3)
        for c1, c2, c3 in combinations(cols, 3):
             # Tipo: X^2 * Y * Z (permuta qual é X)
             for squared_col, other1, other2 in permutations([c1, c2, c3]):
                 new_col_name = f'{squared_col}pow2_x_{other1}_x_{other2}'
                 if new_col_name not in df_new.columns:
                     df_new[new_col_name] = (df_orig[squared_col]**2) * df_orig[other1] * df_orig[other2]
                     self.new_feature_names_.append(new_col_name)

             # Tipo: X^2 * Y^2 * Z (permuta qual é Z)
             for single_col, sq1, sq2 in permutations([c1, c2, c3]):
                 # Cria nome canônico ordenando os termos ao quadrado
                 sorted_sq = sorted([sq1, sq2])
                 new_col_name = f'{sorted_sq[0]}pow2_x_{sorted_sq[1]}pow2_x_{single_col}'
                 if new_col_name not in df_new.columns:
                     df_new[new_col_name] = (df_orig[sq1]**2) * (df_orig[sq2]**2) * df_orig[single_col]
                     self.new_feature_names_.append(new_col_name)

        # 8. Diferenças de Quadrados
        for c1, c2 in combinations(cols, 2):
            new_col_name = f'{c1}pow2_minus_{c2}pow2'
            if new_col_name not in df_new.columns:
                 df_new[new_col_name] = df_orig[c1]**2 - df_orig[c2]**2
                 self.new_feature_names_.append(new_col_name)
            # A diferença inversa não é necessária aqui pois é apenas -1* a anterior
            # Mas podemos adicionar se quisermos mais features (embora colineares)
            # new_col_name = f'{c2}pow2_minus_{c1}pow2'
            # if new_col_name not in df_new.columns:
            #      df_new[new_col_name] = df_orig[c2]**2 - df_orig[c1]**2
            #      self.new_feature_names_.append(new_col_name)


        # --- Fim da Geração ---

        print(f"FeatureCombiner: Gerou {len(self.new_feature_names_)} novas features.")

        # Concatenar features originais e novas
        X_transformed = pd.concat([df_orig, df_new], axis=1)

        # Substituir infinitos por NaN (resultado de divisões por zero ou overflows)
        X_transformed = X_transformed.replace([np.inf, -np.inf], np.nan)

        # Armazenar nomes finais para get_feature_names_out
        self.feature_names_out_ = list(X_transformed.columns)

        return X_transformed

    def get_feature_names_out(self, input_features=None):
        """
        Retorna os nomes das features após a transformação.

        Parâmetros:
        ----------
        input_features : array-like de str ou None, default=None
            Nomes das features de entrada. Se None, usa `feature_names_in_`.

        Retorna:
        -------
        feature_names_out : list de str
            Lista dos nomes de todas as features na saída (originais + novas).
        """
        if not hasattr(self, '_is_fitted') or not self._is_fitted:
            raise RuntimeError("Este estimador FeatureCombiner não foi ajustado ('fitted').")

        # Idealmente, deveria verificar a consistência de input_features se fornecido
        # Mas para este caso, simplesmente retornamos os nomes calculados no último transform
        if hasattr(self, 'feature_names_out_'):
             return self.feature_names_out_
        else:
             # Caso transform não tenha sido chamado ainda após o fit
             # Retorna os nomes originais + os nomes potenciais (menos robusto)
             if hasattr(self, 'new_feature_names_') and self.new_feature_names_:
                 return list(self.feature_names_in_) + self.new_feature_names_
             else: # Se nem fit nem transform foram chamados ou transform falhou
                 return list(self.feature_names_in_)





from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error

target = 'Listening_Time_minutes'
features = [
    'Podcast_Name', 
    'Episode_Title', # New is 'Episode_Number'
    'Episode_Length_minutes',
    'Genre', 
    'Host_Popularity_percentage',
    'Publication_Day',
    'Publication_Time',
    'Guest_Popularity_percentage',
    'Number_of_Ads', 
    'Episode_Sentiment'
]
features_cat_ini = [
    'Podcast_Name',
    'Genre',
    'Publication_Day',
    'Publication_Time',
    'Episode_Sentiment',
]
features_num_ini = [
    'Episode_Length_minutes',
    'Host_Popularity_percentage',
    'Guest_Popularity_percentage',
    'Number_of_Ads',
    'Episode_Number',
]


class FrequencyEncoder(BaseEstimator, TransformerMixin):
    """
    Aplica Frequency Encoding a colunas categóricas especificadas.

    Substitui cada categoria pela sua frequência (contagem) observada
    nos dados de treino (`fit`).

    Parâmetros
    ----------
    columns_to_encode : list of str
        Lista contendo os nomes das colunas categóricas a serem codificadas.
    suffix : str, default='_freq'
        Sufixo a ser adicionado ao nome da coluna original para criar o nome
        da nova coluna codificada. Ex: 'coluna' -> 'coluna_freq'.
    handle_unseen : str or int or float, default=0
        Valor a ser usado para categorias presentes nos dados de `transform`
        mas que não foram vistas durante o `fit`. Comuns são 0 ou 1.
    drop_original : bool, default=False
        Se True, remove as colunas categóricas originais após a codificação.
        Se False, mantém as colunas originais junto com as novas codificadas.
    """
    def __init__(self, columns_to_encode, suffix='_freq', handle_unseen=0, drop_original=False):
        if not isinstance(columns_to_encode, list):
            raise TypeError("columns_to_encode deve ser uma lista de nomes de colunas.")
        self.columns_to_encode = columns_to_encode
        self.suffix = suffix
        self.handle_unseen = handle_unseen # Valor para categorias não vistas no treino
        self.drop_original = drop_original
        self._is_fitted = False

    def fit(self, X, y=None):
        """
        Aprende os mapeamentos de frequência para cada categoria.

        Pode ser chamado nos dados de treino completos ou nos dados de treino
        de um fold específico (recomendado para pureza na CV).

        Parâmetros
        ----------
        X : pd.DataFrame
            DataFrame contendo as features.
        y : pd.Series or np.array, optional
            Target, ignorado por este transformador.
        """
        if not isinstance(X, pd.DataFrame):
            raise ValueError("X deve ser um pandas DataFrame.")

        # Armazena os mapeamentos de frequência aprendidos {nome_coluna: pd.Series(mapa)}
        self.frequency_maps_ = {}
        self._feature_names_in = np.array(X.columns, dtype=object) # Guarda nomes de entrada


        for col in self.columns_to_encode:
            if col not in X.columns:
                print(f"  [FreqEncoder Fit] Aviso: Coluna '{col}' não encontrada em X. Pulando.")
                continue

            # Calcula a frequência (contagem) de cada categoria na coluna
            # value_counts já retorna uma Series com categoria como índice e contagem como valor
            frequency_map = X[col].value_counts()
            self.frequency_maps_[col] = frequency_map

        self._is_fitted = True
        return self

    def transform(self, X):
        """
        Aplica os mapeamentos de frequência aprendidos ao DataFrame X.

        Parâmetros
        ----------
        X : pd.DataFrame
            DataFrame a ser transformado.

        Retorna
        -------
        pd.DataFrame
            DataFrame com as novas colunas codificadas por frequência.
        """
        check_is_fitted(self, '_is_fitted')
        if not isinstance(X, pd.DataFrame):
            raise ValueError("X deve ser um pandas DataFrame.")

        X_transformed = X.copy() # Trabalha em uma cópia

        cols_to_drop_list = []

        for col in self.columns_to_encode:
            new_col_name = f"{col}{self.suffix}"

            if col not in self.frequency_maps_:
                # Coluna não foi fitada (talvez não existia no X do fit)
                print(f"  [FreqEncoder Transform] Aviso: Mapa para '{col}' não aprendido. Pulando.")
                continue
            if col not in X_transformed.columns:
                 print(f"  [FreqEncoder Transform] Aviso: Coluna '{col}' não encontrada no X para transformar. Pulando.")
                 continue

            mapping_series = self.frequency_maps_[col]

            # Aplica o mapeamento de frequência
            X_transformed[new_col_name] = X_transformed[col].map(mapping_series)

            # Preenche NaNs (categorias não vistas no fit) com o valor definido
            n_unseen = X_transformed[new_col_name].isnull().sum()
            if n_unseen > 0:
                X_transformed[new_col_name].fillna(self.handle_unseen, inplace=True)
                print(f"      {n_unseen} valores não vistos em '{col}' preenchidos com {self.handle_unseen}.")


            # Converte para int se possível (frequências são contagens)
            try:
                # Verifica se há NaNs remanescentes (se handle_unseen for NaN) ou infinitos
                if X_transformed[new_col_name].isnull().any() or np.isinf(X_transformed[new_col_name]).any():
                     print(f"      Mantendo dtype float para '{new_col_name}' devido a NaN/Inf.")
                else:
                    X_transformed[new_col_name] = X_transformed[new_col_name].astype(np.int64)
            except Exception as e:
                 print(f"      Não foi possível converter '{new_col_name}' para int64: {e}. Mantendo dtype atual.")


            # Adiciona à lista para dropar se necessário
            if self.drop_original:
                cols_to_drop_list.append(col)

        # Dropar colunas originais se solicitado
        if self.drop_original and cols_to_drop_list:
            # Garante que só dropa colunas que existem
            cols_to_drop_final = [c for c in cols_to_drop_list if c in X_transformed.columns]
            if cols_to_drop_final:
                 print(f"  [FreqEncoder Transform] Removendo colunas originais: {cols_to_drop_final}")
                 X_transformed.drop(columns=cols_to_drop_final, inplace=True)

        return X_transformed

    # get_feature_names_out é útil para pipelines
    def get_feature_names_out(self, input_features=None):
        """Retorna os nomes das features após a transformação."""
        check_is_fitted(self, '_is_fitted')

        if input_features is None:
             # Tenta usar os nomes vistos no fit
             if hasattr(self, '_feature_names_in'):
                  input_features = self._feature_names_in
             else: # Fallback muito básico
                  raise ValueError("Nomes das features de entrada não fornecidos e não inferidos no fit.")

        original_cols = list(input_features)
        output_feature_names = []

        # Colunas que foram mapeadas para frequência
        mapped_cols = set(self.frequency_maps_.keys())

        for col in original_cols:
            if col in mapped_cols:
                # Se a coluna foi mapeada E não foi dropada, adiciona o nome codificado
                if not self.drop_original:
                    output_feature_names.append(col) # Mantém original se drop_original=False
                output_feature_names.append(f"{col}{self.suffix}") # Adiciona a nova
            else:
                # Se a coluna não foi mapeada, apenas a mantém
                output_feature_names.append(col)

        # Remove duplicatas (caso drop_original=False e nome original seja igual ao codificado por algum motivo)
        return np.array(list(dict.fromkeys(output_feature_names)), dtype=object)

    def get_params(self, deep=True):
        return {
            "columns_to_encode": self.columns_to_encode,
            "suffix": self.suffix,
            "handle_unseen": self.handle_unseen,
            "drop_original": self.drop_original
        }

    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        # Adicionar validações se necessário
        if 'columns_to_encode' in parameters:
            if not isinstance(self.columns_to_encode, list):
                  raise TypeError("columns_to_encode deve ser uma lista de nomes de colunas.")
        return self


class BasicFeatureSelector(BaseEstimator, TransformerMixin):
    """
    Realiza seleção básica de features com base em características intrínsecas.

    Remove features que atendem aos seguintes critérios (configuráveis):
    1. Baixa Variância (quase constante) - para features numéricas.
    2. Colunas Idênticas (duplicadas).
    3. Alta Porcentagem de Valores Ausentes.
    4. Alta Correlação entre pares de features numéricas (mantém uma).

    Parâmetros
    ----------
    variance_threshold : float or None, default=0.01
        Limiar de variância. Features numéricas com variância abaixo
        deste valor serão removidas. Se None, desativa esta seleção.
        Nota: Variância 0 remove features constantes.
    select_identical : bool, default=True
        Se True, remove colunas que são cópias exatas umas das outras,
        mantendo a primeira ocorrência.
    missing_threshold : float or None, default=0.95
        Limiar de valores ausentes. Features com proporção de valores
        ausentes maior que este valor serão removidas. Se None, desativa.
    correlation_threshold : float or None, default=0.98
        Limiar de correlação (valor absoluto). Para pares de features
        numéricas com correlação acima deste valor, uma delas será
        removida. Se None, desativa esta seleção.
    """
    def __init__(self, variance_threshold=0.01, select_identical=True,
                 missing_threshold=0.95, correlation_threshold=0.98):
        self.variance_threshold = variance_threshold
        self.select_identical = select_identical
        self.missing_threshold = missing_threshold
        self.correlation_threshold = correlation_threshold
        self._is_fitted = False

    def fit(self, X, y=None):
        """
        Identifica as features a serem removidas com base nos critérios.

        Parâmetros
        ----------
        X : pd.DataFrame
            DataFrame de treino contendo as features.
        y : pd.Series or np.array, optional
            Target, ignorado por este transformador.
        """
        if not isinstance(X, pd.DataFrame):
            raise ValueError("X deve ser um pandas DataFrame.")

        self._feature_names_in = list(X.columns)
        self.features_to_drop_ = set() # Usar set para evitar duplicatas

        print("[FeatureSelector Fit] Iniciando identificação de features a remover...")
        """
        # --- 1. Seleção por Baixa Variância (Numéricas) ---
        if self.variance_threshold is not None:
            print(f"  Verificando baixa variância (threshold={self.variance_threshold})...")
            numerical_cols = X.select_dtypes(include=np.number).columns
            variances = X[numerical_cols].var()
            low_variance_cols = variances[variances < self.variance_threshold].index.tolist()
            if low_variance_cols:
                print(f"    Identificadas {len(low_variance_cols)} colunas com baixa variância: {low_variance_cols}")
                self.features_to_drop_.update(low_variance_cols)
            else:
                print("    Nenhuma coluna com baixa variância encontrada.")
        """
        # --- 2. Seleção por Colunas Idênticas ---
        if self.select_identical:
            print("  Verificando colunas idênticas...")
            # Técnica eficiente: Transpor e encontrar linhas duplicadas
            duplicated_cols = set()
            cols_to_check = [c for c in X.columns if c not in self.features_to_drop_] # Otimização: não re-checar as já marcadas
            if len(cols_to_check) > 1 :
                transposed_data = X[cols_to_check].T
                # keep=False marca TODAS as ocorrências de duplicatas
                # Precisamos iterar para manter a primeira e marcar as outras
                duplicates_mask = transposed_data.duplicated(keep=False)
                if duplicates_mask.any():
                    potential_duplicates = transposed_data[duplicates_mask].index.tolist()
                    # Agrupa para saber quais são idênticas entre si
                    seen_groups = set()
                    cols_to_keep_from_duplicates = set()

                    for col1 in potential_duplicates:
                        if col1 in cols_to_keep_from_duplicates or col1 in duplicated_cols:
                            continue # Já tratada

                        group_key = tuple(X[col1].tolist()) # Chave baseada nos valores
                        if group_key not in seen_groups:
                             # Primeira vez que vemos este grupo de valores
                             identical_group = [c for c in potential_duplicates if tuple(X[c].tolist()) == group_key]
                             if len(identical_group) > 1:
                                  print(f"    Grupo de colunas idênticas encontrado: {identical_group}")
                                  # Mantém a primeira, marca as outras para dropar
                                  cols_to_keep_from_duplicates.add(identical_group[0])
                                  duplicated_cols.update(identical_group[1:])
                             seen_groups.add(group_key) # Marca o grupo como visto

                    if duplicated_cols:
                        print(f"    Marcadas {len(duplicated_cols)} colunas idênticas para remoção: {list(duplicated_cols)}")
                        self.features_to_drop_.update(duplicated_cols)
                    else:
                        print("    Nenhuma coluna idêntica encontrada (após checagem inicial).")
                else:
                    print("    Nenhuma coluna idêntica encontrada.")
            else:
                print("    Não há colunas suficientes para verificar duplicatas.")


        # --- 3. Seleção por Alta Porcentagem de Missing ---
        if self.missing_threshold is not None:
            print(f"  Verificando alta porcentagem de missing (threshold={self.missing_threshold})...")
            missing_ratios = X.isnull().mean()
            high_missing_cols = missing_ratios[missing_ratios > self.missing_threshold].index.tolist()
            # Não adiciona se já estiver marcada para dropar por outro motivo
            new_high_missing = [col for col in high_missing_cols if col not in self.features_to_drop_]
            if new_high_missing:
                print(f"    Identificadas {len(new_high_missing)} colunas com alta % de missing: {new_high_missing}")
                self.features_to_drop_.update(new_high_missing)
            else:
                print("    Nenhuma coluna com alta porcentagem de missing encontrada.")

        # --- 4. Seleção por Alta Correlação (Numéricas) ---
        if self.correlation_threshold is not None:
            print(f"  Verificando alta correlação (threshold={self.correlation_threshold})...")
            numerical_cols_remaining = [
                col for col in X.select_dtypes(include=np.number).columns
                if col not in self.features_to_drop_
            ]
            if len(numerical_cols_remaining) > 1:
                corr_matrix = X[numerical_cols_remaining].corr().abs()
                upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)) # Pega triângulo superior

                # Encontra colunas com correlação > threshold
                to_drop_corr = set() # Usar set para evitar adicionar a mesma coluna múltiplas vezes
                for i in range(len(upper_tri.columns)):
                    for j in range(i):
                         # Se col_i já vai ser dropada, não precisa checar pares com ela
                         if upper_tri.columns[i] in to_drop_corr: continue
                         # Se col_j já vai ser dropada, não precisa checar pares com ela
                         if upper_tri.columns[j] in to_drop_corr: continue

                         if upper_tri.iloc[j, i] > self.correlation_threshold:
                            # Se encontrou par altamente correlacionado, decide qual dropar
                            # Heurística comum: dropar a segunda coluna do par (índice i)
                            col_to_drop_here = upper_tri.columns[i]
                            print(f"    Correlação alta ({upper_tri.iloc[j, i]:.3f}) entre '{upper_tri.columns[j]}' e '{col_to_drop_here}'. Removendo '{col_to_drop_here}'.")
                            to_drop_corr.add(col_to_drop_here)

                if to_drop_corr:
                     print(f"    Identificadas {len(to_drop_corr)} colunas altamente correlacionadas para remoção: {list(to_drop_corr)}")
                     self.features_to_drop_.update(to_drop_corr)
                else:
                     print("    Nenhuma coluna altamente correlacionada encontrada.")
            else:
                print("    Não há colunas numéricas suficientes para verificar correlação.")


        # Finaliza
        self.features_to_drop_ = sorted(list(self.features_to_drop_)) # Ordena para consistência
        self.features_kept_ = [col for col in self._feature_names_in if col not in self.features_to_drop_]
        self._n_features_out = len(self.features_kept_) # Número de features de saída

        print(f"[FeatureSelector Fit] Concluído. {len(self.features_to_drop_)} features marcadas para remoção.")
        print(f"  Features a remover: {self.features_to_drop_}")
        print(f"  Features a manter: {self.features_kept_}")

        self._is_fitted = True
        return self

    def transform(self, X):
        """
        Remove as features identificadas no `fit`.

        Parâmetros
        ----------
        X : pd.DataFrame
            DataFrame a ser transformado.

        Retorna
        -------
        pd.DataFrame
            DataFrame com as features irrelevantes removidas.
        """
        check_is_fitted(self, '_is_fitted')
        if not isinstance(X, pd.DataFrame):
            raise ValueError("X deve ser um pandas DataFrame.")

        print("[FeatureSelector Transform] Removendo features identificadas...")
        # Garante que só tenta remover colunas que existem no X atual
        cols_to_drop_now = [col for col in self.features_to_drop_ if col in X.columns]

        if cols_to_drop_now:
            print(f"  Removendo {len(cols_to_drop_now)} colunas: {cols_to_drop_now}")
            X_transformed = X.drop(columns=cols_to_drop_now)
            print(f"  Shape após remoção: {X_transformed.shape}")
        else:
            print("  Nenhuma coluna identificada no fit precisou ser removida do DataFrame atual.")
            X_transformed = X.copy() # Retorna cópia se nada foi dropado

        # Verifica se as colunas restantes correspondem ao esperado
        remaining_cols = list(X_transformed.columns)
        if remaining_cols != self.features_kept_:
             # Isso pode acontecer se X no transform já não tinha algumas colunas originais
             print("  Aviso: Colunas restantes no transform não batem exatamente com as esperadas no fit.")
             # O importante é que as colunas a serem dropadas foram removidas.

        return X_transformed

    def get_feature_names_out(self, input_features=None):
        """Retorna os nomes das features após a transformação."""
        check_is_fitted(self, '_is_fitted')
        # Retorna a lista de features mantidas, calculada no fit
        return np.array(self.features_kept_, dtype=object)

    def get_params(self, deep=True):
        return {
            "variance_threshold": self.variance_threshold,
            "select_identical": self.select_identical,
            "missing_threshold": self.missing_threshold,
            "correlation_threshold": self.correlation_threshold
        }

    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self


# ================================================================================================
#   --- Extract number episode ---

ep_extract = ExtractEpisodeNumber(column_name='Episode_Title', new_column_name='Episode_Number')
X_fe = ep_extract.fit_transform(train[features])
test_fe = ep_extract.transform(test[features])
# ================================================================================================
# --- Simple imputer ---
imputation_map = {
    'Episode_Length_minutes': 'mean',
    'Guest_Popularity_percentage': 0,
    'Number_of_Ads':0
}

imputer = CustomImputer(imputation_map=imputation_map)
X_fe = imputer.fit_transform(X_fe)
test_fe = imputer.transform(test_fe)
# ================================================================================================
# --- FE Combinations on Numerics ---
f_combiner = FeatureCombiner()
X_fe = f_combiner.fit_transform(X_fe)
test_fe = f_combiner.transform(test_fe)

# --- FE Frequency---
cols_to_freq_encode = ['Publication_Day']

freq_encoder = FrequencyEncoder(columns_to_encode=cols_to_freq_encode,
                                suffix='_freq',
                                handle_unseen=0, # Novas categorias terão frequência 0
                                drop_original=False) # Manter colunas originais
X_fe = freq_encoder.fit_transform(X_fe)
test_fe = freq_encoder.transform(test_fe)

# --- Strategic FE ---
strategic_fe = StrategicFeatureEngineer()
X_fe = strategic_fe.fit_transform(X_fe)
test_fe = strategic_fe.transform(test_fe)

# ================================================================================================
# --- Encoding ---

episode_sentiment_map = {
    'Positive':3, 
    'Neutral':2, 
    'Negative':1
}

publication_time_map = {
    'Morning':1, 
    'Afternoon':2, 
    'Evening':3,
    'Night':4
}

encoding_estrategy_map = {
    'Podcast_Name':'onehot',
    'Genre':'onehot',
    'Publication_Day':'onehot', # Onehot dias da semana
    'Publication_Time':publication_time_map,
    'Episode_Sentiment':episode_sentiment_map,
    'Genre_Day':'onehot',
    'Genre_Time':'onehot',
    'Genre_Sentiment':'onehot',
    'Day_Time_Combo':'onehot',
    'HasGuest_x_Genre':'onehot',
    'HasGuest_x_Day':'onehot'
}
encoder = PandasCompatEncoder(encoding_map=encoding_estrategy_map)
X_fe = encoder.fit_transform(X_fe)
test_fe = encoder.transform(test_fe)
# ================================================================================================
# --- Seletor ---
"""
selector = BasicFeatureSelector(
    variance_threshold=0.01,     # Remove 'constante' e 'quase_const'
    select_identical=True,      # Remove 'col_identica'
    missing_threshold=0.85,     # Remove 'muito_nan_acima'
    correlation_threshold=0.95  # Remove 'altamente_corr' (ou 'normal_num')
)
X_fe = selector.fit_transform(X_fe)
test_fe = selector.transform(test_fe)
"""
#features_to_drop = ['Episode_Sentiment_mapped', 'HasGuest_x_Day_ohe_Guest1_Friday', 'HasGuest_x_Day_ohe_Guest1_Monday', 'HasGuest_x_Day_ohe_Guest1_Saturday', 'HasGuest_x_Day_ohe_Guest1_Sunday', 'HasGuest_x_Day_ohe_Guest1_Thursday', 'HasGuest_x_Day_ohe_Guest1_Tuesday', 'HasGuest_x_Day_ohe_Guest1_Wednesday', 'HasGuest_x_Genre_ohe_Guest1_Business', 'HasGuest_x_Genre_ohe_Guest1_Comedy', 'HasGuest_x_Genre_ohe_Guest1_Education', 'HasGuest_x_Genre_ohe_Guest1_Health', 'HasGuest_x_Genre_ohe_Guest1_Lifestyle', 'HasGuest_x_Genre_ohe_Guest1_Music', 'HasGuest_x_Genre_ohe_Guest1_News', 'HasGuest_x_Genre_ohe_Guest1_Sports', 'HasGuest_x_Genre_ohe_Guest1_Technology', 'HasGuest_x_Genre_ohe_Guest1_True Crime']
#X_fe = X_fe.copy().drop(features_to_drop, axis=1)
#test_fe = test_fe.copy().drop(features_to_drop, axis=1)

best_features = [
    'ELen_Int', 'Episode_Length_minutes', 'Ad_Density', 'SinEpLen', 'CosEpLen', 'all4_prod', 
    'Episode_Length_minutespow2_x_Host_Popularity_percentagepow2_x_Episode_Number',
    '(Episode_Length_minutesxHost_Popularity_percentage)_div_(Guest_Popularity_percentage+Episode_Number)', 
    'Host_Popularity_percentagepow2_minus_Episode_Numberpow2', '(Episode_Length_minutes_plus_Episode_Number)_div_Host_Popularity_percentage', 
    'Guest_Popularity_percentagepow2_div_Episode_Numberpow1', 'Guest_Popularity_percentage_plus_Episode_Number',
    'Host_Popularity_percentage', 'Episode_Number', 'Guest_Popularity_percentage', 'Podcast_Name_encoded', 
    'Sentiment_Numeric', 'ELen_Dec', 'Day_Time_Combo_encoded', 'Number_of_Ads_0', 'Host_Popularity_percentagepow3_div_Episode_Length_minutespow2',
    'Episode_Numberpow3_div_Episode_Length_minutespow1', 'Episode_Length_minutespow1_x_Guest_Popularity_percentagepow2', 
    'Episode_Length_minutespow2_div_Host_Popularity_percentagepow1', 'CosTime', 'Episode_Length_minutespow2_div_Host_Popularity_percentagepow3', 'Genre_Health'
]
X_fe_final = X_fe.copy()[best_features]
test_fe_final = test_fe.copy()[best_features]
# ================================================================================================


print(X_fe_final.shape)
print(test_fe_final.shape)


import xgboost as xgb
import optuna
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.metrics import make_scorer, mean_squared_error, log_loss, roc_auc_score, accuracy_score
from sklearn.base import clone
import numpy as np
import pandas as pd
import time
import warnings
import os

# --- Funções auxiliares de métricas ---
def rmse(y_true, y_pred):
    """Root Mean Squared Error"""
    if np.any(np.isnan(y_pred)) or np.any(np.isinf(y_pred)):
        return np.inf
    return np.sqrt(mean_squared_error(y_true, y_pred))

# --- Função Principal de Tuning CORRIGIDA ---

def tune_xgboost_with_early_stopping(
    X: pd.DataFrame,
    y: pd.Series,
    model_type: str = 'regression', # 'regression' ou 'classification'
    n_trials: int = 50,
    cv_folds: int = 5,
    objective_metric: str = 'auto',
    # param_space: dict = None, # REMOVIDO por enquanto para simplificar - as sugestões estão agora dentro do objective
    use_gpu: bool = False,
    random_state: int = 42,
    timeout: int = None,
    early_stopping_rounds: int = None,
    verbose_cv: bool = False
) -> tuple:
    """
    Realiza a otimização de hiperparâmetros para XGBoost usando Optuna,
    Validação Cruzada Manual e Early Stopping dentro de cada fold.
    (CORRIGIDO para definir sugestões dentro da função objective)

    Args:
        X: DataFrame de features para treino.
        y: Series do target para treino.
        model_type: Tipo de problema ('regression' ou 'classification').
        n_trials: Número de combinações de hiperparâmetros a serem testadas pelo Optuna.
        cv_folds: Número de folds para a validação cruzada.
        objective_metric: Métrica a ser otimizada.
        use_gpu: Se True, tenta usar a GPU.
        random_state: Seed para reprodutibilidade.
        timeout: Tempo máximo (em segundos) para a execução do tuning.
        early_stopping_rounds: Número de rounds para early stopping DENTRO de cada fold do CV.
        verbose_cv: Se True, imprime detalhes de cada fold durante o trial.

    Returns:
        tuple: Contendo:
            - best_params (dict): Melhores hiperparâmetros encontrados.
            - avg_best_iteration (float or None): Média das melhores iterações do melhor trial.
    """
    start_time = time.time()

    # --- Determinar Métrica, Scorer e Direção da Otimização ---
    # (Mesma lógica de antes para determinar métrica e direção)
    is_regression = model_type == 'regression'
    if objective_metric == 'auto':
        if is_regression:
            objective_metric = 'rmse'
        else: # classification
            if y.nunique() == 2:
                objective_metric = 'logloss'
            else:
                objective_metric = 'accuracy'

    optimization_direction = 'minimize'
    xgb_eval_metric = None
    scoring_func = None

    if objective_metric == 'rmse':
        optimization_direction = 'minimize'
        xgb_eval_metric = 'rmse'
        scoring_func = rmse
    elif objective_metric == 'mae':
        optimization_direction = 'minimize'
        xgb_eval_metric = 'mae'
        from sklearn.metrics import mean_absolute_error
        scoring_func = mean_absolute_error
    elif objective_metric == 'logloss':
        optimization_direction = 'minimize'
        xgb_eval_metric = 'logloss' if y.nunique() == 2 else 'mlogloss'
        scoring_func = log_loss
    elif objective_metric == 'auc':
        if is_regression or y.nunique() != 2:
             raise ValueError("AUC só é válida para classificação binária.")
        optimization_direction = 'maximize'
        xgb_eval_metric = 'auc'
        scoring_func = roc_auc_score
    elif objective_metric == 'accuracy':
        if is_regression:
             raise ValueError("Accuracy só é válida para classificação.")
        optimization_direction = 'maximize'
        xgb_eval_metric = 'merror' if y.nunique() > 2 else 'error'
        scoring_func = accuracy_score
    else:
        raise ValueError(f"Métrica objetiva '{objective_metric}' não reconhecida.")

    print(f"Iniciando tuning para {model_type}...")
    print(f"Otimizando métrica: {objective_metric} (Optuna direction: {optimization_direction})")
    if xgb_eval_metric:
        print(f"XGBoost internal eval_metric for early stopping: {xgb_eval_metric}")
    print(f"Número de trials: {n_trials}, Folds CV: {cv_folds}")
    if early_stopping_rounds:
        print(f"Early stopping ativado com {early_stopping_rounds} rounds.")

    # --- Função Objetivo para Optuna (com CV manual) ---
    def objective(trial: optuna.trial.Trial) -> float:

        # <<< CORREÇÃO: Sugestões de parâmetros MOVIDAS PARA DENTRO do objective >>>
        params = {
            'objective': 'reg:squarederror' if is_regression else ('binary:logistic' if y.nunique()==2 else 'multi:softmax'),
            'random_state': random_state,
            'device': 'cuda' if use_gpu else 'cpu', # Definir tree_method aqui
            # --- Hiperparâmetros a serem otimizados ---
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.06, log=True),
            'max_depth': trial.suggest_int('max_depth', 11, 14),
            'subsample': trial.suggest_float('subsample', 0.5, 1.0, step=0.05),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0, step=0.05),
            'gamma': trial.suggest_float('gamma', 0, 5, step=0.1),
            'lambda': trial.suggest_float('lambda', 1e-8, 10.0, log=True), # L2
            'alpha': trial.suggest_float('alpha', 1e-8, 10.0, log=True), # L1
            'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
            'early_stopping_rounds': early_stopping_rounds
        }

        # Adicionar parâmetros específicos de classificação/GPU se necessário
        if not is_regression:
            params['use_label_encoder'] = False
            if y.nunique() > 2:
                params['num_class'] = y.nunique()
            # Adicionar scale_pos_weight se desejado (ex: para dados desbalanceados)
            # if y.nunique() == 2:
            #     # Exemplo: sugerir baseado na proporção ou deixar fixo/otimizar
            #     scale_val = trial.suggest_float('scale_pos_weight', 1, sum(y==0)/sum(y==1) * 1.5)
            #     params['scale_pos_weight'] = scale_val


        # Adiciona eval_metric e n_estimators alto se early stopping for usado
        if early_stopping_rounds and xgb_eval_metric:
            params['eval_metric'] = xgb_eval_metric
            params['n_estimators'] = 5000 # Valor alto, será limitado pelo early stopping
        else:
            # Se não usar early stopping, precisamos definir n_estimators
            params['n_estimators'] = trial.suggest_int('n_estimators', 100, 2000, step=100) # Otimizar n_estimators


        fold_scores = []
        fold_best_iterations = []

        # --- Configurar CV ---
        if is_regression:
            cv = KFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
        else:
            cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)

        if verbose_cv: print(f"--- Trial {trial.number} Params: { {k: f'{v:.4f}' if isinstance(v, float) else v for k,v in params.items()} } ---")

        # --- Loop de Validação Cruzada Manual ---
        for fold, (train_idx, valid_idx) in enumerate(cv.split(X, y)):
            X_train_fold, X_valid_fold = X.iloc[train_idx], X.iloc[valid_idx]
            y_train_fold, y_valid_fold = y.iloc[train_idx], y.iloc[valid_idx]

            # --- Configurar Modelo XGBoost para o Fold ---
            model_fold = None
            try:
                if is_regression:
                    model_fold = xgb.XGBRegressor(**params)
                else:
                    model_fold = xgb.XGBClassifier(**params)

                # --- Treinamento com Early Stopping (se aplicável) ---
                fit_params = {}
                if early_stopping_rounds:
                    fit_params['eval_set'] = [(X_valid_fold, y_valid_fold)]
                    fit_params['verbose'] = False

                model_fold.fit(X_train_fold, y_train_fold, **fit_params)

                # --- Capturar Melhor Iteração ---
                best_iter = getattr(model_fold, 'best_iteration', params.get('n_estimators'))
                if best_iter is not None:
                    fold_best_iterations.append(best_iter)
                else: # Caso fallback se best_iteration não for encontrado
                     fold_best_iterations.append(params.get('n_estimators'))

                # --- Previsão e Score do Fold ---
                # (Mesma lógica de antes para predição e cálculo do score)
                if is_regression or (not is_regression and objective_metric not in ['logloss', 'auc']):
                     preds = model_fold.predict(X_valid_fold)
                else:
                     preds_proba = model_fold.predict_proba(X_valid_fold)
                     if objective_metric == 'auc':
                          preds = preds_proba[:, 1]
                     elif objective_metric == 'logloss':
                          preds = preds_proba

                # Lidar com possíveis NaNs/Infs nas predições antes de calcular score
                if np.any(np.isnan(preds)) or np.any(np.isinf(preds)):
                    if verbose_cv: print(f"  Fold {fold+1}/{cv_folds}: ERRO - NaN/Inf nas predições.")
                    # Pula este fold ou retorna score ruim? Vamos pular para média
                    continue # Ou: fold_scores.append(np.inf if optimization_direction == 'minimize' else -np.inf)

                fold_score = scoring_func(y_valid_fold, preds)
                fold_scores.append(fold_score)

                if verbose_cv:
                     iter_info = f", Best Iter: {best_iter}" if best_iter is not None else f", Iters: {params.get('n_estimators')}"
                     print(f"  Fold {fold+1}/{cv_folds}: Score={fold_score:.6f}{iter_info}")

            except Exception as e:
                if verbose_cv: print(f"  Fold {fold+1}/{cv_folds}: ERRO - {e}")
                # Se um fold falhar gravemente, penaliza o trial inteiro
                import traceback
                # traceback.print_exc() # Descomentar para debug detalhado
                return np.inf if optimization_direction == 'minimize' else -np.inf


        # --- Calcular Média do Score e das Iterações para o Trial ---
        if not fold_scores: # Se todos os folds falharam ou foram pulados
             if verbose_cv: print(f"--- Trial {trial.number} Falhou (sem scores válidos) ---")
             return np.inf if optimization_direction == 'minimize' else -np.inf

        avg_score = np.mean(fold_scores)
        # Calcula a média apenas se houver iterações (early stopping foi usado)
        avg_best_iter = np.mean(fold_best_iterations) if fold_best_iterations and early_stopping_rounds else None

        if avg_best_iter is not None:
            trial.set_user_attr('average_best_iteration', avg_best_iter)
        if verbose_cv:
             iter_str = f", Avg Best Iter={avg_best_iter:.1f}" if avg_best_iter is not None else ""
             print(f"--- Trial {trial.number} Summary: Avg Score={avg_score:.6f}{iter_str} ---")

        # Lidar com NaN/Inf no score final do trial (segurança)
        if np.isnan(avg_score) or np.isinf(avg_score):
            return np.inf if optimization_direction == 'minimize' else -np.inf

        return avg_score

    # --- Executar Otimização Optuna ---
    study = optuna.create_study(direction=optimization_direction,
                                sampler=optuna.samplers.TPESampler(seed=random_state))
    try:
        # Avisar sobre n_jobs > 1 com GPU (pode causar problemas)
        num_jobs = 1 # Default para segurança com CV manual e GPU
        if use_gpu and os.cpu_count() > 1: # os não importado, mas como exemplo
             print("Aviso: Usando n_jobs=1 para otimização Optuna com use_gpu=True para evitar conflitos de recursos.")
        # elif not use_gpu:
        #      num_jobs = -1 # Usar todos os cores se não usar GPU

        study.optimize(objective, n_trials=n_trials, timeout=timeout, n_jobs=num_jobs)
    except KeyboardInterrupt:
        print("\nTuning interrompido pelo usuário.")
    except Exception as e:
        print(f"\nErro durante a otimização: {e}")
        import traceback
        traceback.print_exc()


    # --- Resultados ---
    elapsed_time = time.time() - start_time
    print(f"\nTuning concluído em {elapsed_time:.2f} segundos.")

    best_params_final = {}
    avg_best_iteration_final = None

    if not study.trials:
         print("Nenhum trial foi completado.")
         return best_params_final, avg_best_iteration_final

    completed_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    if not completed_trials:
        print("Nenhum trial foi completado com SUCESSO.")
        # Tenta pegar o melhor mesmo assim, pode ser um trial com erro mas que retornou valor
        try:
            best_trial = study.best_trial
            print(f"Melhor trial (pode ter tido erro/score ruim): Número {best_trial.number}, Valor: {best_trial.value}")
        except ValueError: # Caso nenhum trial tenha valor
             print("Não foi possível encontrar o melhor trial.")
        return best_params_final, avg_best_iteration_final


    try:
        best_trial = study.best_trial
        # Pega os parâmetros que foram efetivamente sugeridos e usados no melhor trial
        best_params_final = best_trial.params
        best_value = best_trial.value

        print(f"Melhor score ({objective_metric}, CV médio): {best_value:.6f} (Trial {best_trial.number})")
        print("Melhores hiperparâmetros encontrados:")
        for key, value in best_params_final.items():
            print(f"  {key}: {value}")

        # Recupera a média das iterações do melhor trial
        avg_best_iteration_final = best_trial.user_attrs.get('average_best_iteration', None)

        # Definir n_estimators final baseado no early stopping ou otimização
        if avg_best_iteration_final is not None:
             final_n_estimators = int(np.round(avg_best_iteration_final))
             print(f"\nMédia das melhores iterações (CV) para o melhor trial: {avg_best_iteration_final:.1f}")
             print(f"  -> Sugestão para n_estimators no treino final: {final_n_estimators}")
        elif 'n_estimators' in best_params_final: # Se n_estimators foi otimizado (sem early stopping)
             final_n_estimators = best_params_final['n_estimators']
             print(f"\nUsando n_estimators otimizado (sem early stopping): {final_n_estimators}")
        else: # Fallback se nem early stopping nem otimização de n_estimators ocorreu
             final_n_estimators = 500 # Ou outro default
             print(f"\nAviso: n_estimators não determinado. Usando default: {final_n_estimators}")


        # --- Montar dicionário final de parâmetros para o modelo ---
        # Começa com os parâmetros otimizados do melhor trial
        final_model_params = best_trial.params.copy()
        final_model_params['n_estimators'] = final_n_estimators # Adiciona n_estimators ajustado

        # Adiciona/Sobrescreve parâmetros fixos importantes
        final_model_params['random_state'] = random_state
        final_model_params['device'] = 'cuda' if use_gpu else 'cpu'
        if is_regression:
             final_model_params['objective'] = 'reg:squarederror'
        else:
             final_model_params['objective'] = 'binary:logistic' if y.nunique()==2 else 'multi:softmax'
             final_model_params['use_label_encoder'] = False
             if y.nunique() > 2:
                  final_model_params['num_class'] = y.nunique()

        # Remover parâmetros que não são de __init__ ou que foram auxiliares
        final_model_params.pop('eval_metric', None)

        # Retorna os parâmetros prontos para instanciar o modelo final
        best_params_final = final_model_params


    except Exception as e:
        print(f"Erro ao obter os resultados do melhor trial: {e}")
        import traceback
        traceback.print_exc()
        print("Verifique se algum trial foi concluído com sucesso.")
        # Tentar retornar os parâmetros do melhor trial mesmo com erro na pós-análise
        try:
            best_params_final = study.best_trial.params
        except:
            best_params_final = {}


    return best_params_final, avg_best_iteration_final



best_params_reg, avg_iter_reg = tune_xgboost_with_early_stopping(
    X=X_fe_final,
    y=train[target],
    model_type='regression',
    n_trials=35,
    cv_folds=6,
    objective_metric='rmse',
    early_stopping_rounds=100,
    random_state=rs_seed,
    verbose_cv=True, 
    use_gpu=True,
)
print("\nMelhores parâmetros para Regressão:", best_params_reg)


from sklearn.model_selection import KFold
from sklearn.base import clone, is_regressor, BaseEstimator
from sklearn.metrics import mean_squared_error # Ou sua métrica de score
import time
import warnings # Para controlar avisos, se necessário

# --- Defina sua função de scoring aqui (ex: rmse) ---
def rmse(y_true, y_pred):
    """Calcula o Root Mean Squared Error."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    if np.any(np.isnan(y_pred)) or np.any(np.isinf(y_pred)):
        # print("Aviso: NaNs ou Infs encontrados nas previsões. Retornando infinito para RMSE.")
        return np.inf # Retorna infinito se houver inválidos
    # Verifica se y_true tem variância zero para evitar erro no MSE se tudo for constante
    if np.var(y_true) == 0 and np.all(y_true == y_pred):
        return 0.0
    # Adiciona verificação para evitar erro se y_pred for constante e diferente de y_true constante
    if len(np.unique(y_pred)) == 1 and len(np.unique(y_true)) > 1:
         # Ou se ambos constantes mas diferentes
         pass # Deixa o mean_squared_error lidar, mas pode dar warning
    # Tratar caso onde todos os y_true são iguais, mas y_pred varia
    # O mean_squared_error deve funcionar corretamente aqui.

    # Previne erro se y_pred for infinito ou NaN após verificações iniciais (segurança)
    y_pred = np.nan_to_num(y_pred, nan=np.nan, posinf=np.finfo(np.float64).max, neginf=np.finfo(np.float64).min)
    # Re-checa NaN após nan_to_num, pois ele pode não ter substituído se já era NaN
    if np.any(np.isnan(y_pred)):
        return np.inf

    try:
        return np.sqrt(mean_squared_error(y_true, y_pred))
    except ValueError as e:
        print(f"Erro no cálculo do RMSE: {e}")
        print(f"y_true (tipo {type(y_true)}, shape {y_true.shape}): {y_true[:5]}...")
        print(f"y_pred (tipo {type(y_pred)}, shape {y_pred.shape}): {y_pred[:5]}...")
        return np.inf

# ----------------------------------------------------

def cross_validate_and_predict_simple(
    model: BaseEstimator,
    X: pd.DataFrame,
    y: pd.Series,
    X_test: pd.DataFrame, # << ADICIONADO: Precisa do conjunto de teste
    cv: KFold,
    scoring_func=rmse,    # << ADICIONADO: Função de score
    verbose: bool = True,
    scaler: bool = False,
    scale_epsilon: float = 1e-9 # << ADICIONADO: Tolerância para std próximo de zero
) -> tuple:
    """
    Realiza validação cruzada KFold simples, com padronização por fold e
    tratamento de desvio padrão zero/próximo de zero.
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
        scale_epsilon: Tolerância para considerar o desvio padrão como zero
                       durante a padronização.

    Returns:
        tuple: Contendo:
            - oof_preds (pd.Series): Previsões OOF para o conjunto de treino,
                                     com o índice original de X.
            - avg_test_preds (np.ndarray): Previsões médias para o conjunto de teste.
            - scores (list): Lista de scores (um por fold). Retorna np.inf se
                             ocorrer erro no fold ou score inválido.
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
    # Faz a cópia aqui para evitar modificar o X_test original fora da função
    X_test_processed = X_test[X.columns].copy()

    oof_preds = np.zeros(len(X)) * np.nan # Inicializa OOF com NaN
    test_preds_list = [] # Armazena previsões de teste de cada fold
    scores = []

    if verbose: print(f"  Iniciando CV Simples ({cv.get_n_splits()} folds) com Predições e Scaling por Fold...")

    for fold, (train_idx, valid_idx) in enumerate(cv.split(X, y)):
        start_fold_time = time.time()
        if verbose: print(f"    --- Fold {fold + 1}/{cv.get_n_splits()} ---")

        # --- Divisão ---
        # Cria cópias para evitar SettingWithCopyWarning e modificar o X/y originais do loop
        X_train_fold = X.iloc[train_idx].copy()
        X_valid_fold = X.iloc[valid_idx].copy()
        y_train_fold, y_valid_fold = y.iloc[train_idx].copy(), y.iloc[valid_idx].copy()
        # Cria cópia do X_test para este fold específico
        X_test_fold = X_test_processed.copy()
        
        if scaler:
            # --- Scaling e Tratamento de NaN/Inf ---
            if verbose: print("      Aplicando scaling (StandardScaler) e tratando NaNs...")
            features_to_scale = X_train_fold.columns.tolist() # Ou selecione as colunas numéricas
    
            for feat in features_to_scale:
                # Calcula mean e std SOMENTE no treino do fold
                mean = X_train_fold[feat].mean()
                std = X_train_fold[feat].std()
    
                # VERIFICAÇÃO E TRATAMENTO DE STD PRÓXIMO DE ZERO
                if abs(std) > scale_epsilon:
                    # Aplica scaling normal se std for suficientemente grande
                    X_train_fold[feat] = (X_train_fold[feat] - mean) / std
                    X_valid_fold[feat] = (X_valid_fold[feat] - mean) / std
                    X_test_fold[feat] = (X_test_fold[feat] - mean) / std
                else:
                    # Se std é zero ou muito pequeno, a feature é constante no treino.
                    # O valor escalado deve ser 0.
                    if verbose and abs(std) > 0: # Informa se era *próximo* de zero, mas não zero exato
                         print(f"      Aviso: Desvio padrão de '{feat}' no fold {fold + 1} é próximo de zero ({std:.2e}). Definindo valores escalados como 0.")
                    elif verbose and abs(std) == 0:
                         print(f"      Info: Desvio padrão de '{feat}' no fold {fold + 1} é zero. Definindo valores escalados como 0.")
    
                    # Define como 0 para todos os conjuntos, baseado na estatística do treino
                    X_train_fold[feat] = 0.0
                    X_valid_fold[feat] = (X_valid_fold[feat] - mean) # Subtrai a média (que é o próprio valor constante) resultando em 0
                    X_valid_fold[feat] = 0.0 # Simplificando, já que std=0 implica X_valid=mean
                    X_test_fold[feat] = (X_test_fold[feat] - mean) # Subtrai a média do treino
                    # Se std=0, não podemos dividir. Se a feature em X_test for igual à média do treino, o resultado é 0.
                    # Se for diferente, a padronização é indefinida. Definir como 0 é uma escolha comum.
                    # Uma alternativa seria manter o valor original ou usar um valor específico, mas 0 é mais simples.
                    X_test_fold[feat] = 0.0 # Define como 0 por consistência.
    
    
                # Tratamento de NaN (após scaling, caso algum NaN original exista)
                # Se o scaling foi feito, NaNs permanecem NaNs. Se foi setado para 0, não haverá NaNs.
                # O fillna(0) garante que qualquer NaN *original* ou resultante de 0/0 (embora evitado acima) seja tratado.
                X_train_fold[feat] = X_train_fold[feat].fillna(0)
                X_valid_fold[feat] = X_valid_fold[feat].fillna(0)
                X_test_fold[feat] = X_test_fold[feat].fillna(0)
    
                # Verificação final por segurança (opcional, mas bom para debug)
                if np.any(np.isinf(X_train_fold[feat])) or np.any(np.isinf(X_valid_fold[feat])) or np.any(np.isinf(X_test_fold[feat])):
                    print(f"      ALERTA: Infinito detectado em '{feat}' APÓS scaling/tratamento no fold {fold + 1}!")
                    # Poderia adicionar substituição aqui como último recurso, mas o ideal é evitar antes
                    X_train_fold[feat].replace([np.inf, -np.inf], 0, inplace=True)
                    X_valid_fold[feat].replace([np.inf, -np.inf], 0, inplace=True)
                    X_test_fold[feat].replace([np.inf, -np.inf], 0, inplace=True)


        # --- Treinamento ---
        model_fold = clone(model)
        try:
            if verbose: print("      Treinando modelo...")
            # Usa os dados processados do fold
            model_fold.fit(X_train_fold, y_train_fold)

            # --- Previsão OOF ---
            if verbose: print("      Prevendo no conjunto de validação (OOF)...")
            fold_oof_preds = model_fold.predict(X_valid_fold)
            # Verifica inf/nan nas previsões OOF antes de atribuir/calcular score
            if np.any(np.isinf(fold_oof_preds)) or np.any(np.isnan(fold_oof_preds)):
                print(f"      Aviso: Inf/NaN detectado nas PREVISÕES OOF do fold {fold+1}. Tratando como erro de score.")
                score = np.inf # Penaliza o fold
                # Não preenche OOF com predições inválidas, mantém NaN
            else:
                oof_preds[valid_idx] = fold_oof_preds
                # --- Score ---
                if verbose: print("      Calculando score...")
                score = scoring_func(y_valid_fold, fold_oof_preds)


            # --- Previsão Teste ---
            if verbose: print("      Prevendo no conjunto de teste...")
            fold_test_preds = model_fold.predict(X_test_fold)
            # Verifica inf/nan nas previsões de teste antes de adicionar à lista
            if np.any(np.isinf(fold_test_preds)) or np.any(np.isnan(fold_test_preds)):
                 print(f"      Aviso: Inf/NaN detectado nas PREVISÕES DE TESTE do fold {fold+1}. Substituindo por 0 nas previsões deste fold.")
                 # Substitui por 0 ou outra estratégia (e.g., média das outras previsões - complexo)
                 # Usar 0 é simples, mas pode enviesar a média final. Outra opção é descartar este fold da média de teste.
                 # Vamos substituir por NaN e tratar na média final.
                 fold_test_preds = np.where(np.isinf(fold_test_preds) | np.isnan(fold_test_preds), np.nan, fold_test_preds)


            # Adiciona previsões (possivelmente com NaNs tratados) e score
            test_preds_list.append(fold_test_preds)
            scores.append(score)

            fold_duration = time.time() - start_fold_time
            if verbose:
                score_str = f"{score:.6f}" if np.isfinite(score) else "Inf/NaN"
                print(f"      Score Fold {fold + 1}: {score_str} ({fold_duration:.2f}s)")


        except Exception as e:
            print(f"      ERRO no Fold {fold + 1}/{cv.get_n_splits()}: {e}")
            import traceback # Para mais detalhes do erro
            traceback.print_exc()
            scores.append(np.inf)
            # Não adiciona nada a test_preds_list neste caso de erro grave

    # --- Finalização ---
    # Calcula média das previsões de teste (ignorando NaNs introduzidos por folds com predições inválidas)
    if test_preds_list:
         test_preds_arrays = [np.asarray(preds) for preds in test_preds_list]
         # Usa nanmean para calcular a média ignorando os NaNs
         with warnings.catch_warnings(): # Suprime avisos de "mean of empty slice" se todos forem NaN para um ponto
             warnings.simplefilter("ignore", category=RuntimeWarning)
             avg_test_preds = np.nanmean(np.stack(test_preds_arrays, axis=0), axis=0)
         # Se algum valor ainda for NaN (porque TODOS os folds deram NaN para ele), substitui por 0 ou outra estratégia
         if np.any(np.isnan(avg_test_preds)):
              print("Aviso: Alguns valores médios de previsão de teste são NaN (todos os folds falharam/geraram NaN). Substituindo por 0.")
              avg_test_preds = np.nan_to_num(avg_test_preds, nan=0.0)

    else:
         print("Aviso: Nenhuma previsão de teste válida foi gerada.")
         avg_test_preds = np.zeros(len(X_test_processed)) * np.nan # Retorna NaNs se nada foi gerado

    # Cria Series OOF com o índice original
    oof_preds_series = pd.Series(oof_preds, index=original_index, name='oof_predictions')

    if verbose:
        # Calcula média dos scores finitos
        finite_scores = [s for s in scores if np.isfinite(s)]
        if finite_scores:
            mean_cv_score = np.mean(finite_scores)
            print(f"  Score CV Médio Final (folds finitos): {mean_cv_score:.6f}")
        else:
            print("  Score CV Médio Final: N/A (nenhum fold com score finito)")
        print(f"  Número de folds com score Inf/NaN: {len(scores) - len(finite_scores)}")
        print("-" * 20)

    if np.all(avg_test_preds == 0) or np.all(avg_test_preds == 1):
        # Esta condição pode ser muito específica, talvez remover ou ajustar
        pass

    return oof_preds_series, avg_test_preds, scores


%%time
import xgboost as xgb

y = train[target]

N_SPLITS = 10
rs_seed = 42
kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=rs_seed)

final_params = {
    'device':'cuda',
    'random_state':rs_seed,
}
final_params.update({'learning_rate': 0.01922275746199718, 'max_depth': 12, 'subsample': 0.8500000000000001, 'colsample_bytree': 0.75, 'gamma': 2.3000000000000003, 'lambda': 1.1744084029711715e-08, 'alpha': 0.00032126076915987595, 'min_child_weight': 7})
final_params['n_estimators'] = 2000

instance_model = xgb.XGBRegressor(**final_params)

oof_predictions, test_predictions, fold_scores = cross_validate_and_predict_simple(
    model=instance_model,
    X=X_fe_final,
    y=y,
    X_test=test_fe_final,
    cv=kf,
    scoring_func=rmse, # Passa a função RMSE
    verbose=True,
    scaler=False
)

print(f"\n--- Resultados Finais ---")
print("Shape das previsões OOF:", oof_predictions.shape)
print("Primeiras 5 previsões OOF:\n", oof_predictions.head())
print("\nShape das previsões de teste:", test_predictions.shape)
print("Primeiras 5 previsões de teste:\n", test_predictions[:5])
print("\nScores por fold:", fold_scores)
print(f"Score médio CV (RMSE): {np.mean(fold_scores):.5f}", f" +- {np.std(fold_scores):.5f}")


submission = pd.read_csv("/kaggle/input/playground-series-s5e4/sample_submission.csv")

submission[target] = test_predictions

display(submission.head())
display(submission.drop("id", axis=1).hist(bins=50))

name_model = "xgb_13f"

submission.to_csv(f"{name_model}_test_predictions.csv", index=False)
pd.DataFrame(
    {
        "id": train.id,
        target: oof_predictions.values
    }
).to_csv(f"{name_model}_oof_predictions.csv")

