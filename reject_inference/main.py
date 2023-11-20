import pandas as pd
import numpy as np

class RejectInference:
    def __init__(self, 
                 data: pd.DataFrame,
                 target_column_future_name: str = None,
                 bad_rate_column: str = None,
                 prob_column: str = None,
                 application_vars: list = None):
        self.data = data
        self.bad_rate_column = bad_rate_column
        self.target_column_future_name = target_column_future_name
        self.prob_column = prob_column
        self.application_vars = application_vars

        self._aux_parceling()
        self._aux_monte_carlo_parceling()

    def augmentation(self,
                     flag_aprovado_column: str):

        df_AR = self.data.copy()

        # Define as classes de risco (decis) que serão usadas para o cálculo dos pesos
        bins = pd.qcut(df_AR[self.prob_column], q=10, retbins=True, labels=False)[1]
        labels = [i for i in range(1, (len(bins)))]
        df_AR['classes_risco_AR'] = pd.cut(df_AR[self.prob_column], bins=bins, labels=labels)

        classes_df = pd.crosstab(index=df_AR['classes_risco_AR'],
                                columns=df_AR['aprovacao']).reset_index().rename(columns={
                                    1: 'Aprovados', 
                                    0: 'Recusados'
                                })

        # Calcula o peso para cada decil (classe de risco)
        classes_df['weight'] = (classes_df['Aprovados']+classes_df['Recusados'])\
            /classes_df['Aprovados']

        # Cria o dataset de aprovados "aumentados" com seus respectivos pesos
        tmp = df_AR.loc[df_AR[flag_aprovado_column] == 1].drop(
            columns=flag_aprovado_column
        ).copy()

        augmented_df = tmp.merge(
            classes_df[['classes_risco_AR', 'weight']],
            how='left', on='classes_risco_AR'
        ).drop(columns='classes_risco_AR')

        return augmented_df

    
    def _aux_parceling(self,
                       row = None, 
                       fator_lambda: float = 1.0):
        # Pega a bad rate da faixa de risco que o rejeitado foi 
        # alocado pelo modelo dos aprovados
        if row is not None:
            prob_scorecardA = fator_lambda * row[self.bad_rate_column]

            # Gera número aleatório entre 0 e 1
            r = np.random.uniform(0, 1)

            # Atribui default baseado na distribuição de maus
            # e bons por faixa de risco
            if r < prob_scorecardA:
                row[self.target_column_future_name] = int(1)
            else:
                row[self.target_column_future_name] = int(0)

        return row



    def parceling(self,
                  fator_lambda: float = 1.0):
        
        inferidos = self.data.apply(self._aux_parceling, args=(fator_lambda,), axis=1)

        inferidos[self.target_column_future_name] = inferidos[self.target_column_future_name].astype(int)

        return inferidos
    

    def _aux_monte_carlo_parceling(self,
                                   row = None,
                                   fator_lambda: float = 1.0,
                                   num_simulacoes: int = 1):
        # Baseado no escore do modelo KGB:
        if row is not None:
            prob_scorecardA = fator_lambda * row[self.prob_column]

            performances_simuladas = []

            for i in range(num_simulacoes):
                r = np.random.uniform(0, 1)

                if r < prob_scorecardA:
                    performances_simuladas.append(1)
                else:
                    performances_simuladas.append(0)

            # Tira a média das performances simuladas
            row['infer'] = np.mean(performances_simuladas)

        return row
    
    def monte_carlo_parceling(self,
                              fator_lambda: float = 1.0,
                              num_simulacoes: int = 1):
        
        inferidos = self.data.apply(self._aux_monte_carlo_parceling,
                                    args=(fator_lambda, num_simulacoes,), axis=1)

        # Define o default inferido baseado na bad rate da faixa de risco 
        # mapeada pelo modelo KGB
        inferidos[self.target_column_future_name] = np.where(inferidos['infer'] \
            > inferidos[self.bad_rate_column], 1, 0)

        return inferidos
    

    def simple_augmentation(self,
                            expected_bad_rate: float):
        
        inferidos = self.data.copy()

        cutoff_threshold = np.percentile(self.data[self.prob_column], 100 * (1 - expected_bad_rate))

        inferidos[self.target_column_future_name] = np.where(inferidos[self.prob_column] > cutoff_threshold, 1, 0)

        return inferidos

        
    def fuzzy_augmentation(self,
                           approval_prob = 1.0,
                           fator_lambda: float = 1.0):
        # Verificação de formato do dataset fornecido
        unknown_cols = [coluna for coluna in self.data.columns if coluna not in self.application_vars]
        float_cols = [coluna for coluna in unknown_cols if self.data[coluna].dtype == 'float64']

        if (
            set(self.application_vars).issubset(self.data.columns) and
            len(self.data.columns == len(self.application_vars) + 2) and
            len(float_cols) == len(unknown_cols)
        ):
            print("Sucesso: Dataset com formato esperado: Variáveis do KGB Model + Probas 1/0")

            # Transforma cada rejeitado em 2 registros (bom/mau parcial) 
            # ponderado pela probabilidade de ser bom/mau provenientes do KGB model
            inferidos = pd.melt(self.data,
                                id_vars=self.application_vars,
                                var_name='prob_type',
                                value_name='weight')
            
            inferidos[self.target_column_future_name] = np.where(inferidos['prob_type'] == self.prob_column,
                                                                1, 0)
            
            inferidos.drop(columns='prob_type', inplace=True)

            inferidos.loc[inferidos[self.target_column_future_name] == 1, 'weight'] *= fator_lambda

            if isinstance(approval_prob, float):
                inferidos['weight'] = inferidos['weight'] * approval_prob
            
            elif isinstance(approval_prob, np.ndarray):
                p_approve = np.tile(approval_prob, 2)
                inferidos['weight'] = inferidos['weight'] * p_approve
            else:
                inferidos = None
                print("Tipo de parâmetro não suportado")
        else:
            print("Falha: O dataset não corresponde ao formato esperado: Variáveis do KGB Model + Probas 1/0")
            inferidos = None   

        return inferidos
