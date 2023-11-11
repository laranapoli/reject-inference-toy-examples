import pandas as pd
import numpy as np

class RejectInference:
    def __init__(self, 
                 data: pd.DataFrame,
                 target_column_future_name: str,
                 bad_rate_column: str = None,
                 bad_prob_column: str = None,
                 application_vars: list = None):
        self.data = data
        self.bad_rate_column = bad_rate_column
        self.target_column_future_name = target_column_future_name
        self.bad_prob_column = bad_prob_column
        self.application_vars = application_vars

        self._aux_parceling()
        self._aux_monte_carlo_parceling()

    
    def _aux_parceling(self,
                       row = None, 
                       fator_lambda: float = 1.0):
        # Pega a probabilidade da faixa de risco que o rejeitado foi 
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
        # Usa a probabilidade de ser mau do modelo dos aprovados
        # (não da faixa de risco)
        if row is not None:
            prob_scorecardA = fator_lambda * row[self.bad_prob_column]

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
        
        inferidos = self.data.apply(self._aux_monte_carlo_parceling, args=(fator_lambda, num_simulacoes,), axis=1)

        # Define o default inferido baseado na bad rate da faixa de risco mapeada pelo modelo A
        inferidos[self.target_column_future_name] = np.where(inferidos['infer'] > inferidos[self.bad_rate_column], 1, 0)

        return inferidos
    

    def simple_augmentation(self,
                            expected_bad_rate: float):
        
        inferidos = self.data.copy()

        cutoff_threshold = np.percentile(self.data[self.bad_prob_column], 100 * (1 - expected_bad_rate))

        inferidos[self.target_column_future_name] = np.where(inferidos[self.bad_prob_column] > cutoff_threshold, 1, 0)

        return inferidos

        
    def fuzzy_augmentation(self,
                           approval_prob = 1.0):
        
        inferidos = pd.melt(self.data,
                            id_vars=self.application_vars,
                            var_name='prob_type',
                            value_name='weight')
        
        inferidos[self.target_column_future_name] = np.where(inferidos['prob_type'] == self.bad_prob_column,
                                                            1, 0)
        
        inferidos.drop(columns='prob_type', inplace=True)

        if isinstance(approval_prob, float):
            inferidos['weight'] = inferidos['weight'] * approval_prob
        
        elif isinstance(approval_prob, np.ndarray):
            p_approve = np.tile(approval_prob, 2)
            inferidos['weight'] = inferidos['weight'] * p_approve
        else:
            inferidos = None
            print("Tipo de parâmetro não suportado")
        
        return inferidos
