from scipy.stats import ks_2samp
import pandas as pd
from sklearn.metrics import roc_auc_score

class CreditMetrics:
    def __init__(self, 
                 data: pd.DataFrame, 
                 target_column: str,
                 inference_reference_column: str = None):
        self.data = data
        self.target_column = target_column
        self.inference_reference_column = inference_reference_column

    def get_ks(self,
               proba_column: str):
        ks = ks_2samp(self.data.loc[self.data[self.target_column] == 0][proba_column],
                      self.data.loc[self.data[self.target_column] == 1][proba_column])[0]
        
        return round(ks, 3)


    def generate_report(self,
                        y_train_true: pd.Series, 
                        y_train_score: pd.Series, 
                        y_test_true: pd.Series, 
                        y_test_score: pd.Series):
        # Calcula gini
        auc_train = roc_auc_score(y_train_true, y_train_score)
        auc_test = roc_auc_score(y_test_true, y_test_score)

        gini_train = 100 * 2 * (auc_train - 0.5)
        gini_test = 100 * 2 * (auc_test - 0.5)
        delta_gini = gini_train - gini_test

        # Calcula bad rate total (inferidos + KGB)
        overall_bad_rate = 100 * self.data[self.target_column].mean()

        if self.inference_reference_column is None:
            odds_ratio = 'NA'
        else:
            # Calcula a relação KGB odds / Inf. GB odds
            kgb_goods = self.data.loc[self.data[self.inference_reference_column] == 'kgb'][self.target_column].value_counts().get(0, 0)
            kbg_bads = self.data.loc[self.data[self.inference_reference_column] == 'kgb'][self.target_column].value_counts().get(1, 0)

            infer_goods = self.data.loc[self.data[self.inference_reference_column] == 'infer'][self.target_column].value_counts().get(0, 0)
            infer_bads = self.data.loc[self.data[self.inference_reference_column] == 'infer'][self.target_column].value_counts().get(1, 0)

            odds_ratio = (kgb_goods/kbg_bads) / (infer_goods/infer_bads)

        # Summary
        result = {
            'Overall Bad Rate': [overall_bad_rate],
            'Known GB Odds/Inferred GB Odds': [odds_ratio],
            'Gini Dev': [gini_train],
            'Gini Test': [gini_test],
            'Overfitting (gini_dev - gini_test)': [delta_gini]
        }

        return pd.DataFrame(result).round(2)
