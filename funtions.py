import numpy as np
import pandas as pd

def prepare_data(data, numeric_cols, bets_cols):
    htr = data['HTR'].astype('category').cat.codes
    home_teams = data['HomeTeam'].astype('category').cat.codes
    away_teams = data['AwayTeam'].astype('category').cat.codes
    
    X_with_bets = pd.concat([
        pd.DataFrame(home_teams, columns=['HomeTeam']),
        pd.DataFrame(away_teams, columns=['AwayTeam']),
        pd.DataFrame(htr, columns=['HTR']),
        data[numeric_cols].reset_index(drop=True),
        data[bets_cols].reset_index(drop=True)
    ], axis=1)
    
    X_without_bets = pd.concat([
        pd.DataFrame(home_teams, columns=['HomeTeam']),
        pd.DataFrame(away_teams, columns=['AwayTeam']),
        pd.DataFrame(htr, columns=['HTR']),
        data[numeric_cols].reset_index(drop=True)
    ], axis=1)
    
    return X_with_bets, X_without_bets


def LimpiarDF(df):
    incompleto = [item for item in titles if df[item].isna().any()]
    return df.drop(incompleto)