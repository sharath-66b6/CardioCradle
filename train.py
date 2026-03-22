import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from imblearn.over_sampling import SMOTE
import pickle
from tqdm import tqdm
import time

class ModelTrainer:
    def __init__(self):
        self.numerical_features = ['bmi', 'physicalhealth', 'mentalhealth', 'sleeptime']
        self.categorical_features = ['smoking', 'alcoholdrinking', 'stroke', 'diffwalking', 
                                   'sex', 'agecategory', 'race', 'diabetic', 
                                   'physicalactivity', 'genhealth', 'asthma', 
                                   'kidneydisease', 'skincancer']
        self.models = {
            'logistic': (LogisticRegression, {'random_state': 42, 'max_iter': 1000}),
            'decision_tree': (DecisionTreeClassifier, {'random_state': 42, 'max_depth': 10}),
            'random_forest': (RandomForestClassifier, {'random_state': 42, 'n_estimators': 100}),
            'xgboost': (XGBClassifier, {'random_state': 42, 'n_estimators': 100}),
            'lightgbm': (LGBMClassifier, {'random_state': 42, 'n_estimators': 100})
        }
        
    def prepare_data(self, df):
        scaler = StandardScaler()
        X_num = scaler.fit_transform(df[self.numerical_features])
        
        cat_data = df[self.categorical_features].copy()
        for col in self.categorical_features:
            cat_data[col] = cat_data[col].str.lower().str.replace(' ', '_')
        
        dv = DictVectorizer(sparse=False)
        X_cat = dv.fit_transform(cat_data.to_dict(orient='records'))
        
        X = np.hstack([X_num, X_cat])
        
        return X, scaler, dv
    
    def train_model(self, X, y, model_class, model_params, use_smote=False):
        if use_smote:
            smote = SMOTE(random_state=42)
            X, y = smote.fit_resample(X, y)
        
        model = model_class(**model_params)
        model.fit(X, y)
        return model
    
    def train_all_models(self, df_train, y_train, df_test):
        X, scaler, dv = self.prepare_data(df_train)
        
        df_test.to_pickle('data/df_test.pkl')
        print("Saved test data")
        
        with tqdm(total=len(self.models) * 2) as pbar:
            for model_name, (model_class, params) in self.models.items():
                model = self.train_model(X, y_train, model_class, params, use_smote=False)
                self.save_model(model, scaler, dv, f'models/{model_name}_original.bin')
                pbar.update(1)
                
                model_smote = self.train_model(X, y_train, model_class, params, use_smote=True)
                self.save_model(model_smote, scaler, dv, f'models/{model_name}_smote.bin')
                pbar.update(1)
                
                time.sleep(0.1)
    
    def save_model(self, model, scaler, dv, filename):
        with open(filename, 'wb') as f_out:
            pickle.dump((dv, scaler, model), f_out)
        print(f"Saved model: {filename}")

def main():
    df = pd.read_csv('heart_2020_cleaned.csv')
    
    df.columns = df.columns.str.lower()
    for col in df.select_dtypes(['object']).columns:
        df[col] = df[col].str.lower().str.replace(' ', '_')
    
    df['heartdisease'] = (df['heartdisease'] == 'yes').astype(int)
    
    from sklearn.model_selection import train_test_split
    df_train, df_test = train_test_split(df, test_size=0.2, random_state=42)
    
    y_train = df_train.pop('heartdisease').values
    df_test_for_save = df_test.copy()
    y_test = df_test.pop('heartdisease').values
    
    import os
    os.makedirs('models', exist_ok=True)
    os.makedirs('data', exist_ok=True)
    
    trainer = ModelTrainer()
    trainer.train_all_models(df_train, y_train, df_test_for_save)

if __name__ == "__main__":
    main()