import streamlit as st
import pandas as pd
import numpy as np
import pickle
from pathlib import Path
import os

class HeartDiseasePredictor:
    def __init__(self):
        self.load_data()
        self.setup_page()

    def load_data(self):
        try:
            self.df_test = pd.read_pickle('data/df_test.pkl')
            self.models_path = Path('models')
            self.available_models = list(self.models_path.glob('*.bin'))
            if not self.available_models:
                st.error("No model files found in models/ directory. Please train models first.")
                st.stop()
        except Exception as e:
            st.error(f"Error loading data: {str(e)}")
            st.stop()

    def setup_page(self):
        st.title('Heart Disease Prediction App')
        st.sidebar.title('Model Selection')

        if not self.available_models:
            st.sidebar.error("No models available")
            return

        self.selected_model_path = st.sidebar.selectbox(
            'Choose a model:',
            options=self.available_models,
            format_func=lambda x: x.stem
        )

        model_parts = self.selected_model_path.stem.split('_')
        model_type = model_parts[0]
        is_smote = 'smote' in self.selected_model_path.stem

        st.sidebar.markdown(f"""
        **Selected Model Info:**
        - **Type:** {model_type.upper()}
        - **SMOTE:** {'Yes' if is_smote else 'No'}
        - **File:** {self.selected_model_path.name}
        """)

        # Input mode selection
        self.input_mode = st.sidebar.radio(
            "Input Mode:",
            ["Sample from Test Data", "Manual Input"]
        )

    def load_model(self, model_path):
        try:
            with open(model_path, 'rb') as f:
                return pickle.load(f)
        except Exception as e:
            st.error(f"Error loading model: {str(e)}")
            return None, None, None

    def create_sample_input(self):
        st.header('Sample Patient Data')

        test_index = st.number_input(
            'Select test set index:',
            min_value=0,
            max_value=len(self.df_test)-1,
            value=0,
            help="Choose a sample from the test dataset"
        )

        selected_patient = self.df_test.iloc[test_index].to_dict()

        st.subheader('Patient Details:')
        col1, col2 = st.columns(2)

        with col1:
            st.metric("BMI", f"{selected_patient['bmi']:.1f}")
            st.metric("Physical Health", f"{selected_patient['physicalhealth']:.1f}")
            st.metric("Mental Health", f"{selected_patient['mentalhealth']:.1f}")
            st.metric("Sleep Time", f"{selected_patient['sleeptime']:.1f}")

        with col2:
            st.metric("Age Category", selected_patient['agecategory'])
            st.metric("Sex", selected_patient['sex'])
            st.metric("Smoking", selected_patient['smoking'])
            st.metric("General Health", selected_patient['genhealth'])

        return selected_patient

    def create_manual_input(self):
        st.header('Manual Patient Input')

        col1, col2 = st.columns(2)

        numerical_features = {}
        with col1:
            st.subheader('Numerical Features')
            numerical_features['bmi'] = st.number_input('BMI:', min_value=10.0, max_value=50.0, value=25.0, step=0.1)
            numerical_features['physicalhealth'] = st.number_input('Physical Health (days):', min_value=0, max_value=30, value=0)
            numerical_features['mentalhealth'] = st.number_input('Mental Health (days):', min_value=0, max_value=30, value=0)
            numerical_features['sleeptime'] = st.number_input('Sleep Time (hours):', min_value=1, max_value=24, value=8)

        categorical_features = {}
        with col2:
            st.subheader('Categorical Features')

            categorical_features['smoking'] = st.selectbox('Smoking:', ['yes', 'no'])
            categorical_features['alcoholdrinking'] = st.selectbox('Alcohol Drinking:', ['yes', 'no'])
            categorical_features['stroke'] = st.selectbox('Previous Stroke:', ['yes', 'no'])
            categorical_features['diffwalking'] = st.selectbox('Difficulty Walking:', ['yes', 'no'])
            categorical_features['sex'] = st.selectbox('Sex:', ['female', 'male'])
            categorical_features['agecategory'] = st.selectbox('Age Category:', [
                '18-24', '25-29', '30-34', '35-39', '40-44', '45-49',
                '50-54', '55-59', '60-64', '65-69', '70-74', '75-79', '80_or_older'
            ])
            categorical_features['race'] = st.selectbox('Race:', [
                'white', 'black', 'asian', 'hispanic', 'american_indian/alaskan_native', 'other'
            ])
            categorical_features['diabetic'] = st.selectbox('Diabetic:', [
                'yes', 'no', 'no,_borderline_diabetes', 'yes_(during_pregnancy)'
            ])
            categorical_features['physicalactivity'] = st.selectbox('Physical Activity:', ['yes', 'no'])
            categorical_features['genhealth'] = st.selectbox('General Health:', [
                'excellent', 'very_good', 'good', 'fair', 'poor'
            ])
            categorical_features['asthma'] = st.selectbox('Asthma:', ['yes', 'no'])
            categorical_features['kidneydisease'] = st.selectbox('Kidney Disease:', ['yes', 'no'])
            categorical_features['skincancer'] = st.selectbox('Skin Cancer:', ['yes', 'no'])

        return {**numerical_features, **categorical_features}

    def predict(self, patient_data, dv, scaler, model):
        try:
            patient_df = pd.DataFrame([patient_data])

            numerical_features = ['bmi', 'physicalhealth', 'mentalhealth', 'sleeptime']
            X_num = scaler.transform(patient_df[numerical_features])

            categorical_dict = {k: v for k, v in patient_data.items()
                              if k not in numerical_features}
            X_cat = dv.transform([categorical_dict])

            X = np.hstack([X_num, X_cat])

            if hasattr(model, 'predict_proba'):
                return model.predict_proba(X)[0, 1]
            else:
                return model.predict(X)[0]
        except Exception as e:
            st.error(f"Prediction error: {str(e)}")
            return None

    def run(self):
        dv, scaler, model = self.load_model(self.selected_model_path)
        if dv is None:
            return

        if self.input_mode == "Sample from Test Data":
            patient_data = self.create_sample_input()
        else:
            patient_data = self.create_manual_input()

        st.markdown("---")

        if st.button('Predict Heart Disease Risk', type='primary'):
            with st.spinner('Analyzing patient data...'):
                progress_bar = st.progress(0)
                for i in range(100):
                    progress_bar.progress(i + 1)
                    import time
                    time.sleep(0.01)

            probability = self.predict(patient_data, dv, scaler, model)

            if probability is not None:
                st.header('Prediction Result')

                # Risk visualization
                col1, col2 = st.columns([2, 1])

                with col1:
                    st.subheader("Risk Probability")
                    st.progress(probability)

                    # Gauge-like visualization
                    if probability < 0.3:
                        st.success(f"🟢 Low Risk: {probability:.1%}")
                    elif probability < 0.7:
                        st.warning(f"🟡 Medium Risk: {probability:.1%}")
                    else:
                        st.error(f"🔴 High Risk: {probability:.1%}")

                with col2:
                    st.metric(
                        label="Heart Disease Probability",
                        value=f"{probability:.1%}"
                    )

                # Risk interpretation
                if probability >= 0.7:
                    risk_message = "🔴 **HIGH RISK** - Immediate medical consultation recommended!"
                    risk_color = "red"
                elif probability >= 0.3:
                    risk_message = "🟡 **MEDIUM RISK** - Regular health check-ups advised."
                    risk_color = "orange"
                else:
                    risk_message = "🟢 **LOW RISK** - Maintain healthy lifestyle."
                    risk_color = "green"

                st.markdown(f"<h3 style='color: {risk_color};'>{risk_message}</h3>", unsafe_allow_html=True)

                # Key metrics
                st.subheader('Key Health Metrics')
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("BMI", f"{patient_data['bmi']:.1f}")
                with col2:
                    st.metric("Physical Health", f"{patient_data['physicalhealth']:.0f} days")
                with col3:
                    st.metric("Mental Health", f"{patient_data['mentalhealth']:.0f} days")
                with col4:
                    st.metric("Sleep Time", f"{patient_data['sleeptime']:.0f} hours")

                # Additional insights
                with st.expander("Detailed Risk Factors"):
                    st.write("**High Risk Factors:**")
                    risk_factors = []
                    if patient_data['smoking'] == 'yes':
                        risk_factors.append("Smoking")
                    if patient_data['alcoholdrinking'] == 'yes':
                        risk_factors.append("Alcohol consumption")
                    if patient_data['stroke'] == 'yes':
                        risk_factors.append("Previous stroke")
                    if patient_data['diffwalking'] == 'yes':
                        risk_factors.append("Difficulty walking")
                    if patient_data['diabetic'] in ['yes', 'yes_(during_pregnancy)']:
                        risk_factors.append("Diabetes")
                    if patient_data['genhealth'] in ['fair', 'poor']:
                        risk_factors.append("Poor general health")

                    if risk_factors:
                        for factor in risk_factors:
                            st.write(f"• {factor}")
                    else:
                        st.write("No major risk factors identified from input data.")

if __name__ == "__main__":
    app = HeartDiseasePredictor()
    app.run()
