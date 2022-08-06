import numpy as np
import pickle
import streamlit as st

loaded_model = pickle.load(open('C:/Users/Owner/Desktop/summer/ML_dataset/trained_model_diabetic', 'rb'))

def diabetic(input_data):
    input_tr = np.array(input_data)
    input_tr_reshape=input_tr.reshape(1, -1)
    predict = loaded_model.predict(input_tr_reshape)

    if predict[0]==0:
        return 'The person is Non-Diabetic'
    else:
        return 'The person is Diabetic'

def main():

    st.title('Diabetic Prediction Web-App')

    Pregnancies = st.text_input('Number of Pregnancies')
    Glucose = st.text_input('Body Glucose Level')
    BloodPressure = st.text_input('Blood Pressure Value')
    SkinThickness = st.text_input('Skin Thickness Value')
    Insulin =st.text_input('Body Insulin Level')
    BMI = st.text_input('BMI Value')
    DiabetesPedigreeFunction = st.text_input('Diabetes Pedigree Function Value')
    Age = st.text_input('Age of the Person')

    diagnosis=''
    if st.button('Diabetic Test Result'):
        diagnosis = diabetic([Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age])

    st.success(diagnosis)

if __name__ == '__main__':
    main()
