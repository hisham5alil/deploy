import streamlit as st
import pickle

file_name = 'model.pkl'
with open(file_name, 'rb') as file:
    model = pickle.load(file)


st.title("Classifier Model")
st.header('Hello')

if st.button('button'):
    st.write('This is a button')
    # age = st.sidebar()

# st.sidebar()