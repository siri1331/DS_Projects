import pickle
from sklearn.ensemble import RandomForestClassifier
import streamlit as st
import pandas as pd

def Logistic_model():
    train = pd.read_csv('loan_data.csv')
    #categorical to numerical
    train['Gender']= train['Gender'].map({'Male':0, 'Female':1})
    train['Married']= train['Married'].map({'No':0, 'Yes':1})
    train['Loan_Status']= train['Loan_Status'].map({'N':0, 'Y':1})

    # separating dependent and independent variables
    X = train[['Gender', 'Married', 'ApplicantIncome', 'LoanAmount']]
    y = train.Loan_Status

    # training the logistic regression model
    model = RandomForestClassifier() 
    model.fit(X, y)

    pickle_out = open("classifier_rf.pkl", mode = "wb") 
    pickle.dump(model, pickle_out) 
    pickle_out.close()



# this is the main function in which we define our app  
def main():       
    # header of the page 
    html_temp = """ 
    <div style ="background-color:yellow;padding:13px"> 
    <h1 style ="color:black;text-align:center;">Check your Loan Eligibility</h1> 
    </div> 
    """
    st.markdown(html_temp, unsafe_allow_html = True) 

    # following lines create boxes in which user can enter data required to make prediction 
    Gender = st.selectbox('Gender',("Male","Female","Other"))
    Married = st.selectbox('Marital Status',("Unmarried","Married","Other")) 
    ApplicantIncome = st.number_input("Monthly Income in Rupees") 
    LoanAmount = st.number_input("Loan Amount in Rupees")
    result =""
      
    # when 'Check' is clicked, make the prediction and store it 
    if st.button("Check"): 
        result = prediction(Gender, Married, ApplicantIncome, LoanAmount) 
        st.success('Your loan is {}'.format(result))
 
# prediction using the data which the user inputs 
def prediction(Gender, Married, ApplicantIncome, LoanAmount): 
    
    # loading the trained model
    pickle_in = open('classifier_rf.pkl', 'rb') 
    classifier = pickle.load(pickle_in)

    # Pre-processing the data 
    if Gender == "Male":
        Gender = 0
    else:
        Gender = 1

    if Married == "Married":
        Married = 1
    else:
        Married = 0

    prediction = classifier.predict( 
        [[Gender, Married, ApplicantIncome, LoanAmount]])
     
    if prediction == 1:
        pred = 'Approved'
    else:
        pred = 'Rejected'
    return pred
     
if __name__=='__main__': 
    Logistic_model()
    main()