import streamlit as st
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

def main():
    st.title("SIMPLE LINEAR REGRESSION ANALYSIS")
    st.write("Enter the data and analyze the linear regression model easily.")

    # Input form
    st.subheader("Input Data")
    data_file = st.file_uploader("Upload a CSV file", type=["csv"])
    if data_file is not None:
        df = pd.read_csv(data_file)
        st.write(df)

        x_col = st.selectbox("Select the independent variable (X)", df.columns)
        y_col = st.selectbox("Select the dependent variable (Y)", df.columns)

        # Train the linear regression model
        model = LinearRegression()
        model.fit(df[[x_col]], df[y_col])

        # Prediction
        df['Predicted Y'] = model.predict(df[[x_col]])

        # Model evaluation
        mse = mean_squared_error(df[y_col], df['Predicted Y'])
        r2 = r2_score(df[y_col], df['Predicted Y'])

        # Display the results
        st.subheader("Regression Model Results")
        st.write("Mean Squared Error (MSE):", mse)
        st.write("R-squared (R2) Score:", r2)

        st.subheader("Data with Predictions")
        st.write(df)

if __name__ == '__main__':
    main()
