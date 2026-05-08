import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, roc_curve, auc

# --- Page Configuration ---
st.set_page_config(page_title="Financial Fraud Detection Dashboard", layout="wide")
st.title("🛡️ PaySim Financial Fraud Analysis")

# --- Data Loading (Cached) ---
@st.cache_data
def load_data():
    df = pd.read_csv('Pay_financial_fraud.csv')
    df.drop_duplicates(inplace=True)
    # Basic Feature Engineering for Viz
    df['hour'] = df['step'] % 24
    df['errorBalanceOrg'] = df['newbalanceOrig'] + df['amount'] - df['oldbalanceOrg']
    df['errorBalanceDest'] = df['oldbalanceDest'] + df['amount'] - df['newbalanceDest']
    df['dest_type'] = df['nameDest'].str[0]
    return df

@st.cache_resource
def train_model(df):
    # Simplified training for visualization purposes
    temp_df = df.sample(n=50000, random_state=42) # Sample for speed
    X = temp_df[['step', 'amount', 'oldbalanceOrg', 'newbalanceOrig', 'oldbalanceDest', 'newbalanceDest']]
    y = temp_df['isFraud']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestClassifier(n_estimators=10, random_state=42)
    model.fit(X_train, y_train)
    return model, X_test, y_test

df = load_data()

# --- Sidebar Navigation ---
st.sidebar.header("Navigation")
viz_options = [
    "1. Class Distribution",
    "2. Fraud by Transaction Type",
    "3. Hourly Patterns",
    "4. Amount Outliers",
    "5. Balance Error Signature",
    "6. Amount vs. Origin Balance",
    "7. Fraud by Destination Type",
    "8. Correlation Heatmap",
    "9. Model Performance (Confusion Matrix)",
    "10. ROC Curve"
]

# Session state to track current plot index
if 'viz_index' not in st.session_state:
    st.session_state.viz_index = 0

def next_plot():
    if st.session_state.viz_index < len(viz_options) - 1:
        st.session_state.viz_index += 1

def prev_plot():
    if st.session_state.viz_index > 0:
        st.session_state.viz_index -= 1

selected_viz = st.sidebar.selectbox("Select Graph", viz_options, index=st.session_state.viz_index)
st.session_state.viz_index = viz_options.index(selected_viz)

# --- Navigation Buttons ---
col1, col2, col3 = st.columns([1, 1, 4])
with col1:
    st.button("⬅️ Previous", on_click=prev_plot)
with col2:
    st.button("Next ➡️", on_click=next_plot)

# --- Main Dashboard Logic ---
st.subheader(f"Displaying: {selected_viz}")

if selected_viz == "1. Class Distribution":
    fig, ax = plt.subplots(1, 2, figsize=(14, 5))
    sns.countplot(x='isFraud', data=df, palette='magma', ax=ax[0])
    ax[0].set_title('Class Distribution')
    df['isFraud'].value_counts().plot.pie(autopct='%1.1f%%', colors=['skyblue', 'salmon'], ax=ax[1])
    ax[1].set_title('Fraud Percentage')
    st.pyplot(fig)

elif selected_viz == "2. Fraud by Transaction Type":
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.countplot(x='type', hue='isFraud', data=df, ax=ax)
    ax.set_yscale('log')
    ax.set_title('Fraud by Transaction Type (Log Scale)')
    st.pyplot(fig)

elif selected_viz == "3. Hourly Patterns":
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.lineplot(x='hour', y='isFraud', data=df, color='red', ax=ax)
    ax.set_title('Fraud Probability by Hour')
    st.pyplot(fig)

elif selected_viz == "4. Amount Outliers":
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.boxplot(x='isFraud', y='amount', data=df, ax=ax)
    ax.set_yscale('log')
    ax.set_title('Amount Outliers')
    st.pyplot(fig)

elif selected_viz == "5. Balance Error Signature":
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.violinplot(x='isFraud', y='errorBalanceOrg', data=df, ax=ax)
    ax.set_title('Origin Balance Error Signature')
    st.pyplot(fig)

elif selected_viz == "6. Amount vs. Origin Balance":
    fig, ax = plt.subplots(figsize=(10, 5))
    st.write("Showing a sample of 1000 points for performance:")
    legit = df[df.isFraud == 0].sample(1000)
    fraud = df[df.isFraud == 1].sample(min(1000, len(df[df.isFraud==1])))
    ax.scatter(legit['amount'], legit['oldbalanceOrg'], label='Legit', alpha=0.2)
    ax.scatter(fraud['amount'], fraud['oldbalanceOrg'], label='Fraud', color='red')
    ax.set_title('Amount vs. Origin Balance')
    ax.legend()
    st.pyplot(fig)

elif selected_viz == "7. Fraud by Destination Type":
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.countplot(x='dest_type', hue='isFraud', data=df, ax=ax)
    ax.set_yscale('log')
    ax.set_title('Fraud by Destination Type')
    st.pyplot(fig)

elif selected_viz == "8. Correlation Heatmap":
    fig, ax = plt.subplots(figsize=(12, 8))
    sns.heatmap(df.select_dtypes(include=[np.number]).corr(), annot=True, fmt='.2f', cmap='coolwarm', ax=ax)
    ax.set_title('Feature Correlation Heatmap')
    st.pyplot(fig)

elif selected_viz in ["9. Model Performance (Confusion Matrix)", "10. ROC Curve"]:
    with st.spinner("Training model for evaluation..."):
        model, X_test, y_test = train_model(df)
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]

    if selected_viz == "9. Model Performance (Confusion Matrix)":
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Blues', ax=ax)
        ax.set_title('Confusion Matrix')
        ax.set_xlabel('Predicted')
        ax.set_ylabel('Actual')
        st.pyplot(fig)
    else:
        fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
        roc_auc = auc(fpr, tpr)
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
        ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title('ROC Curve')
        ax.legend(loc='lower right')
        st.pyplot(fig)

# --- Summary Stats ---
st.sidebar.markdown("---")
st.sidebar.write(f"**Total Transactions:** {len(df)}")
st.sidebar.write(f"**Fraudulent Cases:** {df['isFraud'].sum()}")