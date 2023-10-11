import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import os as os
from sklearn.impute import SimpleImputer
import imblearn
from PIL import Image
from imblearn.under_sampling import RandomUnderSampler
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import xgboost as xgb
from catboost import CatBoostClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, roc_auc_score, classification_report, roc_curve



banner = Image.open('banner.jpg')
logo = Image.open('logo.png')


st.set_page_config(
    layout = 'wide',
    page_title = 'Streamlit Web Application'
)

st.sidebar.image(logo, use_column_width='always')

menu = st.sidebar.selectbox('', ['Homepage', 'EDA', 'Modeling'])

if menu == 'Homepage':
    st.header('Homepage')
    st.image(banner, use_column_width='always')

    data = st.selectbox('Select Dataset', ['Water Portability', 'Loan Prediction'])

    st.markdown(f'Selected: {data} dataset')
    if data == 'Loan Prediction':
        st.header('Our problem')

        st.warning('''Dream Housing Finance company deals in 
                   all home loans. They have presence 
                   across all urban, semi urban and rural 
                   areas. Customer first apply for home 
                   loan after that company validates the 
                   customer eligibility for loan.
                   Company wants to automate the loan 
                   eligibility process (real time) 
                   based on customer detail provided 
                   while filling online application 
                   form. These details are Gender, 
                   Marital Status, Education, Number 
                   of Dependents, Income, Loan Amount, 
                   Credit History and others. 
                   To automate this process, they have
                   given a problem to identify the 
                   customers segments, those are 
                   eligible for loan amount so that 
                   they can specifically target these 
                   customers. Here they have provided 
                   a partial data set.''')
        st.header('Dataset')
        df = pd.read_csv('loan_pred (1).csv')
        st.dataframe(df)
    else:
        st.header('Our problem')

        st.warning('''Access to safe drinking-water is 
                   essential to health, a basic human 
                   right and a component of effective 
                   policy for health protection. This 
                   is important as a health and 
                   development issue at a national, 
                   regional and local level. In some 
                   regions, it has been shown that 
                   investments in water supply and 
                   sanitation can yield a net economic 
                   benefit, since the reductions in 
                   adverse health effects and health 
                   care costs outweigh the costs of 
                   undertaking the interventions.''')
        st.header('Dataset')
        df = pd.read_csv('water_potability (1).csv')
        st.dataframe(df)

elif menu == 'EDA':
    def outlier(datacol):
        sorted(datacol)
        q1, q3 = np.percentile(datacol, [25,75])
        iqr = q3-q1
        lower_range = q1 - (1.5*iqr)
        upper_range = q3 + (1.5*iqr)
        return lower_range, upper_range
    
    def describe(df):
        st.dataframe(df)
        st.subheader('Describtion of dataset (statistics)')
        df.describe().T

        st.subheader('Data balance')
        st.bar_chart(df.iloc[:,-1].value_counts())

        null_df = df.isnull().sum().to_frame().reset_index()
        null_df.columns = ['Columns', 'Counts']

        c1, c2, c3 = st.columns([3,2,3])

        c1.subheader('Null values')
        c1.dataframe(null_df)

        c2.subheader('Imputation')
        cat_met = c2.radio('Categorical', ['Mode', 'Backfill', 'Ffill'])
        num_met = c2.radio('Numerical', ['Median', 'Mean'])

        c2.subheader('Feature engineering')
        bal_prob = c2.checkbox('Under Sampling')
        outl_prob = c2.checkbox('Clean Outlier')

        if c2.button('Data Preprocessing'):

            cat_array = df.iloc[:,:-1].select_dtypes(include='object').columns
            num_array = df.iloc[:, :-1].select_dtypes(exclude='object').columns

            if cat_array.size > 0:
                if cat_met == 'Mode':
                    imp_cat = SimpleImputer(
                        missing_values=np.nan, strategy='most_frequent'
                    )
                    df[cat_array] = imp_cat.fit_transform(df[cat_array])

                elif cat_met == 'Backfill':
                    df[cat_array].fillna(method='backfill', inplace=True)

                else:
                    df[cat_array].fillna(method='ffill', inplace=True)
            
            if num_array.size > 0 :
                if num_met == 'Median':
                    imp_num = SimpleImputer(
                        missing_values=np.nan, strategy='median'
                    )
                    df[num_array] = imp_num.fit_transform(df[num_array])

                else:
                    imp_num = SimpleImputer(
                        missing_values=np.nan, strategy='mean'
                    )
                    df[num_array] = imp_num.fit_transform(df[num_array])

            if bal_prob:
                rs = RandomUnderSampler()
                x = df.iloc[:, :-1]
                y = df.iloc[:, -1]

                x, y = rs.fit_resample(x, y)
                df = pd.concat([x,y], axis=1)

            if outl_prob:
                for col in num_array:
                    lowb, uppb = outlier(df[col])
                    df[col] = np.clip(df[col], a_min=lowb, a_max=uppb)
            
        null_df = df.isnull().sum().to_frame().reset_index()
        null_df.columns = ['Columns', 'Counts']

        c3.subheader('Null values')
        c3.dataframe(null_df)

        st.subheader('Balance of data')
        st.bar_chart(df.iloc[:, -1].value_counts())

        heatmap = px.imshow(df.corr())
        st.plotly_chart(heatmap)
        st.dataframe(df)

        if os.path.exists('model.csv'):
            os.remove('model.csv')
        df.to_csv('model.csv', index=False)

    st.header('EDA')
    dataset = st.selectbox('Select Dataset', ['Water Portability', 'Loan Prediction'])
    st.markdown(f'Selected: {dataset} dataset')
    
    if dataset == 'Loan Prediction':
        df = pd.read_csv('loan_pred (1).csv')
        describe(df)
    else:
        df = pd.read_csv('water_potability (1).csv')
        describe(df)

else:
    st.header('Modeling')
    if not os.path.exists('model.csv'):
        st.header('Run Preprocessing Please')
    else:
        df = pd.read_csv('model.csv')
        st.dataframe(df)

        c1, c2 = st.columns([3,3])

        c1.subheader('Scaling')
        sc_mod = c1.radio('', ['Standard', 'Robust', 'MinMax'])
        c2.subheader('Encoder')
        en_mod = c2.radio('', ['Label', 'One-Hot'])

        y = df.iloc[:, -1]
        cat_array = df.iloc[:, :-1].select_dtypes(include='object').columns
        num_array = df.iloc[:, :-1].select_dtypes(exclude='object').columns

        if num_array.size > 0 :
            if sc_mod == 'Standard':
                sc = StandardScaler()
            elif sc_mod == 'Robust':
                sc = RobustScaler()
            else: 
                sc = MinMaxScaler()

        if cat_array.size > 0 :
            if en_mod == 'Label':
                lb = LabelEncoder()
                for col in cat_array:
                    df[col] = lb.fit_transform(df[col])

            else:
                df.drop(df.iloc[:, -1], axis=1, inplace=True)
                d_df = df[cat_array]
                d_df = pd.get_dummies(d_df, drop_first=True)
                df_ = df.drop(cat_array, axis=1)
                df = pd.concat([df_, d_df, y], axis=1)
        
        st.header('Train/Test Split')
        st.dataframe(df)

        test_size, random_state = st.columns([2,2])
        test_size = test_size.number_input('Type a test size: ', value=0.3)
        random_state = random_state.number_input('Type a random state: ')
        random_state = round(int(random_state), 0)
        x = df.iloc[:, :-1]
        y = df.iloc[:, -1]
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=float(test_size), random_state=random_state)

        st.markdown(f'X_train size = {x_train.shape}')
        st.markdown(f'X_test size = {x_test.shape}')
        st.markdown(f'y_train size = {y_train.shape}')
        st.markdown(f'y_test size = {y_test.shape}')

        model = st.selectbox('Select a model ', ['XGBoost', 'CatBoost'])

        if model == 'XGBoost':
            model = xgb.XGBClassifier().fit(x_train, y_train)

        else:
            model = CatBoostClassifier().fit(x_train, y_train)

        
        if st.button('Press to see the results'):
            preds = model.predict(x_test)

            c1, c2 = st.columns(2)
            accuracy = accuracy_score(y_test, preds)
            accuracy = round(accuracy, 2)
            st.header(f'Accuracy is: {accuracy}')

            auc = roc_auc_score(y_test, preds)
            auc = round(auc, 2)
            st.header(f'Auc Score is: {auc}')

            confusion = confusion_matrix(y_test, preds)
            ss = pd.DataFrame(confusion)
            st.header('Confusion Matrix')
            st.dataframe(ss)

            cr = classification_report(y_test, preds, output_dict=True)
            sr = pd.DataFrame(cr).transpose()
            st.header('Classification Report')
            st.dataframe(sr)

            logit_roc_auc = roc_auc_score(y_test, model.predict(x_test))
            fpr, tpr, thresholds = roc_curve(y_test, model.predict_proba(x_test)[:, 1])
            fig = plt.figure(figsize=(10,5))
            plt.plot(fpr, tpr, label='Classification Model (area = %0.2f)')
            plt.plot([0,1], [0,1], 'r--')
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.legend(loc='lower right')
            st.header('ROC Curve')
            st.pyplot(fig)











