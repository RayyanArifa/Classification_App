import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import warnings
warnings.filterwarnings('ignore')

# Konfigurasi halaman
st.set_page_config(
    page_title="Klasifikasi Data AI",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS untuk styling
st.markdown("""
<style>
.main {
    padding-top: 1rem;
}
.stTitle {
    color: #2E86AB;
    font-size: 3rem !important;
    text-align: center;
    margin-bottom: 2rem;
}
.stSubheader {
    color: #A23B72;
    font-size: 1.5rem !important;
}
.metric-card {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    padding: 1rem;
    border-radius: 10px;
    color: white;
    margin: 0.5rem 0;
}
.upload-section {
    background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
    padding: 2rem;
    border-radius: 15px;
    margin: 1rem 0;
    color: white;
}
.sidebar .sidebar-content {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
}
</style>
""", unsafe_allow_html=True)

def main():
    # Header
    st.markdown('<h1 class="stTitle">🤖 Klasifikasi Data dengan AI</h1>', unsafe_allow_html=True)
    st.markdown('<p style="text-align: center; color: #666; font-size: 1.2rem;">Upload dataset Anda dan lakukan klasifikasi dengan berbagai algoritma machine learning</p>', unsafe_allow_html=True)
    
    # Sidebar
    st.sidebar.markdown("## ⚙️ Pengaturan Model")
    
    # Upload file
    st.markdown('<div class="upload-section">', unsafe_allow_html=True)
    st.markdown("### 📁 Upload Dataset")
    uploaded_file = st.file_uploader(
        "Pilih file CSV atau Excel",
        type=['csv', 'xlsx', 'xls'],
        help="Upload file dataset dalam format CSV atau Excel"
    )
    st.markdown('</div>', unsafe_allow_html=True)
    
    if uploaded_file is not None:
        # Load data
        try:
            if uploaded_file.name.endswith('.csv'):
                df = pd.read_csv(uploaded_file)
            else:
                df = pd.read_excel(uploaded_file)
            
            st.success(f"✅ Dataset berhasil dimuat! Shape: {df.shape}")
            
            # Tampilkan informasi dataset
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.markdown(f'<div class="metric-card"><h3>{df.shape[0]}</h3><p>Jumlah Baris</p></div>', unsafe_allow_html=True)
            with col2:
                st.markdown(f'<div class="metric-card"><h3>{df.shape[1]}</h3><p>Jumlah Kolom</p></div>', unsafe_allow_html=True)
            with col3:
                st.markdown(f'<div class="metric-card"><h3>{df.isnull().sum().sum()}</h3><p>Missing Values</p></div>', unsafe_allow_html=True)
            with col4:
                st.markdown(f'<div class="metric-card"><h3>{df.select_dtypes(include=[np.number]).shape[1]}</h3><p>Kolom Numerik</p></div>', unsafe_allow_html=True)
            
            # Tabs untuk navigasi
            tab1, tab2, tab3, tab4 = st.tabs(["📊 Eksplorasi Data", "📈 Visualisasi", "🤖 Klasifikasi", "📋 Hasil"])
            
            with tab1:
                st.markdown("### 📊 Eksplorasi Data")
                
                # Tampilkan sample data
                st.subheader("Sample Data")
                st.dataframe(df.head(10), use_container_width=True)
                
                # Informasi statistik
                col1, col2 = st.columns(2)
                
                with col1:
                    st.subheader("Informasi Dataset")
                    info_df = pd.DataFrame({
                        'Tipe Data': df.dtypes,
                        'Non-Null Count': df.count(),
                        'Null Count': df.isnull().sum()
                    })
                    st.dataframe(info_df, use_container_width=True)
                
                with col2:
                    st.subheader("Statistik Deskriptif")
                    st.dataframe(df.describe(), use_container_width=True)
            
            with tab2:
                st.markdown("### 📈 Visualisasi Data")
                
                # Pilih kolom untuk visualisasi
                numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
                categorical_columns = df.select_dtypes(include=['object']).columns.tolist()
                
                if numeric_columns:
                    # Correlation Matrix
                    st.subheader("Correlation Matrix")
                    corr_matrix = df[numeric_columns].corr()
                    fig = px.imshow(corr_matrix, 
                                   text_auto=True, 
                                   aspect="auto",
                                   color_continuous_scale='RdBu_r')
                    fig.update_layout(title="Correlation Matrix", height=600)
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Distribution plots
                    st.subheader("Distribusi Data")
                    selected_columns = st.multiselect(
                        "Pilih kolom untuk visualisasi distribusi:",
                        numeric_columns,
                        default=numeric_columns[:4] if len(numeric_columns) >= 4 else numeric_columns
                    )
                    
                    if selected_columns:
                        cols = st.columns(2)
                        for i, col in enumerate(selected_columns):
                            with cols[i % 2]:
                                fig = px.histogram(df, x=col, nbins=30, 
                                                 title=f"Distribusi {col}",
                                                 color_discrete_sequence=['#667eea'])
                                fig.update_layout(height=400)
                                st.plotly_chart(fig, use_container_width=True)
                
                # Scatter plot
                if len(numeric_columns) >= 2:
                    st.subheader("Scatter Plot")
                    col1, col2 = st.columns(2)
                    with col1:
                        x_axis = st.selectbox("Pilih X-axis:", numeric_columns, index=0)
                    with col2:
                        y_axis = st.selectbox("Pilih Y-axis:", numeric_columns, index=1)
                    
                    color_by = None
                    if categorical_columns:
                        color_by = st.selectbox("Color by (opsional):", ['None'] + categorical_columns)
                        color_by = None if color_by == 'None' else color_by
                    
                    fig = px.scatter(df, x=x_axis, y=y_axis, color=color_by,
                                   title=f"Scatter Plot: {x_axis} vs {y_axis}")
                    fig.update_layout(height=500)
                    st.plotly_chart(fig, use_container_width=True)
            
            with tab3:
                st.markdown("### 🤖 Klasifikasi Data")
                
                # Pilih target column
                target_column = st.selectbox(
                    "Pilih kolom target (yang akan diprediksi):",
                    df.columns.tolist(),
                    help="Pilih kolom yang berisi label/kelas yang ingin diprediksi"
                )
                
                # Pilih feature columns
                feature_columns = st.multiselect(
                    "Pilih kolom fitur:",
                    [col for col in df.columns if col != target_column],
                    default=[col for col in df.columns if col != target_column][:5]
                )
                
                if target_column and feature_columns:
                    # Preprocessing
                    X = df[feature_columns].copy()
                    y = df[target_column].copy()
                    
                    # Handle missing values
                    for col in X.columns:
                        if X[col].isnull().sum() > 0:
                            if X[col].dtype == 'object':
                                X[col] = X[col].fillna(X[col].mode()[0])
                            else:
                                X[col] = X[col].fillna(X[col].mean())
                                    
                    # Encode categorical variables
                    label_encoders = {}
                    for col in X.select_dtypes(include=['object']).columns:
                        le = LabelEncoder()
                        X[col] = le.fit_transform(X[col].astype(str))
                        label_encoders[col] = le
                    
                    # Encode target if categorical
                    if y.dtype == 'object':
                        le_target = LabelEncoder()
                        y = le_target.fit_transform(y)
                    
                    # Split data
                    test_size = st.sidebar.slider("Test Size", 0.1, 0.5, 0.2, 0.05)
                    X_train, X_test, y_train, y_test = train_test_split(
                        X, y, test_size=test_size, random_state=42
                    )
                    
                    # Scale features
                    scaler = StandardScaler()
                    X_train_scaled = scaler.fit_transform(X_train)
                    X_test_scaled = scaler.transform(X_test)
                    
                    # Model selection
                    models = {
                        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
                        'Logistic Regression': LogisticRegression(random_state=42),
                        'SVM': SVC(random_state=42),
                        'K-Nearest Neighbors': KNeighborsClassifier(),
                        'Decision Tree': DecisionTreeClassifier(random_state=42),
                        'Naive Bayes': GaussianNB()
                    }
                    
                    selected_model = st.sidebar.selectbox(
                        "Pilih Model:",
                        list(models.keys())
                    )
                    
                    if st.button("🚀 Mulai Training", type="primary"):
                        with st.spinner("Training model..."):
                            model = models[selected_model]
                            
                            # Training
                            if selected_model in ['Logistic Regression', 'SVM']:
                                model.fit(X_train_scaled, y_train)
                                y_pred = model.predict(X_test_scaled)
                            else:
                                model.fit(X_train, y_train)
                                y_pred = model.predict(X_test)
                            
                            # Store results in session state
                            st.session_state['model'] = model
                            st.session_state['scaler'] = scaler
                            st.session_state['y_test'] = y_test
                            st.session_state['y_pred'] = y_pred
                            st.session_state['accuracy'] = accuracy_score(y_test, y_pred)
                            st.session_state['selected_model'] = selected_model
                            st.session_state['feature_columns'] = feature_columns
                            st.session_state['target_column'] = target_column
                            
                        st.success(f"✅ Model {selected_model} berhasil di-training!")
                        st.rerun()
            
            with tab4:
                st.markdown("### 📋 Hasil Klasifikasi")
                
                if 'model' in st.session_state:
                    model = st.session_state['model']
                    y_test = st.session_state['y_test']
                    y_pred = st.session_state['y_pred']
                    accuracy = st.session_state['accuracy']
                    selected_model = st.session_state['selected_model']
                    
                    # Tampilkan akurasi
                    st.success(f"🎯 Akurasi Model {selected_model}: {accuracy:.4f}")
                    
                    # Confusion Matrix
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.subheader("Confusion Matrix")
                        cm = confusion_matrix(y_test, y_pred)
                        fig = px.imshow(cm, 
                                       text_auto=True, 
                                       aspect="auto",
                                       color_continuous_scale='Blues')
                        fig.update_layout(title="Confusion Matrix", height=400)
                        st.plotly_chart(fig, use_container_width=True)
                    
                    with col2:
                        st.subheader("Classification Report")
                        report = classification_report(y_test, y_pred, output_dict=True)
                        report_df = pd.DataFrame(report).transpose()
                        st.dataframe(report_df, use_container_width=True)
                    
                    # Feature importance (untuk model yang support)
                    if hasattr(model, 'feature_importances_'):
                        st.subheader("Feature Importance")
                        feature_importance = pd.DataFrame({
                            'feature': st.session_state['feature_columns'],
                            'importance': model.feature_importances_
                        }).sort_values('importance', ascending=False)
                        
                        fig = px.bar(feature_importance, 
                                   x='importance', 
                                   y='feature',
                                   orientation='h',
                                   title="Feature Importance",
                                   color='importance',
                                   color_continuous_scale='viridis')
                        fig.update_layout(height=400)
                        st.plotly_chart(fig, use_container_width=True)
                    
                    # Prediksi baru
                    st.subheader("🔮 Prediksi Data Baru")
                    st.write("Masukkan nilai fitur untuk prediksi:")
                    
                    input_data = {}
                    cols = st.columns(3)
                    for i, feature in enumerate(st.session_state['feature_columns']):
                        with cols[i % 3]:
                            if feature in df.select_dtypes(include=[np.number]).columns:
                                input_data[feature] = st.number_input(f"{feature}:", value=float(df[feature].mean()))
                            else:
                                unique_values = df[feature].unique()
                                input_data[feature] = st.selectbox(f"{feature}:", unique_values)
                    
                    if st.button("🎯 Prediksi", type="primary"):
                        # Prepare input
                        input_df = pd.DataFrame([input_data])
                        
                        # Encode categorical variables
                        for col in input_df.select_dtypes(include=['object']).columns:
                            if col in label_encoders:
                                input_df[col] = label_encoders[col].transform(input_df[col])
                        
                        # Make prediction
                        if selected_model in ['Logistic Regression', 'SVM']:
                            input_scaled = st.session_state['scaler'].transform(input_df)
                            prediction = model.predict(input_scaled)[0]
                        else:
                            prediction = model.predict(input_df)[0]
                        
                        # Show result
                        if 'le_target' in locals():
                            prediction_label = le_target.inverse_transform([prediction])[0]
                        else:
                            prediction_label = prediction
                        
                        st.success(f"🎯 Prediksi: **{prediction_label}**")
                
                else:
                    st.info("👆 Silakan lakukan training model terlebih dahulu di tab Klasifikasi")
        
        except Exception as e:
            st.error(f"❌ Error dalam memuat file: {str(e)}")
    
    else:
        # Landing page
        st.markdown("""
        <div style="text-align: center; padding: 3rem; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); border-radius: 15px; margin: 2rem 0; color: white;">
            <h2>🚀 Mulai Klasifikasi Data Anda</h2>
            <p style="font-size: 1.2rem; margin: 1rem 0;">Upload dataset CSV atau Excel untuk memulai analisis dan klasifikasi</p>
            <div style="display: flex; justify-content: center; gap: 2rem; margin-top: 2rem;">
                <div style="background: rgba(255,255,255,0.2); padding: 1rem; border-radius: 10px;">
                    <h3>📊 Eksplorasi</h3>
                    <p>Analisis data mendalam</p>
                </div>
                <div style="background: rgba(255,255,255,0.2); padding: 1rem; border-radius: 10px;">
                    <h3>📈 Visualisasi</h3>
                    <p>Grafik interaktif</p>
                </div>
                <div style="background: rgba(255,255,255,0.2); padding: 1rem; border-radius: 10px;">
                    <h3>🤖 AI Model</h3>
                    <p>6 algoritma tersedia</p>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()