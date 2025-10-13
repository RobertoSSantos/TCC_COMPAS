
import pandas as pd
import pickle
import os

def load_compas_data():
    """
    Carrega dados processados do COMPAS do Google Drive
    
    Returns:
        dict: Dicionário com 'df_principal', 'bias_df', 'metadata'
    """
    base_path = '/content/drive/MyDrive/TCC_COMPAS'
    file_path = f'{base_path}/data/compas_processed.pkl'
    
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Arquivo não encontrado: {file_path}")
    
    with open(file_path, 'rb') as f:
        data = pickle.load(f)
    
    print("✅ DADOS CARREGADOS COM SUCESSO!")
    print(f"📊 Dataset: {data['df_principal'].shape}")
    print(f"📈 Métricas de viés: {data['bias_df'].shape}")
    print(f"🕐 Processado em: {data['metadata']['processing_date']}")
    print(f"👥 Casos: {data['metadata']['total_casos']}")
    
    return data

def get_dataframes():
    """
    Retorna dataframes principais de forma conveniente
    
    Returns:
        tuple: (df_principal, bias_df, metadata)
    """
    data = load_compas_data()
    return data['df_principal'], data['bias_df'], data['metadata']

def setup_environment():
    """
    Configura ambiente completo para análise
    """
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns
    import warnings
    warnings.filterwarnings('ignore')
    
    # Configurações de visualização
    plt.style.use('seaborn-v0_8')
    sns.set_palette("husl")
    
    print("🎉 AMBIENTE CONFIGURADO!")
    print("   • Pandas, NumPy, Matplotlib, Seaborn")
    print("   • Visualizações configuradas")
    print("   • Warnings desativados")
    
    return load_compas_data()
