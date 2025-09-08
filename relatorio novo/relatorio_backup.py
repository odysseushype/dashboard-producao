import dash
from dash import dcc, html, Input, Output, State, ctx, callback_context, ALL
import dash_bootstrap_components as dbc
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from datetime import datetime, timedelta
import base64
import io
import os

# Variáveis globais para armazenar os DataFrames para depuração e análise
df_original_global = None
df_processado_global = None

# Função para calcular o dia produtivo
def calcular_dia_produtivo(data_hora):
    """
    Calcula o dia produtivo com base na data e hora.
    O dia produtivo vai das 06:00 de um dia até as 06:00 do dia seguinte.
    
    Args:
        def update_output(contents, filename):
    if contents is None:
        # Sem upload - mostrar mensagem clara para carregar um arquivo
        # Usar o DataFrame vazio
        empty_df = criar_df_vazio()
        
        setores = ['Montagem', 'Pintura', 'Embalagem']
        opcoes_setores = [{'label': setor, 'value': setor} for setor in setores]
        
        # Opções de tipos de máquina padrão
        opcoes_tipos = [
            {'label': 'Planas', 'value': 'tipo_Planas'},
            {'label': 'Gralex', 'value': 'tipo_Gralex'},
            {'label': 'Clamshell', 'value': 'tipo_Clamshell'},
            {'label': 'Sprinter', 'value': 'tipo_Sprinter'},
            {'label': 'China in Box', 'value': 'tipo_China_in_Box'},
            {'label': 'Janela', 'value': 'tipo_Janela'},
            {'label': 'Outros', 'value': 'tipo_Outros'}
        ]
        
        return (
            empty_df.to_dict('records'), 
            True,  # Mostrar alerta de "sem dados"
            False, 
            "",
            empty_df['data'].iloc[0],
            empty_df['data'].min(),
            empty_df['data'].max(),
            opcoes_setores,
            setores,
            opcoes_tipos
        )- A data e hora a ser convertida
    
    Returns:
        string - Data do dia produtivo no formato YYYY-MM-DD
    """
    if isinstance(data_hora, datetime):
        # Se a hora for menor que 06:00, o dia produtivo é o dia anterior
        if data_hora.hour < 6:
            return (data_hora - timedelta(days=1)).strftime('%Y-%m-%d')
        else:
            return data_hora.strftime('%Y-%m-%d')
    return None

# Função para classificar tipo de máquina
def classificar_tipo_maquina(maquina):
    """
    Classifica o tipo de máquina com base no código.
    
    Args:
        maquina: str - Código da máquina
    
    Returns:
        string - Tipo da máquina (Planas, Gralex, Clamshell, Sprinter, China in Box, Janela)
    """
    if not isinstance(maquina, str):
        return "Outros"
    
    # Normaliza o código da máquina (remove espaços, converte para maiúsculo)
    maquina = maquina.strip().upper()
    
    # Classificar por tipo
    if maquina.startswith("CA"):
        try:
            # Extrair o número da máquina
            num = int(maquina[2:])
            if 1 <= num <= 16:
                return "Planas"
            elif 17 <= num <= 24:
                return "Desativada"  # Ignorar
        except:
            pass
    elif maquina.startswith("GR"):
        # Gralex inclui GR01 a GR06B
        base = maquina[:4]  # Pega os primeiros 4 caracteres (ex: GR01)
        if base in ["GR01", "GR02", "GR03", "GR04", "GR05", "GR06"]:
            return "Gralex"
    elif maquina.startswith("MC"):
        try:
            # Extrair o número da máquina
            num = int(maquina[2:])
            if 1 <= num <= 15:
                return "Clamshell"
        except:
            pass
    elif maquina.startswith("SP"):
        # Sprinter inclui SP01 a SP02B
        base = maquina[:4]  # Pega os primeiros 4 caracteres (ex: SP01)
        if base in ["SP01", "SP02"]:
            return "Sprinter"
    elif maquina.startswith("CH"):
        try:
            # Extrair o número da máquina
            num = int(maquina[2:])
            if 1 <= num <= 2:
                return "China in Box"
        except:
            pass
    elif maquina.startswith("JA"):
        try:
            # Extrair o número da máquina
            num = int(maquina[2:])
            if num == 1:
                return "Janela"
        except:
            pass
    
    # Casos especiais a ignorar - normalizar substituindo espaços por underscores para comparação
    if maquina.replace(" ", "_") in ["DESTAQUE_COLAGEM", "COL_MANUAL"] or maquina in ["DESTAQUE COLAGEM", "COL MANUAL"]:
        return "Ignorar"
        
    return "Outros"

# Criar DataFrame vazio para inicialização
def criar_df_vazio():
    # Criar DataFrame vazio para quando não houver dados
    hoje = datetime.now()
    df = pd.DataFrame({
        'data': [hoje.strftime("%Y-%m-%d")],
        'producao': [0],
        'eficiencia': [0],
        'maquina1': [0],
        'maquina2': [0],
        'maquina3': [0],
        'parada_Manutenção': [0],
        'parada_Setup': [0],
        'velocidade_id': [0]
    })
    return df

# Criar pasta para armazenar uploads se não existir
if not os.path.exists('uploads'):
    os.makedirs('uploads')

# Inicializar o app Dash
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
app.title = "Relatório de Produção"

# Define a função para processar arquivos Excel
def parse_contents(contents, filename):
    content_type, content_string = contents.split(',')
    decoded = base64.b64decode(content_string)
    
    try:
        # Assume que o arquivo é um Excel
        df = pd.read_excel(io.BytesIO(decoded))
        
        # Salvar o DataFrame original
        df_original = df.copy()
        
        # Converter colunas numéricas armazenadas como texto para números
        df = converter_texto_para_numero(df)
        
        # Salva o arquivo na pasta uploads
        with open(os.path.join('uploads', filename), 'wb') as f:
            f.write(decoded)
        
        # Preencher dados faltantes no DataFrame original
        print("Preenchendo dados faltantes no DataFrame original...")
        df_original_preenchido = preencher_dados_faltantes(df_original)
        
        # Guardar o DataFrame original preenchido em uma variável global para facilitar o acesso
        global df_original_global
        df_original_global = df_original_preenchido
        
        # Processar os dados para formato necessário para o dashboard
        df_processado = processar_dados_producao(df)
        
        # Guardar também o DataFrame processado
        global df_processado_global
        df_processado_global = df_processado
        
        # Mostrar que o container de download está disponível
        app.clientside_callback(
            """
            function(n) {
                document.getElementById("download-container").style.display = "block";
                return window.dash_clientside.no_update;
            }
            """,
            Output("download-container", "style"),
            Input("store-data", "data")
        )
        
        return df_processado, f"Arquivo '{filename}' carregado com sucesso! Dados faltantes foram preenchidos automaticamente.", True
    except Exception as e:
        print(e)
        return None, f"Erro ao processar o arquivo: {e}", False

# Função para preencher dados faltantes nas linhas de parada
def preencher_dados_faltantes(df):
    """
    Preenche dados faltantes de roteiro, item e ordem de produção nas linhas onde só existe registro de parada.
    Segue as regras:
    1. Para cada dia produtivo, verificar a primeira entrada de item e preencher até o começo do dia
    2. Preencher para baixo até encontrar uma troca de máquina
    3. Quando encontra paradas de "Acerto" ou "TROCA DE ÍTEM/OS", imediatamente começa a usar o próximo item disponível
    
    Args:
        df: DataFrame com os dados carregados do Excel
        
    Returns:
        DataFrame com os dados faltantes preenchidos
    """
    from datetime import datetime, timedelta
    
    # Criar uma cópia para não modificar o original
    df_preenchido = df.copy()
    
    # Verificar se as colunas necessárias estão presentes (colunas antigas)
    colunas_antigas = ['Centro Trabalho', 'Descrição Parada', 'Ordem Prod', 'Descrição Item', 'Roteiro']
    
    # Verificar se as colunas necessárias estão presentes (colunas novas)
    colunas_novas = ['maquina', 'parada', 'ordem_servico', 'item', 'faca']
    
    # Determinar qual formato de colunas usar
    if all(col in df_preenchido.columns for col in colunas_antigas):
        # Usar o formato antigo (dados brutos do Excel)
        print("Detectado formato de dados original (colunas originais do Excel).")
        
        # Adicionar coluna de dia produtivo se não existir
        if 'dia_produtivo' not in df_preenchido.columns:
            # Criar datetime_inicio para calcular o dia produtivo
            try:
                # Tentar criar datetime combinando data e hora
                if 'Data Início' in df_preenchido.columns and 'Hora Início' in df_preenchido.columns:
                    df_preenchido['data_inicio'] = pd.to_datetime(df_preenchido['Data Início'], errors='coerce')
                    
                    # Processar a hora
                    if isinstance(df_preenchido['Hora Início'].iloc[0], float):
                        # Hora no formato Excel (fração do dia)
                        df_preenchido['hora_inicio'] = pd.to_datetime(
                            (df_preenchido['Hora Início'] * 24 * 3600).apply(
                                lambda x: timedelta(seconds=x) if pd.notna(x) else None
                            )
                        ).dt.time
                    else:
                        # Tentar converter hora para time
                        df_preenchido['hora_inicio'] = pd.to_datetime(df_preenchido['Hora Início'], format='%H:%M:%S', errors='coerce').dt.time
                    
                    # Criar datetime combinando data e hora
                    df_preenchido['datetime_inicio'] = pd.to_datetime(
                        df_preenchido['data_inicio'].dt.strftime('%Y-%m-%d') + ' ' + 
                        df_preenchido['hora_inicio'].astype(str),
                        errors='coerce'
                    )
                    
                    # Calcular o dia produtivo (06:00 de um dia até 06:00 do próximo)
                    df_preenchido['dia_produtivo'] = df_preenchido['datetime_inicio'].apply(
                        lambda x: (x - timedelta(days=1)).strftime('%Y-%m-%d') if pd.notna(x) and x.hour < 6 else 
                                 x.strftime('%Y-%m-%d') if pd.notna(x) else None
                    )
                else:
                    # Se não tiver data/hora, usar a data/hora atual para o dia produtivo
                    df_preenchido['dia_produtivo'] = datetime.now().strftime('%Y-%m-%d')
            except Exception as e:
                print(f"Erro ao calcular dia produtivo: {e}")
                # Usar a data/hora atual para o dia produtivo como fallback
                df_preenchido['dia_produtivo'] = datetime.now().strftime('%Y-%m-%d')
        
        # Ordenar o DataFrame por Centro Trabalho, dia produtivo e datetime_inicio
        try:
            if 'datetime_inicio' in df_preenchido.columns:
                df_preenchido = df_preenchido.sort_values(by=['Centro Trabalho', 'dia_produtivo', 'datetime_inicio'])
            else:
                # Caso não tenha datetime_inicio, ordenar pelo que for disponível
                if 'Data Início' in df_preenchido.columns and 'Hora Início' in df_preenchido.columns:
                    df_preenchido = df_preenchido.sort_values(by=['Centro Trabalho', 'Data Início', 'Hora Início'])
                else:
                    df_preenchido = df_preenchido.sort_values(by=['Centro Trabalho'])
        except Exception as e:
            print(f"Erro ao ordenar o DataFrame: {e}")
        
        # Função auxiliar para verificar se um valor está vazio
        def esta_vazio(valor):
            if pd.isna(valor):
                return True
            if isinstance(valor, str) and valor.strip() == '':
                return True
            return False
        
        # Lista de paradas que indicam troca de item
        paradas_troca_item = ['ACERTO', 'TROCA DE ÍTEM', 'TROCA DE ITEM', 'TROCA DE ÍTEM/OS', 'TROCA DE ITEM/OS', 'SETUP']
        
        # Processamento por dia produtivo e centro de trabalho
        for (centro, dia), grupo in df_preenchido.groupby(['Centro Trabalho', 'dia_produtivo']):
            # Índices do grupo atual, já ordenados por tempo
            indices_grupo = grupo.index.tolist()
            
            # Se o grupo está vazio, continuar para o próximo
            if not indices_grupo:
                continue
                
            # Buscar o primeiro item válido no grupo
            primeiro_item = None
            primeiro_ordem = None
            primeiro_roteiro = None
            
            for idx in indices_grupo:
                if not esta_vazio(df_preenchido.loc[idx, 'Descrição Item']):
                    primeiro_item = df_preenchido.loc[idx, 'Descrição Item']
                    primeiro_ordem = df_preenchido.loc[idx, 'Ordem Prod']
                    primeiro_roteiro = df_preenchido.loc[idx, 'Roteiro']
                    break
            
            # Se encontrou um primeiro item, preencher para cima
            # (registros anteriores ao primeiro item do dia)
            if primeiro_item is not None:
                for idx in reversed(indices_grupo):
                    # Se encontrar um registro com o item já preenchido, para de preencher para cima
                    if not esta_vazio(df_preenchido.loc[idx, 'Descrição Item']):
                        break
                        
                    # Preencher os dados com o primeiro item
                    df_preenchido.loc[idx, 'Descrição Item'] = primeiro_item
                    df_preenchido.loc[idx, 'Ordem Prod'] = primeiro_ordem
                    df_preenchido.loc[idx, 'Roteiro'] = primeiro_roteiro
            
            # Começar a preencher para baixo
            item_atual = None
            ordem_atual = None
            roteiro_atual = None
            
            # Para cada linha no grupo, ordenadas por tempo
            for i, idx in enumerate(indices_grupo):
                # Se este registro tem um item definido, atualizar o item atual
                if not esta_vazio(df_preenchido.loc[idx, 'Descrição Item']):
                    item_atual = df_preenchido.loc[idx, 'Descrição Item']
                    ordem_atual = df_preenchido.loc[idx, 'Ordem Prod']
                    roteiro_atual = df_preenchido.loc[idx, 'Roteiro']
                
                # Verificar se é uma parada específica
                if pd.notna(df_preenchido.loc[idx, 'Descrição Parada']):
                    parada = str(df_preenchido.loc[idx, 'Descrição Parada']).upper()
                    
                    # Se for uma parada de ACERTO ou TROCA DE ITEM, buscar o próximo item disponível
                    if any(tipo_parada in parada for tipo_parada in paradas_troca_item):
                        # Buscar o próximo item disponível
                        proximo_item = None
                        proximo_ordem = None
                        proximo_roteiro = None
                        
                        # Procurar o próximo item válido
                        for j in range(i+1, len(indices_grupo)):
                            prox_idx = indices_grupo[j]
                            if not esta_vazio(df_preenchido.loc[prox_idx, 'Descrição Item']):
                                proximo_item = df_preenchido.loc[prox_idx, 'Descrição Item']
                                proximo_ordem = df_preenchido.loc[prox_idx, 'Ordem Prod']
                                proximo_roteiro = df_preenchido.loc[prox_idx, 'Roteiro']
                                break
                        
                        # Se encontrou um próximo item, atualizar o item atual
                        if proximo_item is not None:
                            item_atual = proximo_item
                            ordem_atual = proximo_ordem
                            roteiro_atual = proximo_roteiro
                
                # Preencher o registro atual com os dados do item atual (se disponível)
                if item_atual is not None:
                    if esta_vazio(df_preenchido.loc[idx, 'Descrição Item']):
                        df_preenchido.loc[idx, 'Descrição Item'] = item_atual
                    if esta_vazio(df_preenchido.loc[idx, 'Ordem Prod']):
                        df_preenchido.loc[idx, 'Ordem Prod'] = ordem_atual
                    if esta_vazio(df_preenchido.loc[idx, 'Roteiro']):
                        df_preenchido.loc[idx, 'Roteiro'] = roteiro_atual
        
    elif all(col in df_preenchido.columns for col in colunas_novas):
        # Usar o formato novo (dados já processados)
        print("Detectado formato de dados processado (colunas padronizadas).")
        
        # Resetar o índice para facilitar a iteração
        df_preenchido = df_preenchido.reset_index(drop=True)
        
        # Verificar se existe a coluna dia_produtivo, caso não exista, tentar criar
        if 'dia_produtivo' not in df_preenchido.columns:
            # Tentar criar com base na data ou datetime_inicio
            try:
                if 'datetime_inicio' in df_preenchido.columns:
                    # Calcular o dia produtivo (06:00 de um dia até 06:00 do próximo)
                    df_preenchido['dia_produtivo'] = df_preenchido['datetime_inicio'].apply(
                        lambda x: (x - timedelta(days=1)).strftime('%Y-%m-%d') if pd.notna(x) and x.hour < 6 else 
                                  x.strftime('%Y-%m-%d') if pd.notna(x) else None
                    )
                elif 'data' in df_preenchido.columns:
                    # Usar a coluna data como dia_produtivo
                    df_preenchido['dia_produtivo'] = df_preenchido['data']
                else:
                    # Sem data disponível, usar a data atual
                    df_preenchido['dia_produtivo'] = datetime.now().strftime('%Y-%m-%d')
            except Exception as e:
                print(f"Erro ao criar dia_produtivo: {e}")
                # Fallback para data atual
                df_preenchido['dia_produtivo'] = datetime.now().strftime('%Y-%m-%d')
        
        # Ordenar o DataFrame por máquina, dia produtivo e data/hora
        try:
            if 'datetime_inicio' in df_preenchido.columns:
                df_preenchido = df_preenchido.sort_values(by=['maquina', 'dia_produtivo', 'datetime_inicio'])
            elif 'data' in df_preenchido.columns:
                df_preenchido = df_preenchido.sort_values(by=['maquina', 'dia_produtivo', 'data'])
            else:
                df_preenchido = df_preenchido.sort_values(by=['maquina', 'dia_produtivo'])
        except Exception as e:
            print(f"Erro ao ordenar o DataFrame: {e}")
        
        # Resetar o índice novamente após a ordenação
        df_preenchido = df_preenchido.reset_index(drop=True)
        
        # Lista de paradas que indicam troca de item
        paradas_troca_item = ['ACERTO', 'TROCA DE ÍTEM', 'TROCA DE ITEM', 'TROCA DE ÍTEM/OS', 'TROCA DE ITEM/OS', 'SETUP']
        
        # Funções auxiliares para verificar valores faltantes
        def esta_vazio(valor):
            """Verifica se um valor está vazio (None, NaN, string vazia)"""
            if pd.isna(valor):
                return True
            if isinstance(valor, str) and valor.strip() == '':
                return True
            return False
        
        # Identificar registros que têm informações completas (sem dados vazios)
        tem_item = ~df_preenchido['item'].apply(esta_vazio)
        tem_ordem = ~df_preenchido['ordem_servico'].apply(esta_vazio)
        tem_faca = ~df_preenchido['faca'].apply(esta_vazio)
        
        # Registros que têm todas as informações completas (item, ordem, faca)
        registros_completos = tem_item & tem_ordem & tem_faca
        
        # Mapa para armazenar o último item válido para cada máquina
        ultimo_item_por_maquina = {}
        
        # Criar uma coluna auxiliar que indica mudança de máquina
        df_preenchido['nova_maquina'] = False
        maquina_anterior = None
        for idx in range(len(df_preenchido)):
            maquina_atual = df_preenchido.loc[idx, 'maquina']
            if maquina_anterior != maquina_atual:
                df_preenchido.loc[idx, 'nova_maquina'] = True
                maquina_anterior = maquina_atual
        
        # Primeira passagem: preencher da frente para trás
        for idx in range(len(df_preenchido)):
            row = df_preenchido.loc[idx]
            maquina = row['maquina']
            
            # Se é uma nova máquina, limpar o histórico do último item para esta máquina
            if row['nova_maquina'] and maquina in ultimo_item_por_maquina:
                ultimo_item_por_maquina[maquina] = None
            
            # Se este registro tem todas as informações, atualizar o último item válido para esta máquina
            if registros_completos.iloc[idx]:
                ultimo_item_por_maquina[maquina] = {
                    'ordem_servico': row['ordem_servico'],
                    'item': row['item'],
                    'faca': row['faca']
                }
            
            # Se este registro tem uma parada que indica troca de item
            # Não fazer nada aqui, será tratado na próxima passagem
            elif not pd.isna(row['parada']) and any(troca in str(row['parada']).upper() for troca in paradas_troca_item):
                # Limpar as informações para esta máquina para forçar uso do próximo item
                if maquina in ultimo_item_por_maquina:
                    ultimo_item_por_maquina[maquina] = None
                    
            # Se este registro está faltando informações E temos informações válidas para esta máquina
            elif maquina in ultimo_item_por_maquina and ultimo_item_por_maquina[maquina] is not None:
                # Preencher informações faltantes
                if esta_vazio(row['ordem_servico']):
                    df_preenchido.at[idx, 'ordem_servico'] = ultimo_item_por_maquina[maquina]['ordem_servico']
                if esta_vazio(row['item']):
                    df_preenchido.at[idx, 'item'] = ultimo_item_por_maquina[maquina]['item']
                if esta_vazio(row['faca']):
                    df_preenchido.at[idx, 'faca'] = ultimo_item_por_maquina[maquina]['faca']
        
        # Segunda passagem: preencher após paradas de troca de item
        # Identificar índices de linhas com paradas de troca de item
        indices_parada_troca = []
        for idx in range(len(df_preenchido)):
            row = df_preenchido.loc[idx]
            parada = row['parada']
            
            # Se este registro tem uma parada que indica troca de item
            if not pd.isna(parada) and any(troca in str(parada).upper() for troca in paradas_troca_item):
                indices_parada_troca.append(idx)
        
        # Para cada parada de troca de item
        for idx_parada in indices_parada_troca:
            maquina = df_preenchido.loc[idx_parada, 'maquina']
            
            # Encontrar o próximo registro completo da mesma máquina
            idx_proximo = None
            for idx in range(idx_parada + 1, len(df_preenchido)):
                # Verificar se é a mesma máquina e se tem informações completas
                # OU se é a mesma máquina e tem pelo menos o item definido (mesmo sem outros dados)
                if (df_preenchido.loc[idx, 'maquina'] == maquina and registros_completos.iloc[idx]) or \
                   (df_preenchido.loc[idx, 'maquina'] == maquina and tem_item.iloc[idx]):
                    idx_proximo = idx
                    break
                    
                # Se encontrou um próximo registro completo
                if idx_proximo is not None:
                    # Obter informações do próximo registro completo
                    info_item = {
                        'ordem_servico': df_preenchido.loc[idx_proximo, 'ordem_servico'] if not esta_vazio(df_preenchido.loc[idx_proximo, 'ordem_servico']) else '',
                        'item': df_preenchido.loc[idx_proximo, 'item'],  # Item deve estar presente
                        'faca': df_preenchido.loc[idx_proximo, 'faca'] if not esta_vazio(df_preenchido.loc[idx_proximo, 'faca']) else ''
                    }                # Preencher a própria linha da parada e todas as linhas seguintes até encontrar:
                # - Outra parada de troca de item
                # - Uma mudança de máquina
                # - O próximo registro completo
                idx_atual = idx_parada
                while idx_atual <= idx_proximo:
                    row_atual = df_preenchido.loc[idx_atual]
                    
                    # Se for uma nova máquina, parar de preencher
                    if idx_atual > idx_parada and row_atual['nova_maquina']:
                        break
                    
                    # Se for o registro da parada ou um registro subsequente com dados faltantes
                    if esta_vazio(row_atual['ordem_servico']):
                        df_preenchido.at[idx_atual, 'ordem_servico'] = info_item['ordem_servico']
                    if esta_vazio(row_atual['item']):
                        df_preenchido.at[idx_atual, 'item'] = info_item['item']
                    if esta_vazio(row_atual['faca']):
                        df_preenchido.at[idx_atual, 'faca'] = info_item['faca']
                    
                    idx_atual += 1
        
        # Remover a coluna auxiliar
        df_preenchido = df_preenchido.drop(columns=['nova_maquina'])
        
    else:
        print("Não foi possível identificar o formato de dados para preenchimento.")
        return df_preenchido
    
    return df_preenchido

# Função para converter colunas de texto para números
def converter_texto_para_numero(df):
    # Faz uma cópia do DataFrame para evitar modificar o original
    df_convertido = df.copy()
    
    # Para cada coluna no DataFrame
    for coluna in df_convertido.columns:
        # Verifica se a coluna contém valores que parecem ser números armazenados como texto
        if df_convertido[coluna].dtype == 'object':  # Se for tipo objeto (possivelmente texto)
            # Tenta converter para numérico
            try:
                # Primeiro substitui vírgulas por pontos (caso esteja usando formato brasileiro)
                if isinstance(df_convertido[coluna].iloc[0], str):
                    df_convertido[coluna] = df_convertido[coluna].str.replace(',', '.')
                
                # Converte para numérico, coercivamente (forçando a conversão)
                df_convertido[coluna] = pd.to_numeric(df_convertido[coluna], errors='coerce')
                
                # Se a conversão resultou em muitos NaN, volta para a coluna original
                if df_convertido[coluna].isna().sum() > len(df_convertido) * 0.5:  # Se mais de 50% virou NaN
                    df_convertido[coluna] = df[coluna]
            except:
                # Se der erro, mantém a coluna original
                pass
                
    return df_convertido

# Função para processar os dados do Excel no formato real
def processar_dados_producao(df):
    # Verificar se o DataFrame contém as colunas necessárias (conforme definido pelo cliente)
    colunas_necessarias = [
        'Data Início', 'Hora Início', 'Data Término', 'Hora Fim',
        'Centro Trabalho', 'Ordem Prod', 'Descrição Item', 
        'Roteiro', 'Qtd Aprovada', 'Descrição Parada', 'Parada Real Útil'
    ]
    
    # Verificar se as colunas existem
    colunas_faltantes = [col for col in colunas_necessarias if col not in df.columns]
    if colunas_faltantes:
        raise ValueError(f"Colunas faltantes no arquivo: {', '.join(colunas_faltantes)}")
    
    # Criar uma cópia para processamento
    df_proc = df.copy()
    
    # Preencher dados faltantes nas linhas de parada
    print("Preenchendo dados faltantes nas linhas de parada...")
    df_proc = preencher_dados_faltantes(df_proc)
    
    # Converter a data e hora para datetime
    try:
        # Tentar formatar a data assumindo formato DD/MM/YYYY
        df_proc['data_inicio_dt'] = pd.to_datetime(df_proc['Data Início'], format='%d/%m/%Y')
    except:
        try:
            # Caso falhe, tentar formato padrão
            df_proc['data_inicio_dt'] = pd.to_datetime(df_proc['Data Início'])
        except:
            # Se ainda falhar, pode ser que a data esteja como número de série do Excel
            # Tenta converter usando a função do pandas para datas do Excel
            df_proc['data_inicio_dt'] = pd.to_datetime(pd.to_numeric(df_proc['Data Início']), 
                                          origin='1899-12-30', unit='D')
    
    # Processar hora de início
    try:
        # Tenta converter hora para datetime.time
        if 'Hora Início' in df_proc.columns:
            # Se for string no formato HH:MM:SS
            if isinstance(df_proc['Hora Início'].iloc[0], str):
                df_proc['hora_inicio_dt'] = pd.to_datetime(df_proc['Hora Início'], format='%H:%M:%S').dt.time
            # Se for float no formato Excel (fração do dia)
            elif isinstance(df_proc['Hora Início'].iloc[0], float):
                df_proc['hora_inicio_dt'] = pd.to_datetime(
                    (df_proc['Hora Início'] * 24 * 3600).apply(
                        lambda x: timedelta(seconds=x)
                    )
                ).dt.time
            # Caso seja datetime
            elif hasattr(df_proc['Hora Início'].iloc[0], 'hour'):
                df_proc['hora_inicio_dt'] = df_proc['Hora Início'].apply(lambda x: x.time())
    except Exception as e:
        print(f"Erro ao processar hora de início: {e}")
        # Valor padrão se não conseguir processar
        df_proc['hora_inicio_dt'] = datetime.strptime('00:00:00', '%H:%M:%S').time()
    
    # Combinar data e hora para criar um datetime completo
    df_proc['datetime_inicio'] = pd.to_datetime(
        df_proc['data_inicio_dt'].dt.strftime('%Y-%m-%d') + ' ' + 
        df_proc['hora_inicio_dt'].astype(str)
    )
    
    # Determinar o dia produtivo (06:00 de um dia até 06:00 do próximo) usando a função auxiliar
    df_proc['dia_produtivo'] = df_proc['datetime_inicio'].apply(calcular_dia_produtivo)
    
    # Usar o dia produtivo como a data para agrupamento
    df_proc['data'] = df_proc['dia_produtivo']
    
    # Garantir que as colunas numéricas sejam realmente numéricas
    colunas_numericas = ['Qtd Aprovada', 'Parada Real Útil']
    for col in colunas_numericas:
        try:
            df_proc[col] = pd.to_numeric(df_proc[col], errors='coerce').fillna(0)
        except:
            print(f"Erro ao converter coluna {col} para numérico. Usando valores como estão.")
    
    # Criar colunas padronizadas para o dashboard conforme mapeamento definido
    df_proc['maquina'] = df_proc['Centro Trabalho']
    df_proc['ordem_servico'] = df_proc['Ordem Prod']
    df_proc['item'] = df_proc['Descrição Item']
    df_proc['faca'] = df_proc['Roteiro']
    df_proc['producao'] = df_proc['Qtd Aprovada']
    df_proc['parada'] = df_proc['Descrição Parada']
    df_proc['tempo_parada'] = df_proc['Parada Real Útil']
    
    # Preencher dados faltantes nas linhas de parada
    df_proc = preencher_dados_faltantes(df_proc)
    
    # Classificar tipos de máquinas
    df_proc['tipo_maquina'] = df_proc['maquina'].apply(classificar_tipo_maquina)
    
    # Armazenar hora inicio para filtragem posterior
    df_proc['hora_inicio'] = df_proc['datetime_inicio'].dt.hour
    
    # Criar identificador de velocidade (Máquina-Roteiro)
    df_proc['velocidade_id'] = df_proc['maquina'] + '-' + df_proc['faca'].astype(str)
    
    # Criar um dicionário para armazenar agregações por máquina e faca
    maquina_faca_agg = {}
    tipo_maquina_agg = {}
    
    # Filtrar máquinas que não devem ser ignoradas
    df_proc_filtrado = df_proc[df_proc['tipo_maquina'] != 'Ignorar'].copy()
    df_proc_filtrado = df_proc_filtrado[df_proc_filtrado['tipo_maquina'] != 'Desativada'].copy()
    
    # Agrupar dados por máquina para análise
    maquinas = df_proc_filtrado['maquina'].unique()
    
    # Para cada máquina, criar uma coluna com a quantidade produzida
    for maquina in maquinas:
        if isinstance(maquina, str):  # Garantir que a máquina seja uma string
            col_name = maquina.replace(' ', '_')  # Formatar nome da coluna
            mask = df_proc['maquina'] == maquina
            df_proc.loc[~mask, col_name] = 0  # Para linhas que não são desta máquina, valor é 0
            df_proc.loc[mask, col_name] = pd.to_numeric(df_proc.loc[mask, 'producao'], errors='coerce').fillna(0)  # Para linhas desta máquina, valor é a produção, garantindo que seja numérico
            # Adicionar ao dicionário de agregação
            maquina_faca_agg[col_name] = 'sum'
    
    # Agrupar por tipo de máquina
    tipos_maquina = sorted(df_proc_filtrado['tipo_maquina'].unique())
    for tipo in tipos_maquina:
        if tipo not in ['Ignorar', 'Desativada']:
            tipo_col = f"tipo_{tipo.replace(' ', '_')}"
            mask = df_proc['tipo_maquina'] == tipo
            df_proc.loc[~mask, tipo_col] = 0
            df_proc.loc[mask, tipo_col] = pd.to_numeric(df_proc.loc[mask, 'producao'], errors='coerce').fillna(0)
            # Adicionar ao dicionário de agregação
            tipo_maquina_agg[tipo_col] = 'sum'
            
            # Criar também uma coluna que mapeia cada tipo para suas máquinas
            df_proc[f"maquinas_{tipo_col}"] = ""
            for idx, row in df_proc.loc[mask].iterrows():
                if isinstance(row['maquina'], str) and len(row['maquina']) > 0:
                    if df_proc.loc[idx, f"maquinas_{tipo_col}"] == "":
                        df_proc.loc[idx, f"maquinas_{tipo_col}"] = row['maquina']
                    else:
                        df_proc.loc[idx, f"maquinas_{tipo_col}"] += f", {row['maquina']}"
            
            # Adicionar ao dicionário de agregação para preservar a lista
            tipo_maquina_agg[f"maquinas_{tipo_col}"] = lambda x: list(set(filter(None, x)))
    
    # Agrupar também por tipo de parada para análise
    if len(df_proc['parada'].dropna()) > 0:  # Se houver paradas registradas
        paradas = df_proc['parada'].dropna().unique()
        
        # Para cada tipo de parada, criar uma coluna com o tempo
        for parada in paradas:
            if isinstance(parada, str) and len(parada.strip()) > 0:  # Verificar se é string válida
                col_name = f"parada_{parada.replace(' ', '_')}"[:30]  # Nome de coluna formatado
                mask = df_proc['parada'] == parada
                df_proc.loc[~mask, col_name] = 0  # Para linhas que não são desta parada, valor é 0
                df_proc.loc[mask, col_name] = df_proc.loc[mask, 'tempo_parada']  # Para linhas desta parada, valor é o tempo
                # Adicionar ao dicionário de agregação
                maquina_faca_agg[col_name] = 'sum'
    
    # Calcular eficiência com base na produção (pode ser ajustado conforme necessário)
    df_proc['eficiencia'] = 1.0  # Valor padrão, depois pode ser calculado de forma mais precisa
    
    # Incluir hora_inicio e tipo_maquina no dicionário de agregação para filtragem posterior
    maquina_faca_agg['hora_inicio'] = list  # Lista para preservar todos os valores de hora
    maquina_faca_agg['datetime_inicio'] = list  # Lista para preservar todos os valores de datetime
    maquina_faca_agg['tipo_maquina'] = list  # Lista para preservar os tipos de máquina
    
    # Combinar as agregações de máquina e tipo de máquina
    todas_agregacoes = {**maquina_faca_agg, **tipo_maquina_agg}
    
    # Agrupar por dia_produtivo para obter totais diários
    df_agrupado = df_proc.groupby('data').agg({
        'producao': 'sum',
        'eficiencia': 'mean',
        'hora_inicio': list,
        'datetime_inicio': list,
        'tipo_maquina': list,
        **todas_agregacoes
    }).reset_index()
    
    return df_agrupado

# Layout do aplicativo
app.layout = dbc.Container([
    # Store para armazenar os dados do dataset
    dcc.Store(id='store-data'),
    
    dbc.Row([
        dbc.Col(html.H1("Dashboard de Relatório de Produção", 
                        className="text-center mb-4 mt-3"), width=12)
    ]),
    
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("Upload de Dados"),
                dbc.CardBody([
                    dcc.Upload(
                        id='upload-data',
                        children=html.Div([
                            'Arraste e solte ou ',
                            html.A('selecione um arquivo Excel')
                        ]),
                        style={
                            'width': '100%',
                            'height': '60px',
                            'lineHeight': '60px',
                            'borderWidth': '1px',
                            'borderStyle': 'dashed',
                            'borderRadius': '5px',
                            'textAlign': 'center',
                            'margin': '10px'
                        },
                        multiple=False
                    ),
                    html.Div(id='output-data-upload', className="mt-3"),
                    dbc.Alert(
                        "Nenhum arquivo carregado. Por favor, faça o upload do arquivo Excel com os dados de produção.",
                        color="warning",
                        id="alert-no-data",
                        is_open=True
                    ),
                    dbc.Alert(
                        id="alert-data-loaded",
                        color="success",
                        is_open=False
                    ),
                    html.Hr(),
                    html.Div([
                        html.H5("Exportar Dados Originais Preenchidos"),
                        html.P("Baixe os dados originais com itens, ordens e roteiros preenchidos:"),
                        dbc.Button(
                            "Baixar dados originais preenchidos (Excel)", 
                            id="btn-download-xlsx", 
                            color="primary", 
                            className="me-2"
                        ),
                        dbc.Button(
                            "Baixar dados originais preenchidos (CSV)", 
                            id="btn-download-csv", 
                            color="secondary"
                        ),
                        html.P("Observação: Os arquivos incluirão todas as colunas do Excel original, com os dados faltantes de item, roteiro e ordem de serviço preenchidos automaticamente.", 
                               style={"fontSize": "0.8rem", "fontStyle": "italic", "marginTop": "10px"}),
                        dcc.Download(id="download-dataframe-xlsx"),
                        dcc.Download(id="download-dataframe-csv"),
                    ], id="download-container", style={"display": "none"})
                ])
            ], className="mb-4")
        ], width=12)
    ]),
    
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("Filtros"),
                dbc.CardBody([
                    html.Label("Modo de seleção de data:"),
                    dcc.RadioItems(
                        id='modo-data',
                        options=[
                            {'label': 'Data única', 'value': 'single'},
                            {'label': 'Intervalo de datas', 'value': 'range'}
                        ],
                        value='single',
                        inline=True
                    ),
                    html.Div(
                        id='container-data-single',
                        children=[
                            html.Label("Selecione a data do dia produtivo:"),
                            dcc.DatePickerSingle(
                                id='date-single',
                                date=datetime.now().strftime("%Y-%m-%d"),
                                display_format='DD/MM/YYYY',
                                clearable=False
                            )
                        ],
                        style={'marginTop': '10px'}
                    ),
                    html.Div(
                        id='container-data-range',
                        children=[
                            html.Label("Selecione o período:"),
                            dcc.DatePickerRange(
                                id='date-range',
                                start_date=datetime.now().strftime("%Y-%m-%d"),
                                end_date=datetime.now().strftime("%Y-%m-%d"),
                                display_format='DD/MM/YYYY'
                            )
                        ],
                        style={'display': 'none', 'marginTop': '10px'}
                    ),
                    html.Hr(),
                    html.Label("Visualização por:"),
                    dcc.RadioItems(
                        id='view-type',
                        options=[
                            {'label': 'Máquinas individuais', 'value': 'maquinas'},
                            {'label': 'Grupos de máquinas', 'value': 'grupos'}
                        ],
                        value='maquinas',
                        labelStyle={'display': 'block', 'margin': '5px 0'}
                    ),
                    html.Hr(),
                    html.Div(
                        id='container-maquinas',
                        children=[
                            html.Label("Selecione as máquinas:"),
                            dcc.Checklist(
                                id='setor-checklist',
                                options=[],  # Será preenchido dinamicamente
                                value=[],
                            )
                        ]
                    ),
                    html.Div(
                        id='container-tipos-maquinas',
                        children=[
                            html.Label("Selecione os tipos de máquinas:"),
                            html.P(
                                "Agrupamento de máquinas por tipo: Planas (CA01-CA16), Gralex (GR01-GR06B), Clamshell (MC01-MC15), Sprinter (SP01-SP02), China in Box (CH01-CH02), Janela (JA01)",
                                style={"fontSize": "0.8rem", "color": "#666", "marginBottom": "10px"}
                            ),
                            dcc.Checklist(
                                id='tipo-checklist',
                                options=[
                                    {'label': 'Planas (0)', 'value': 'tipo_Planas'},
                                    {'label': 'Gralex (0)', 'value': 'tipo_Gralex'},
                                    {'label': 'Clamshell (0)', 'value': 'tipo_Clamshell'},
                                    {'label': 'Sprinter (0)', 'value': 'tipo_Sprinter'},
                                    {'label': 'China in Box (0)', 'value': 'tipo_China_in_Box'},
                                    {'label': 'Janela (0)', 'value': 'tipo_Janela'},
                                    {'label': 'Outros (0)', 'value': 'tipo_Outros'},
                                ],
                                value=['tipo_Planas', 'tipo_Gralex', 'tipo_Clamshell', 'tipo_Sprinter', 'tipo_China_in_Box', 'tipo_Janela', 'tipo_Outros'],
                            )
                        ],
                        style={'display': 'none'}
                    )
                ])
            ], className="mb-4")
        ], width=12, lg=3),
        
        dbc.Col([
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader("Produção Total"),
                        dbc.CardBody(html.H3(id="producao-total"))
                    ], className="mb-4")
                ], width=6, md=3),
                
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader("Total de Paradas (h)"),
                        dbc.CardBody(html.H3(id="total-paradas"))
                    ], className="mb-4")
                ], width=6, md=3),
                
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader("Máquina Mais Produtiva"),
                        dbc.CardBody(html.H3(id="maquina-produtiva"))
                    ], className="mb-4")
                ], width=6, md=3),
                
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader("Total Ordens Serviço"),
                        dbc.CardBody(html.H3(id="total-ordens"))
                    ], className="mb-4")
                ], width=6, md=3),
            ]),
            
            dbc.Card([
                dbc.CardHeader("Produção Diária por Máquina"),
                dbc.CardBody(dcc.Graph(id="grafico-producao-diaria"))
            ], className="mb-4"),
            
            dbc.Card([
                dbc.CardHeader(id="header-producao-maquina", children="Produção por Máquina"),
                dbc.CardBody(dcc.Graph(id="grafico-producao-maquina"))
            ], className="mb-4"),
            
            dbc.Card([
                dbc.CardHeader("Análise de Paradas"),
                dbc.CardBody(dcc.Graph(id="grafico-paradas"))
            ], className="mb-4"),
            
            dbc.Card([
                dbc.CardHeader("Velocidade por Máquina-Faca"),
                dbc.CardBody(dcc.Graph(id="grafico-velocidade"))
            ])
        ], width=12, lg=9)
    ])
], fluid=True)

# Callback para alternar entre visualização por máquinas e por grupos
@app.callback(
    [Output('container-maquinas', 'style'),
     Output('container-tipos-maquinas', 'style')],
    [Input('view-type', 'value')]
)
def toggle_view_type(view_type):
    if view_type == 'maquinas':
        return {'display': 'block'}, {'display': 'none'}
    else:
        return {'display': 'none'}, {'display': 'block'}

# Callback para atualizar a lista de máquinas com base no tipo selecionado
@app.callback(
    [Output('setor-checklist', 'options', allow_duplicate=True),
     Output('setor-checklist', 'value', allow_duplicate=True)],
    [Input('tipo-checklist', 'value'),
     Input('store-data', 'data')],
    [State('view-type', 'value')],
    prevent_initial_call=True
)
def update_maquinas_by_tipo(tipos_selecionados, data_json, view_type):
    # Se não estivermos no modo de grupos, não atualizar
    if view_type != 'grupos' or not tipos_selecionados:
        # Verificar se temos dados
        if data_json:
            df = pd.DataFrame(data_json)
            
            # Obter todas as colunas que representam máquinas
            colunas_nao_maquinas = ['data', 'producao', 'eficiencia', 'velocidade_id', 'hora_inicio', 
                                   'datetime_inicio', 'eh_dia_atual', 'primeira_data_hora', 'tipo_maquina']
            maquinas = [col for col in df.columns 
                       if col not in colunas_nao_maquinas 
                       and not col.startswith('parada_')
                       and not col.startswith('tipo_')
                       and not col.startswith('maquinas_tipo_')
                       and not isinstance(df[col].iloc[0], list)]
                       
            # Ordenar as máquinas para uma visualização mais organizada
            maquinas = sorted(maquinas)
            opcoes_maquinas = [{'label': maquina, 'value': maquina} for maquina in maquinas]
            
            # Selecionar todas as máquinas - sem limitação
            return opcoes_maquinas, maquinas
        else:
            return [], []
    
    # Converter os dados JSON de volta para DataFrame
    df = pd.DataFrame(data_json)
    
    # Lista para armazenar todas as máquinas relacionadas aos tipos selecionados
    todas_maquinas = set()
    
    # Para cada tipo selecionado, obter as máquinas correspondentes
    for tipo in tipos_selecionados:
        try:
            # Verificar se existe a coluna que mapeia tipos para máquinas
            col_maquinas_tipo = f"maquinas_{tipo}"
            if col_maquinas_tipo in df.columns:
                # Extrair máquinas deste tipo
                for maquinas_list in df[col_maquinas_tipo]:
                    if isinstance(maquinas_list, list):
                        todas_maquinas.update(maquinas_list)
                    elif isinstance(maquinas_list, str) and maquinas_list:
                        todas_maquinas.update([maq.strip() for maq in maquinas_list.split(',')])
        except Exception as e:
            print(f"Erro ao processar máquinas do tipo {tipo}: {e}")
    
    # Converter para lista e ordenar
    lista_maquinas = sorted(list(todas_maquinas))
    
    # Criar opções para o checklist
    opcoes_maquinas = [{'label': maquina, 'value': maquina} for maquina in lista_maquinas]
    
    # Selecionar todas as máquinas do grupo, sem limite
    maquinas_selecionadas = lista_maquinas
    
    return opcoes_maquinas, maquinas_selecionadas

# Callback para processar o upload do arquivo
@app.callback(
    [Output('store-data', 'data'),
     Output('alert-no-data', 'is_open'),
     Output('alert-data-loaded', 'is_open'),
     Output('alert-data-loaded', 'children'),
     Output('date-single', 'date'),
     Output('date-range', 'start_date'),
     Output('date-range', 'end_date'),
     Output('setor-checklist', 'options'),
     Output('setor-checklist', 'value'),
     Output('tipo-checklist', 'options')],
    [Input('upload-data', 'contents')],
    [State('upload-data', 'filename')]
)
def update_output(contents, filename):
    if contents is None:
        # Sem upload - mostrar mensagem clara para carregar um arquivo
        # Usar o DataFrame vazio
        empty_df = criar_df_vazio()
        
        setores = ['Montagem', 'Pintura', 'Embalagem']
        opcoes_setores = [{'label': setor, 'value': setor} for setor in setores]
        
        # Opções de tipos de máquina padrão
        opcoes_tipos = [
            {'label': 'Planas (0)', 'value': 'tipo_Planas'},
            {'label': 'Gralex (0)', 'value': 'tipo_Gralex'},
            {'label': 'Clamshell (0)', 'value': 'tipo_Clamshell'},
            {'label': 'Sprinter (0)', 'value': 'tipo_Sprinter'},
            {'label': 'China in Box (0)', 'value': 'tipo_China_in_Box'},
            {'label': 'Janela (0)', 'value': 'tipo_Janela'},
            {'label': 'Outros (0)', 'value': 'tipo_Outros'}
        ]
        
        return (
            empty_df.to_dict('records'), 
            True,  # Mostrar alerta de "sem dados"
            False, 
            "",
            empty_df['data'].iloc[0],
            empty_df['data'].iloc[0],  # start_date
            empty_df['data'].iloc[0],  # end_date
            opcoes_setores,
            setores,
            opcoes_tipos
        )
    
    # Processar o arquivo carregado
    df, message, success = parse_contents(contents, filename)
    
    if success:
        # Se o arquivo foi processado com sucesso
        # Obter lista de máquinas disponíveis (todas colunas que não são dados básicos ou paradas)
        colunas_nao_maquinas = ['data', 'producao', 'eficiencia', 'velocidade_id', 'hora_inicio', 'datetime_inicio', 
                             'eh_dia_atual', 'primeira_data_hora', 'tipo_maquina']
        
        # Filtrar apenas máquinas reais (excluir colunas tipo_*)
        maquinas = [col for col in df.columns 
                   if col not in colunas_nao_maquinas 
                   and not col.startswith('parada_')
                   and not col.startswith('tipo_')
                   and not isinstance(df[col].iloc[0], list)]  # Não incluir colunas que contêm listas
        
        opcoes_maquinas = [{'label': maquina, 'value': maquina} for maquina in maquinas]
        
        # Obter tipos de máquina disponíveis e verificar se existem dados para cada tipo
        tipos_maquina = [col for col in df.columns if col.startswith('tipo_') and not col.startswith('tipo_maquina')]
        opcoes_tipos = []
        for tipo in tipos_maquina:
            try:
                # Verificar se é uma coluna numérica
                if isinstance(df[tipo].iloc[0], (int, float)):
                    soma = df[tipo].sum()
                    if soma > 0:
                        nome_tipo = tipo.replace('tipo_', '').replace('_', ' ')
                        opcoes_tipos.append({'label': f"{nome_tipo} ({soma:,.0f})", 'value': tipo})
                # Se for lista ou outro tipo não numérico, ignorar
                else:
                    nome_tipo = tipo.replace('tipo_', '').replace('_', ' ')
                    opcoes_tipos.append({'label': f"{nome_tipo}", 'value': tipo})
            except:
                # Em caso de erro, apenas adicionar sem informação de soma
                nome_tipo = tipo.replace('tipo_', '').replace('_', ' ')
                opcoes_tipos.append({'label': f"{nome_tipo}", 'value': tipo})
        
        today = datetime.now().strftime("%Y-%m-%d")
        data_inicial = df['data'].min() if len(df) > 0 else today
        data_final = df['data'].max() if len(df) > 0 else today
        data_mais_recente = df['data'].iloc[0] if len(df) > 0 else today
        
        return (
            df.to_dict('records'), 
            False, 
            True, 
            message,
            data_mais_recente,  # date-single
            data_inicial,       # date-range start
            data_final,         # date-range end
            opcoes_maquinas,
            maquinas,
            opcoes_tipos
        )
    else:
        # Se houve erro, mostrar mensagem de erro
        # Usar o DataFrame vazio
        empty_df = criar_df_vazio()
        
        setores = ['Montagem', 'Pintura', 'Embalagem']
        opcoes_setores = [{'label': setor, 'value': setor} for setor in setores]
        
        # Opções de tipos de máquina padrão
        opcoes_tipos = [
            {'label': 'Planas', 'value': 'tipo_Planas'},
            {'label': 'Gralex', 'value': 'tipo_Gralex'},
            {'label': 'Clamshell', 'value': 'tipo_Clamshell'},
            {'label': 'Sprinter', 'value': 'tipo_Sprinter'},
            {'label': 'China in Box', 'value': 'tipo_China_in_Box'},
            {'label': 'Janela', 'value': 'tipo_Janela'},
            {'label': 'Outros', 'value': 'tipo_Outros'}
        ]
        
        return (
            empty_df.to_dict('records'), 
            False, 
            True, 
            message,  # Mostrar a mensagem de erro
            empty_df['data'].iloc[0],
            empty_df['data'].iloc[0],  # start_date
            empty_df['data'].iloc[0],  # end_date
            opcoes_setores,
            setores,
            opcoes_tipos
        )

# Callback para alternar entre os modos de seleção de data
@app.callback(
    [Output('container-data-single', 'style'),
     Output('container-data-range', 'style')],
    [Input('modo-data', 'value')]
)
def toggle_date_mode(modo_data):
    if modo_data == 'single':
        return {'marginTop': '10px'}, {'display': 'none', 'marginTop': '10px'}
    else:
        return {'display': 'none', 'marginTop': '10px'}, {'marginTop': '10px'}

# Callback para atualizar o cabeçalho do card do gráfico de produção conforme o tipo de visualização
@app.callback(
    Output('header-producao-maquina', 'children'),
    [Input('view-type', 'value')]
)
def update_producao_header(view_type):
    if view_type == 'grupos':
        return "Produção por Tipo de Máquina"
    else:
        return "Produção por Máquina"

# Callbacks para atualizar os componentes
@app.callback(
    [Output("producao-total", "children"),
     Output("total-paradas", "children"),
     Output("maquina-produtiva", "children"),
     Output("total-ordens", "children"),
     Output("grafico-producao-diaria", "figure"),
     Output("grafico-producao-maquina", "figure"),
     Output("grafico-paradas", "figure"),
     Output("grafico-velocidade", "figure")],
    [Input("modo-data", "value"),
     Input("date-single", "date"),
     Input("date-range", "start_date"),
     Input("date-range", "end_date"),
     Input("view-type", "value"),
     Input("setor-checklist", "value"),
     Input("tipo-checklist", "value"),
     Input("store-data", "data")]
)
def update_dashboard(modo_data, selected_date, start_date, end_date, view_type, maquinas_selecionadas, tipos_selecionados, data_json):
    # Converter os dados JSON de volta para DataFrame
    df = pd.DataFrame(data_json)
    
    # Converter as listas armazenadas de volta para objetos Python
    if 'datetime_inicio' in df.columns:
        # Se os datetimes estão armazenados como listas de strings, converter de volta para datetime
        try:
            if isinstance(df['datetime_inicio'].iloc[0], list):
                # Criar uma nova coluna com o primeiro datetime de cada linha para facilitar a filtragem
                df['primeira_data_hora'] = df['datetime_inicio'].apply(
                    lambda x: pd.to_datetime(x[0]) if x and len(x) > 0 else None
                )
        except:
            pass
    
    # Aplicar filtro com base no modo de seleção de data
    if modo_data == 'single':
        # Filtrar dados para o dia produtivo selecionado
        mask = df['data'] == selected_date
        periodo_str = f"no Dia {selected_date}"
    else:
        # Filtrar dados para o período selecionado
        mask = (df['data'] >= start_date) & (df['data'] <= end_date)
        periodo_str = f"no Período de {start_date} a {end_date}"
    
    df_filtrado = df.loc[mask].copy()
    
    # Adicionar informação de dia produtivo para facilitar análises
    df_filtrado['eh_dia_atual'] = True
    
    # Se o dataframe estiver vazio ou muito pequeno, retornar gráficos vazios
    if len(df_filtrado) == 0:
        empty_fig = px.line(title="Sem dados disponíveis para o período selecionado")
        empty_fig.update_layout(
            xaxis={"visible": False},
            yaxis={"visible": False},
            annotations=[
                {
                    "text": "Sem dados disponíveis para o período selecionado",
                    "xref": "paper",
                    "yref": "paper",
                    "showarrow": False,
                    "font": {"size": 16}
                }
            ]
        )
        return (
            "0", "0", "N/A", "0",
            empty_fig, empty_fig, empty_fig, empty_fig
        )
    
    # Calcular métricas principais com base no modo de visualização e seleções
    if view_type == 'maquinas' and maquinas_selecionadas:
        # Verificar quais máquinas selecionadas existem no DataFrame
        maquinas_existentes = [col for col in maquinas_selecionadas if col in df_filtrado.columns]
        
        if maquinas_existentes:
            # Converter colunas para numéricas
            df_maq_numerico = df_filtrado.copy()
            for col in maquinas_existentes:
                df_maq_numerico[col] = pd.to_numeric(df_maq_numerico[col], errors='coerce').fillna(0)
            
            # Somar produção apenas das máquinas selecionadas
            producao_total = sum(df_maq_numerico[col].sum() for col in maquinas_existentes)
        else:
            producao_total = 0
    elif view_type == 'grupos' and tipos_selecionados:
        # Verificar quais tipos selecionados existem no DataFrame
        tipos_validos = [tipo for tipo in tipos_selecionados if tipo in df_filtrado.columns]
        
        if tipos_validos:
            # Converter colunas para numéricas
            df_tipo_numerico = df_filtrado.copy()
            for col in tipos_validos:
                df_tipo_numerico[col] = pd.to_numeric(df_tipo_numerico[col], errors='coerce').fillna(0)
            
            # Somar produção apenas dos tipos selecionados
            producao_total = sum(df_tipo_numerico[col].sum() for col in tipos_validos)
        else:
            producao_total = 0
    else:
        # Fallback para o cálculo original caso nada esteja selecionado
        producao_total = df_filtrado['producao'].sum()
    
    # Total de paradas (em horas) - filtrando pelo mesmo critério de máquinas/grupos
    colunas_paradas = [col for col in df_filtrado.columns if col.startswith('parada_')]
    
    if view_type == 'maquinas' and maquinas_selecionadas:
        # Para modo de máquinas, filtrar paradas relacionadas às máquinas selecionadas
        if colunas_paradas:
            total_paradas = 0
            # Verificar quais máquinas selecionadas existem no DataFrame
            maquinas_existentes = [col for col in maquinas_selecionadas if col in df_filtrado.columns]
            
            if maquinas_existentes:
                try:
                    # Criar uma máscara para filtrar apenas registros das máquinas selecionadas
                    mascara_maquinas = pd.Series(False, index=df_filtrado.index)
                    for maquina in maquinas_existentes:
                        # Adicionar à máscara registros onde o valor da máquina é > 0
                        maquina_valor = pd.to_numeric(df_filtrado[maquina], errors='coerce').fillna(0)
                        mascara_maquinas |= maquina_valor > 0
                    
                    # Se temos registros filtrados
                    if mascara_maquinas.any():
                        # Filtrar o DataFrame para incluir apenas os registros das máquinas selecionadas
                        df_maquinas = df_filtrado[mascara_maquinas]
                        
                        # Somar as paradas apenas para os registros filtrados
                        if not df_maquinas.empty:
                            total_paradas = df_maquinas[colunas_paradas].sum().sum()
                    else:
                        total_paradas = 0
                except Exception as e:
                    print(f"Erro ao calcular total de paradas para máquinas: {e}")
                    total_paradas = 0
            else:
                total_paradas = 0
        else:
            total_paradas = 0
    elif view_type == 'grupos' and tipos_selecionados:
        # Para modo de grupos, filtrar paradas relacionadas aos tipos selecionados
        if colunas_paradas:
            total_paradas = 0
            # Verificar quais tipos selecionados existem no DataFrame
            tipos_validos = [tipo for tipo in tipos_selecionados if tipo in df_filtrado.columns]
            
            if tipos_validos:
                try:
                    # Criar uma máscara para filtrar apenas registros dos tipos selecionados
                    mascara_tipos = pd.Series(False, index=df_filtrado.index)
                    for tipo in tipos_validos:
                        # Adicionar à máscara registros onde o valor do tipo é > 0
                        tipo_valor = pd.to_numeric(df_filtrado[tipo], errors='coerce').fillna(0)
                        mascara_tipos |= tipo_valor > 0
                    
                    # Se temos registros filtrados
                    if mascara_tipos.any():
                        # Filtrar o DataFrame para incluir apenas os registros dos tipos selecionados
                        df_tipos = df_filtrado[mascara_tipos]
                        
                        # Somar as paradas apenas para os registros filtrados
                        if not df_tipos.empty:
                            total_paradas = df_tipos[colunas_paradas].sum().sum()
                    else:
                        total_paradas = 0
                except Exception as e:
                    print(f"Erro ao calcular total de paradas para tipos: {e}")
                    total_paradas = 0
            else:
                total_paradas = 0
        else:
            total_paradas = 0
    else:
        # Fallback para o cálculo original
        if colunas_paradas:
            total_paradas = df_filtrado[colunas_paradas].sum().sum()
        else:
            total_paradas = 0
    
    # Identificar a máquina mais produtiva
    colunas_maquinas = maquinas_selecionadas if maquinas_selecionadas else [col for col in df_filtrado.columns if col not in ['data', 'producao', 'eficiencia', 'hora_inicio', 'datetime_inicio', 'eh_dia_atual', 'primeira_data_hora'] 
                                             and not col.startswith('parada_')
                                             and not col == 'velocidade_id']
    
    if colunas_maquinas:
        # Filtrar colunas que realmente existem no DataFrame
        colunas_existentes = [col for col in colunas_maquinas if col in df_filtrado.columns]
        
        if colunas_existentes:
            # Garantir que todas as colunas sejam numéricas antes de fazer a soma
            df_numerico = df_filtrado[colunas_existentes].apply(pd.to_numeric, errors='coerce').fillna(0)
            
            try:
                # Tentar encontrar o máximo de forma segura
                maquina_prod = df_numerico.sum().idxmax()
                valor_maquina_prod = df_numerico.sum().max()
                maquina_produtiva = f"{maquina_prod} ({valor_maquina_prod:,.0f})".replace(",", ".")
            except (TypeError, ValueError, KeyError) as e:
                print(f"Erro ao identificar máquina mais produtiva: {e}")
                maquina_produtiva = "N/A"
        else:
            maquina_produtiva = "N/A"
    else:
        maquina_produtiva = "N/A"
    
    # Contagem de ordens de serviço únicas (quando disponível)
    total_ordens = 0
    
    # Como as ordens de serviço são específicas para cada registro/linha,
    # a filtragem deve ser feita com base nas máquinas/grupos selecionados
    if 'ordem_servico' in df_filtrado.columns:
        if view_type == 'maquinas' and maquinas_selecionadas:
            # Verificar quais máquinas selecionadas existem no DataFrame
            maquinas_existentes = [col for col in maquinas_selecionadas if col in df_filtrado.columns]
            if maquinas_existentes:
                try:
                    # Criar uma máscara para filtrar apenas registros das máquinas selecionadas
                    mascara_maquinas = pd.Series(False, index=df_filtrado.index)
                    for maquina in maquinas_existentes:
                        # Adicionar à máscara registros onde o valor da máquina é > 0
                        mascara_maquinas |= pd.to_numeric(df_filtrado[maquina], errors='coerce').fillna(0) > 0
                    
                    # Filtrar o DataFrame e contar ordens únicas
                    if mascara_maquinas.any():
                        df_ordens = df_filtrado[mascara_maquinas]
                        total_ordens = df_ordens['ordem_servico'].nunique()
                    else:
                        total_ordens = 0
                except Exception as e:
                    print(f"Erro ao calcular total de ordens para máquinas: {e}")
                    total_ordens = 0
            
        elif view_type == 'grupos' and tipos_selecionados:
            # Verificar quais tipos selecionados existem no DataFrame
            tipos_validos = [tipo for tipo in tipos_selecionados if tipo in df_filtrado.columns]
            if tipos_validos:
                try:
                    # Criar uma máscara para filtrar apenas registros dos tipos selecionados
                    mascara_tipos = pd.Series(False, index=df_filtrado.index)
                    for tipo in tipos_validos:
                        # Adicionar à máscara registros onde o valor do tipo é > 0
                        mascara_tipos |= pd.to_numeric(df_filtrado[tipo], errors='coerce').fillna(0) > 0
                    
                    # Filtrar o DataFrame e contar ordens únicas
                    if mascara_tipos.any():
                        df_ordens = df_filtrado[mascara_tipos]
                        total_ordens = df_ordens['ordem_servico'].nunique()
                    else:
                        total_ordens = 0
                except Exception as e:
                    print(f"Erro ao calcular total de ordens para tipos: {e}")
                    total_ordens = 0
            
        else:
            # Se nenhum filtro específico, usar o total geral
            total_ordens = df_filtrado['ordem_servico'].nunique()
    
    # 1. Gráfico de produção
    if modo_data == 'single' and 'hora_inicio' in df_filtrado.columns and isinstance(df_filtrado['hora_inicio'].iloc[0], list) and len(df_filtrado) > 0:
        # Para data única: tentar mostrar distribuição por hora
        try:
            # Criar um DataFrame auxiliar para o gráfico
            dados_hora = []
            for idx, row in df_filtrado.iterrows():
                if isinstance(row['hora_inicio'], list) and len(row['hora_inicio']) > 0:
                    for hora in row['hora_inicio']:
                        hora_int = int(hora) if isinstance(hora, (int, float)) else 0
                        dados_hora.append({
                            'Hora': hora_int,
                            'Produção': row['producao'] / len(row['hora_inicio'])  # Distribuir produção igualmente
                        })
            
            if dados_hora:
                df_hora = pd.DataFrame(dados_hora)
                df_hora = df_hora.groupby('Hora').sum().reset_index()
                fig_producao_diaria = px.bar(df_hora, x='Hora', y='Produção',
                                           title=f'Produção por Hora {periodo_str}',
                                           labels={'Produção': 'Quantidade', 'Hora': 'Hora do Dia'})
            else:
                fig_producao_diaria = px.bar(title=f'Produção Total {periodo_str}')
                fig_producao_diaria.update_layout(
                    annotations=[{"text": f"Produção total: {df_filtrado['producao'].sum():,.0f}".replace(',', '.'),
                                "showarrow": False, "font": {"size": 16}, "x": 0.5, "y": 0.5}]
                )
        except Exception as e:
            print(f"Erro ao criar gráfico por hora: {e}")
            fig_producao_diaria = px.bar(title=f'Produção Total {periodo_str}')
            fig_producao_diaria.update_layout(
                annotations=[{"text": f"Produção total: {df_filtrado['producao'].sum():,.0f}".replace(',', '.'),
                            "showarrow": False, "font": {"size": 16}, "x": 0.5, "y": 0.5}]
            )
    elif modo_data == 'range' and len(df_filtrado) > 0:
        # Para intervalo de datas: mostrar produção diária
        df_dia = df_filtrado.groupby('data')['producao'].sum().reset_index()
        if len(df_dia) > 0:
            fig_producao_diaria = px.line(df_dia, x='data', y='producao',
                                        title=f'Produção Diária {periodo_str}',
                                        labels={'producao': 'Quantidade', 'data': 'Data'})
            fig_producao_diaria.update_layout(legend_title_text='')
        else:
            fig_producao_diaria = px.line(title=f'Produção Total {periodo_str}')
            fig_producao_diaria.update_layout(
                annotations=[{"text": f"Sem dados disponíveis para o período", 
                            "showarrow": False, "font": {"size": 16}, "x": 0.5, "y": 0.5}]
            )
    else:
        # Gráfico padrão quando não houver dados
        fig_producao_diaria = px.bar(title=f'Produção Total {periodo_str}')
        fig_producao_diaria.update_layout(
            annotations=[{"text": f"Produção total: {df_filtrado['producao'].sum():,.0f}".replace(',', '.'),
                        "showarrow": False, "font": {"size": 16}, "x": 0.5, "y": 0.5}]
        )
    
    # 2. Gráfico de produção por máquina/grupo
    if view_type == 'maquinas' and maquinas_selecionadas:
        # Verificar quais máquinas selecionadas realmente existem no DataFrame
        maquinas_existentes = [col for col in maquinas_selecionadas if col in df_filtrado.columns]
        
        # Se não houver máquinas válidas, criar um DataFrame vazio
        if not maquinas_existentes:
            df_maquinas = pd.DataFrame({'Máquina': ['Sem dados'], 'Produção': [0]})
        else:
            # Visualização por máquinas individuais
            # Garantir que as colunas de máquinas selecionadas sejam numéricas
            df_maq_numerico = df_filtrado.copy()
            for col in maquinas_existentes:
                df_maq_numerico[col] = pd.to_numeric(df_maq_numerico[col], errors='coerce').fillna(0)
            
            if modo_data == 'single':
                # Para data única, criar DataFrame sem a coluna de data
                df_maquinas = pd.DataFrame({
                    'Máquina': maquinas_existentes,
                    'Produção': [df_maq_numerico[col].sum() for col in maquinas_existentes]
                })
            else:
                # Para intervalo de datas, incluir a coluna de data para agrupar os dados
                if len(maquinas_existentes) > 0:
                    df_maquinas = df_maq_numerico[['data'] + maquinas_existentes].melt(
                        id_vars=['data'],
                        value_vars=maquinas_existentes,
                        var_name='Máquina',
                        value_name='Produção'
                    )
                else:
                    # Se não houver máquinas para processar
                    df_maquinas = pd.DataFrame({'Máquina': ['Sem dados'], 'Produção': [0]})
    elif view_type == 'grupos' and tipos_selecionados:
        # Visualização por grupos de máquinas
        # Verificar quais tipos selecionados realmente existem no DataFrame
        tipos_validos = [tipo for tipo in tipos_selecionados if tipo in df_filtrado.columns]
        
        if not tipos_validos:
            df_maquinas = pd.DataFrame({'Máquina': ['Sem dados'], 'Produção': [0]})
        else:
            # Garantir que as colunas de tipos selecionadas sejam numéricas
            df_tipo_numerico = df_filtrado.copy()
            
            for col in tipos_validos:
                # Verificar se a coluna existe e contém valores numéricos
                try:
                    df_tipo_numerico[col] = pd.to_numeric(df_tipo_numerico[col], errors='coerce').fillna(0)
                except Exception as e:
                    print(f"Erro ao processar coluna {col}: {e}")
        
        if modo_data == 'single' and tipos_validos:
            # Para data única, criar DataFrame sem a coluna de data
            df_maquinas = pd.DataFrame({
                'Tipo': [tipo.replace('tipo_', '').replace('_', ' ') for tipo in tipos_validos],
                'Produção': [df_tipo_numerico[col].sum() for col in tipos_validos]
            })
            # Renomear a coluna para manter compatibilidade com o resto do código
            df_maquinas.rename(columns={'Tipo': 'Máquina'}, inplace=True)
        elif tipos_validos:
            # Para intervalo de datas, incluir a coluna de data para agrupar os dados
            df_maquinas = df_tipo_numerico[['data'] + tipos_validos].melt(
                id_vars=['data'],
                value_vars=tipos_validos,
                var_name='Tipo',
                value_name='Produção'
            )
            # Formatar os nomes dos tipos para exibição
            df_maquinas['Tipo'] = df_maquinas['Tipo'].apply(lambda x: x.replace('tipo_', '').replace('_', ' '))
            # Renomear a coluna para manter compatibilidade com o resto do código
            df_maquinas.rename(columns={'Tipo': 'Máquina'}, inplace=True)
        else:
            # Se não houver tipos válidos selecionados
            df_maquinas = pd.DataFrame({'Máquina': ['Sem dados'], 'Produção': [0]})
    else:
        # Se nenhuma máquina ou tipo estiver selecionado, usar valores padrão
        df_maquinas = pd.DataFrame({'Máquina': ['Sem dados'], 'Produção': [0]})
        if len(df_filtrado) > 0:
            cols_maquinas = [col for col in df_filtrado.columns 
                            if col not in ['data', 'producao', 'eficiencia', 'hora_inicio', 'datetime_inicio', 'eh_dia_atual', 'primeira_data_hora']
                            and not col.startswith('parada_') 
                            and not col == 'velocidade_id'
                            and not isinstance(df_filtrado[col].iloc[0], list)]
        
        if cols_maquinas and len(df_filtrado) > 0:
            # Garantir que as colunas sejam numéricas
            df_maq_numerico = df_filtrado.copy()
            for col in cols_maquinas:
                df_maq_numerico[col] = pd.to_numeric(df_maq_numerico[col], errors='coerce').fillna(0)
            
            if modo_data == 'single':
                # Para data única, criar DataFrame sem a coluna de data
                df_maquinas = pd.DataFrame({
                    'Máquina': cols_maquinas,
                    'Produção': [df_maq_numerico[col].sum() for col in cols_maquinas]
                })
            else:
                # Para intervalo de datas, incluir a coluna de data para agrupar os dados
                df_maquinas = df_maq_numerico[['data'] + cols_maquinas].melt(
                    id_vars=['data'],
                    value_vars=cols_maquinas,
                    var_name='Máquina',
                    value_name='Produção'
                )
        else:
            # Fallback se não houver colunas de máquinas
            df_maquinas = pd.DataFrame({'Máquina': ['N/A'], 'Produção': [0]})
    
    # Ajustar o gráfico dependendo do modo de seleção e tipo de visualização
    titulo_grafico = f'Produção por {"Tipo de Máquina" if view_type == "grupos" else "Máquina"} {periodo_str}'
    
    if modo_data == 'single':
        # Para data única: mostrar barras por máquina/grupo
        fig_producao_maquina = px.bar(df_maquinas, x='Máquina', y='Produção', 
                                    title=titulo_grafico)
    else:
        # Para intervalo de datas: mostrar barras agrupadas por data e máquina/grupo
        if 'data' in df_maquinas.columns:
            fig_producao_maquina = px.bar(df_maquinas, x='data', y='Produção', color='Máquina', barmode='stack',
                                       title=titulo_grafico)
        else:
            fig_producao_maquina = px.bar(df_maquinas, x='Máquina', y='Produção', 
                                       title=titulo_grafico)
    
    # 3. Gráfico de análise de paradas
    colunas_paradas = [col for col in df_filtrado.columns if col.startswith('parada_')]
    if colunas_paradas:
        # Calcular total de paradas por tipo
        df_paradas_sum = df_filtrado[colunas_paradas].sum().reset_index()
        df_paradas_sum.columns = ['Tipo de Parada', 'Tempo (horas)']
        df_paradas_sum['Tipo de Parada'] = df_paradas_sum['Tipo de Parada'].str.replace('parada_', '')
        
        fig_paradas = px.pie(df_paradas_sum, values='Tempo (horas)', names='Tipo de Parada',
                          title=f'Distribuição de Paradas {periodo_str}')
    else:
        # Criar gráfico vazio se não houver dados de paradas
        fig_paradas = px.pie(title="Sem dados de paradas disponíveis")
        fig_paradas.update_layout(
            annotations=[{"text": "Sem dados de paradas disponíveis", "showarrow": False, "font": {"size": 16}}]
        )
    
    # 4. Gráfico de velocidade por máquina-faca
    # Tentar calcular a velocidade como produção / tempo para cada combinação máquina-faca
    if 'velocidade_id' in df_filtrado.columns and len(df_filtrado['velocidade_id'].unique()) > 1:
        # Agrupar por velocidade_id e calcular métricas
        df_velocidade = df_filtrado.groupby('velocidade_id').agg({
            'producao': 'sum'
        }).reset_index()
        df_velocidade = df_velocidade.sort_values('producao', ascending=False).head(10)  # Top 10 combinações
        
        fig_velocidade = px.bar(df_velocidade, x='velocidade_id', y='producao',
                               title=f'Produção por Máquina-Faca {periodo_str} (Top 10)',
                               labels={'velocidade_id': 'Máquina-Faca', 'producao': 'Produção Total'})
    else:
        # Criar gráfico vazio se não houver dados de velocidade
        fig_velocidade = px.bar(title="Sem dados de velocidade disponíveis")
        fig_velocidade.update_layout(
            annotations=[{"text": "Sem dados de velocidade disponíveis", "showarrow": False, "font": {"size": 16}}]
        )
    
    return (
        f"{producao_total:,}".replace(",", "."),
        f"{total_paradas:.1f}",
        maquina_produtiva,
        f"{total_ordens:,}".replace(",", "."),
        fig_producao_diaria,
        fig_producao_maquina,
        fig_paradas,
        fig_velocidade
    )

# Callback para mostrar/esconder os botões de download
# Comentado porque agora estamos usando um clientside callback na função parse_contents
# @app.callback(
#     Output("download-container", "style"),
#     [Input("alert-data-loaded", "is_open")]
# )
# def toggle_download_buttons(is_data_loaded):
#     if is_data_loaded:
#         return {"display": "block"}
#     return {"display": "none"}

# Callbacks para download dos dados
@app.callback(
    Output("download-dataframe-xlsx", "data"),
    [Input("btn-download-xlsx", "n_clicks")]
)
def download_xlsx(n_clicks):
    if n_clicks is None:
        raise dash.exceptions.PreventUpdate
    
    # Usar o DataFrame original preenchido com dados faltantes para exportação
    global df_original_global
    if df_original_global is not None:
        return dcc.send_data_frame(df_original_global.to_excel, "dados_preenchidos.xlsx", sheet_name="Dados Preenchidos", index=False)
    return None

@app.callback(
    Output("download-dataframe-csv", "data"),
    [Input("btn-download-csv", "n_clicks")]
)
def download_csv(n_clicks):
    if n_clicks is None:
        raise dash.exceptions.PreventUpdate
    
    # Usar o DataFrame original preenchido com dados faltantes para exportação
    global df_original_global
    if df_original_global is not None:
        return dcc.send_data_frame(df_original_global.to_csv, "dados_preenchidos.csv", index=False)
    return None

# Iniciar o servidor
if __name__ == '__main__':
    app.run(debug=True)