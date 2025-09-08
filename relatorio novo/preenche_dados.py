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
    import pandas as pd
    from datetime import datetime, timedelta
    
    # Criar uma cópia para não modificar o original
    df_preenchido = df.copy()
    
    # Verificar se as colunas necessárias estão presentes (colunas antigas)
    colunas_antigas = ['Centro Trabalho', 'Descrição Parada', 'Ordem Prod', 'Descrição Item', 'Roteiro']
    
    # Verificar se as colunas necessárias estão presentes (colunas novas)
    colunas_novas = ['maquina', 'parada', 'ordem_servico', 'item', 'faca']
    
    # Função auxiliar para verificar valores faltantes
    def esta_vazio(valor):
        """Verifica se um valor está vazio (None, NaN, string vazia)"""
        if pd.isna(valor):
            return True
        if isinstance(valor, str) and valor.strip() == '':
            return True
        return False
    
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
                    # Verificar se este registro está exatamente às 06:00
                    hora_exata_06 = False
                    if 'datetime_inicio' in df_preenchido.columns and pd.notna(df_preenchido.loc[idx, 'datetime_inicio']):
                        dt = df_preenchido.loc[idx, 'datetime_inicio']
                        if dt.hour == 6 and dt.minute == 0:
                            hora_exata_06 = True
                    
                    if hora_exata_06:
                        # Para registros às 06:00, buscar o próximo item abaixo
                        proximo_item = None
                        proximo_ordem = None
                        proximo_roteiro = None
                        for j in range(indices_grupo.index(idx) + 1, len(indices_grupo)):
                            prox_idx = indices_grupo[j]
                            if not esta_vazio(df_preenchido.loc[prox_idx, 'Descrição Item']):
                                proximo_item = df_preenchido.loc[prox_idx, 'Descrição Item']
                                proximo_ordem = df_preenchido.loc[prox_idx, 'Ordem Prod']
                                proximo_roteiro = df_preenchido.loc[prox_idx, 'Roteiro']
                                break
                        if proximo_item is not None:
                            primeiro_item = proximo_item
                            primeiro_ordem = proximo_ordem
                            primeiro_roteiro = proximo_roteiro
                        else:
                            # Se não encontrou próximo, usar este mesmo
                            primeiro_item = df_preenchido.loc[idx, 'Descrição Item']
                            primeiro_ordem = df_preenchido.loc[idx, 'Ordem Prod']
                            primeiro_roteiro = df_preenchido.loc[idx, 'Roteiro']
                    else:
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
                
        # Processamento por máquina e dia produtivo
        for (maquina, dia), grupo in df_preenchido.groupby(['maquina', 'dia_produtivo']):
            # Obter os índices do grupo atual, já ordenados por tempo
            indices_grupo = grupo.index.tolist()
            
            # Se o grupo está vazio, continuar para o próximo
            if not indices_grupo:
                continue
                
            # Buscar o primeiro item válido no grupo
            primeiro_item = None
            primeiro_ordem = None
            primeiro_faca = None
            
            for idx in indices_grupo:
                if not esta_vazio(df_preenchido.loc[idx, 'item']):
                    # Verificar se este registro está exatamente às 06:00
                    hora_exata_06 = False
                    if 'datetime_inicio' in df_preenchido.columns and pd.notna(df_preenchido.loc[idx, 'datetime_inicio']):
                        dt = df_preenchido.loc[idx, 'datetime_inicio']
                        if dt.hour == 6 and dt.minute == 0:
                            hora_exata_06 = True
                    
                    if hora_exata_06:
                        # Para registros às 06:00, buscar o próximo item abaixo
                        proximo_item = None
                        proximo_ordem = None
                        proximo_faca = None
                        for j in range(indices_grupo.index(idx) + 1, len(indices_grupo)):
                            prox_idx = indices_grupo[j]
                            if not esta_vazio(df_preenchido.loc[prox_idx, 'item']):
                                proximo_item = df_preenchido.loc[prox_idx, 'item']
                                proximo_ordem = df_preenchido.loc[prox_idx, 'ordem_servico']
                                proximo_faca = df_preenchido.loc[prox_idx, 'faca']
                                break
                        if proximo_item is not None:
                            primeiro_item = proximo_item
                            primeiro_ordem = proximo_ordem
                            primeiro_faca = proximo_faca
                        else:
                            # Se não encontrou próximo, usar este mesmo
                            primeiro_item = df_preenchido.loc[idx, 'item']
                            primeiro_ordem = df_preenchido.loc[idx, 'ordem_servico']
                            primeiro_faca = df_preenchido.loc[idx, 'faca']
                    else:
                        primeiro_item = df_preenchido.loc[idx, 'item']
                        primeiro_ordem = df_preenchido.loc[idx, 'ordem_servico']
                        primeiro_faca = df_preenchido.loc[idx, 'faca']
                    break
            
            # Se encontrou um primeiro item, preencher para cima
            # (registros anteriores ao primeiro item do dia)
            if primeiro_item is not None:
                for idx in reversed(indices_grupo):
                    # Se encontrar um registro com o item já preenchido, para de preencher para cima
                    if not esta_vazio(df_preenchido.loc[idx, 'item']):
                        break
                        
                    # Preencher os dados com o primeiro item
                    df_preenchido.loc[idx, 'item'] = primeiro_item
                    if esta_vazio(df_preenchido.loc[idx, 'ordem_servico']):
                        df_preenchido.loc[idx, 'ordem_servico'] = primeiro_ordem
                    if esta_vazio(df_preenchido.loc[idx, 'faca']):
                        df_preenchido.loc[idx, 'faca'] = primeiro_faca
            
            # Começar a preencher para baixo
            item_atual = None
            ordem_atual = None
            faca_atual = None
            
            # Para cada linha no grupo, ordenadas por tempo
            for i, idx in enumerate(indices_grupo):
                # Se este registro tem um item definido, atualizar o item atual
                if not esta_vazio(df_preenchido.loc[idx, 'item']):
                    item_atual = df_preenchido.loc[idx, 'item']
                    ordem_atual = df_preenchido.loc[idx, 'ordem_servico']
                    faca_atual = df_preenchido.loc[idx, 'faca']
                
                # Verificar se é uma parada específica
                if pd.notna(df_preenchido.loc[idx, 'parada']):
                    parada = str(df_preenchido.loc[idx, 'parada']).upper()
                    
                    # Se for uma parada de ACERTO ou TROCA DE ITEM, buscar o próximo item disponível
                    if any(tipo_parada in parada for tipo_parada in paradas_troca_item):
                        # Buscar o próximo item disponível
                        proximo_item = None
                        proximo_ordem = None
                        proximo_faca = None
                        
                        # Procurar o próximo item válido
                        for j in range(i+1, len(indices_grupo)):
                            prox_idx = indices_grupo[j]
                            if not esta_vazio(df_preenchido.loc[prox_idx, 'item']):
                                proximo_item = df_preenchido.loc[prox_idx, 'item']
                                proximo_ordem = df_preenchido.loc[prox_idx, 'ordem_servico']
                                proximo_faca = df_preenchido.loc[prox_idx, 'faca']
                                break
                        
                        # Se encontrou um próximo item, atualizar o item atual
                        if proximo_item is not None:
                            item_atual = proximo_item
                            ordem_atual = proximo_ordem
                            faca_atual = proximo_faca
                
                # Preencher o registro atual com os dados do item atual (se disponível)
                if item_atual is not None:
                    if esta_vazio(df_preenchido.loc[idx, 'item']):
                        df_preenchido.loc[idx, 'item'] = item_atual
                    if esta_vazio(df_preenchido.loc[idx, 'ordem_servico']):
                        df_preenchido.loc[idx, 'ordem_servico'] = ordem_atual
                    if esta_vazio(df_preenchido.loc[idx, 'faca']):
                        df_preenchido.loc[idx, 'faca'] = faca_atual
    else:
        print("Não foi possível identificar o formato de dados para preenchimento.")
        return df_preenchido
    
    return df_preenchido
