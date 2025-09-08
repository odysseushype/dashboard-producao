# Função para preencher dados faltantes nas linhas de parada
def preencher_dados_faltantes(df):
    """
    Preenche dados faltantes de roteiro, item e ordem de produção nas linhas onde só existe registro de parada.
    Segue as regras:
    1. Usa informações do primeiro item disponível para preencher para cima e para baixo
    2. Quando Centro de Trabalho muda, usa o próximo item disponível
    3. Quando encontra paradas de "Acerto" ou "TROCA DE ÍTEM/OS", usa o próximo item disponível
    
    Args:
        df: DataFrame com os dados carregados do Excel
        
    Returns:
        DataFrame com os dados faltantes preenchidos
    """
    import pandas as pd
    
    # Criar uma cópia para não modificar o original
    df_preenchido = df.copy()

    # Helper para identificar vazio (NaN, None, string vazia ou só espaços)
    def _eh_vazio(v):
        if v is None:
            return True
        if isinstance(v, float) and pd.isna(v):
            return True
        if isinstance(v, str) and v.strip() == "":
            return True
        return False
    
    # Verificar se as colunas necessárias estão presentes (colunas antigas)
    colunas_antigas = ['Centro Trabalho', 'Descrição Parada', 'Ordem Prod', 'Descrição Item', 'Roteiro']
    
    # Verificar se as colunas necessárias estão presentes (colunas novas)
    colunas_novas = ['maquina', 'parada', 'ordem_servico', 'item', 'faca']
    
    # Determinar qual formato de colunas usar
    if all(col in df_preenchido.columns for col in colunas_antigas):
        # Usar o formato antigo (dados brutos do Excel)
        print("Detectado formato de dados original (colunas originais do Excel).")

        # ================= TRATATIVA Qtd Retrabalhada (formato antigo) =================
        # Identificar coluna de retrabalho (qualquer nome que contenha 'retrabal')
        retrab_cols = [c for c in df_preenchido.columns if c.strip().lower() in ('qtd retrabalhada','qtd_retrabalhada')]
        if not retrab_cols:
            retrab_cols = [c for c in df_preenchido.columns if 'retrabal' in c.lower()]
        if retrab_cols:
            col_retr = retrab_cols[0]
            item_col = 'Descrição Item' if 'Descrição Item' in df_preenchido.columns else None
            if item_col:
                # Ordenar antes (será reordenado novamente depois de qualquer forma)
                try:
                    if 'datetime_inicio' in df_preenchido.columns:
                        df_preenchido = df_preenchido.sort_values(by=['Centro Trabalho','datetime_inicio'])
                except Exception:
                    pass
                to_drop = []
                valores = df_preenchido[col_retr]
                for idx in df_preenchido.index:
                    try:
                        q = valores.loc[idx]
                        q_num = pd.to_numeric(q, errors='coerce')
                        if pd.isna(q_num) or q_num <= 0:
                            continue
                    except Exception:
                        continue
                    curr_item = str(df_preenchido.at[idx, item_col]).strip().upper() if not pd.isna(df_preenchido.at[idx, item_col]) else ''
                    # Buscar item anterior não vazio
                    prev_item = ''
                    for j in range(df_preenchido.index.get_loc(idx)-1, -1, -1):
                        real_idx = df_preenchido.index[j]
                        val = df_preenchido.at[real_idx, item_col]
                        if pd.notna(val) and str(val).strip() != '':
                            prev_item = str(val).strip().upper()
                            break
                    # Buscar item posterior não vazio
                    next_item = ''
                    for j in range(df_preenchido.index.get_loc(idx)+1, len(df_preenchido.index)):
                        real_idx = df_preenchido.index[j]
                        val = df_preenchido.at[real_idx, item_col]
                        if pd.notna(val) and str(val).strip() != '':
                            next_item = str(val).strip().upper()
                            break
                    # Regra: se nenhum vizinho coincide com o item atual -> descartar
                    cond_prev = (prev_item != '' and prev_item == curr_item)
                    cond_next = (next_item != '' and next_item == curr_item)
                    if not (cond_prev or cond_next):
                        to_drop.append(idx)
                if to_drop:
                    print(f"[Retrabalho] Removendo {len(to_drop)} linhas de retrabalho isoladas (formato antigo).")
                    df_preenchido = df_preenchido.drop(index=to_drop)
                    df_preenchido = df_preenchido.reset_index(drop=True)
        # ============================================================================
        
        # Normalizar strings vazias para NaN nas colunas alvo para que ffill/bfill funcionem
        alvo_old = ['Ordem Prod', 'Descrição Item', 'Roteiro']
        for c in alvo_old:
            if c in df_preenchido.columns:
                df_preenchido[c] = df_preenchido[c].apply(lambda x: None if _eh_vazio(x) else x)

        # Ordenar o DataFrame por Centro Trabalho e data/hora
        if 'datetime_inicio' in df_preenchido.columns:
            df_preenchido = df_preenchido.sort_values(by=['Centro Trabalho', 'datetime_inicio'])
        elif 'Data Início' in df_preenchido.columns and 'Hora Início' in df_preenchido.columns:
            df_preenchido = df_preenchido.sort_values(by=['Centro Trabalho', 'Data Início', 'Hora Início'])

        # Regra extra: linhas do início do turno (06:00) com células vazias devem receber o primeiro valor válido abaixo
        # Aplica-se por Centro de Trabalho e somente para as colunas alvo
        if 'Hora Início' in df_preenchido.columns:
            col_hora = 'Hora Início'
            for centro, grupo in df_preenchido.groupby('Centro Trabalho'):
                # Filtrar possíveis registros de 06:00 (tratando diferentes tipos)
                idxs_grupo = grupo.index.tolist()
                for idx in idxs_grupo:
                    hora_val = df_preenchido.loc[idx, col_hora]
                    try:
                        hora_str = str(hora_val).strip()
                        eh_seis = False
                        if isinstance(hora_val, str):
                            eh_seis = hora_str.startswith('06:00')
                        elif pd.notna(hora_val):
                            # Pode vir como time/datetime ou número fracionário do Excel
                            if hasattr(hora_val, 'hour'):
                                eh_seis = hora_val.hour == 6 and getattr(hora_val, 'minute', 0) == 0
                            else:
                                # formato numérico (fração do dia) -> 6h = 6/24
                                try:
                                    val_float = float(hora_val)
                                    eh_seis = abs(val_float - (6/24)) < 1e-6
                                except:
                                    pass
                        if not eh_seis:
                            continue
                    except Exception:
                        continue

                    # Se já tem valores não nulos, não precisa preencher
                    if all(not _eh_vazio(df_preenchido.loc[idx, c]) for c in ['Ordem Prod', 'Descrição Item', 'Roteiro'] if c in df_preenchido.columns):
                        continue

                    # Buscar primeiro registro subsequente com valores válidos nas colunas alvo
                    for prox_idx in idxs_grupo[idxs_grupo.index(idx)+1:]:
                        if all(not _eh_vazio(df_preenchido.loc[prox_idx, c]) for c in ['Descrição Item'] if c in df_preenchido.columns):
                            # Copiar Ordem e Descrição Item se estiverem vazias
                            for c in ['Ordem Prod', 'Descrição Item']:
                                if c in df_preenchido.columns and _eh_vazio(df_preenchido.loc[idx, c]) and not _eh_vazio(df_preenchido.loc[prox_idx, c]):
                                    df_preenchido.loc[idx, c] = df_preenchido.loc[prox_idx, c]
                            # Só preencher Roteiro se o Item de origem for igual ao Item destino
                            if 'Roteiro' in df_preenchido.columns and _eh_vazio(df_preenchido.loc[idx, 'Roteiro']) and not _eh_vazio(df_preenchido.loc[prox_idx, 'Roteiro']):
                                item_dest = df_preenchido.loc[idx, 'Descrição Item'] if not _eh_vazio(df_preenchido.loc[idx, 'Descrição Item']) else None
                                item_origem = df_preenchido.loc[prox_idx, 'Descrição Item'] if not _eh_vazio(df_preenchido.loc[prox_idx, 'Descrição Item']) else None
                                if item_dest is not None and item_origem is not None and str(item_dest).strip() == str(item_origem).strip():
                                    df_preenchido.loc[idx, 'Roteiro'] = df_preenchido.loc[prox_idx, 'Roteiro']
                            break
        
        # Primeira passagem: preencher valores nulos usando o último valor válido
        # Agrupamos por Centro de Trabalho e preenchemos os valores nulos com o último valor válido
        # NOTE: não preenchermos 'Roteiro' por ffill para evitar propagar roteiros entre itens diferentes
        for col in ['Ordem Prod', 'Descrição Item']:
            df_preenchido[col] = df_preenchido.groupby('Centro Trabalho')[col].ffill()
        
        # Segunda passagem: lidar com mudanças de Centro de Trabalho e paradas específicas
        # Para cada grupo de Centro de Trabalho
        for centro, grupo in df_preenchido.groupby('Centro Trabalho'):
            # Índices do grupo atual
            indices_grupo = grupo.index.tolist()
            
            # Usar para rastrear qual será o próximo item após uma parada específica
            proximo_item = None
            proximo_ordem = None
            proximo_roteiro = None
            
            # Para cada linha no grupo
            for i in range(len(indices_grupo)):
                idx = indices_grupo[i]
                
                # Verificar se é uma parada específica
                if pd.notna(df_preenchido.loc[idx, 'Descrição Parada']):
                    parada = str(df_preenchido.loc[idx, 'Descrição Parada']).upper()
                    
                    if 'ACERTO' in parada or 'TROCA DE ITEM' in parada or 'TROCA DE ÍTEM' in parada or 'TROCA DE OS' in parada:
                        # Procurar o próximo item válido
                        for j in range(i+1, len(indices_grupo)):
                            prox_idx = indices_grupo[j]
                            if (not _eh_vazio(df_preenchido.loc[prox_idx, 'Descrição Item'])) and _eh_vazio(df_preenchido.loc[prox_idx, 'Descrição Parada']):
                                proximo_item = df_preenchido.loc[prox_idx, 'Descrição Item']
                                proximo_ordem = df_preenchido.loc[prox_idx, 'Ordem Prod']
                                proximo_roteiro = df_preenchido.loc[prox_idx, 'Roteiro']
                                break
                        
                        # Preencher todas as linhas subsequentes até o final deste Centro de Trabalho ou até outra parada específica
                        if proximo_item is not None:
                            for j in range(i, len(indices_grupo)):
                                prox_idx = indices_grupo[j]
                                parada_atual = df_preenchido.loc[prox_idx, 'Descrição Parada']
                                
                                # Verificar se chegou a outra parada específica
                                if pd.notna(parada_atual):
                                    parada_atual = str(parada_atual).upper()
                                    if 'ACERTO' in parada_atual or 'TROCA DE ITEM' in parada_atual or 'TROCA DE ÍTEM' in parada_atual or 'TROCA DE OS' in parada_atual:
                                        break
                                
                                # Preencher as informações
                                if _eh_vazio(df_preenchido.loc[prox_idx, 'Descrição Item']):
                                    df_preenchido.loc[prox_idx, 'Descrição Item'] = proximo_item
                                if _eh_vazio(df_preenchido.loc[prox_idx, 'Ordem Prod']):
                                    df_preenchido.loc[prox_idx, 'Ordem Prod'] = proximo_ordem
                                if _eh_vazio(df_preenchido.loc[prox_idx, 'Roteiro']):
                                    # Só preencher Roteiro se descrição do item coincidir
                                    try:
                                        item_atual = df_preenchido.loc[prox_idx, 'Descrição Item']
                                    except Exception:
                                        item_atual = None
                                    if item_atual is not None and str(item_atual).strip() == str(proximo_item).strip():
                                        df_preenchido.loc[prox_idx, 'Roteiro'] = proximo_roteiro
        
        # Terceira passagem: preencher valores nulos restantes usando o próximo valor válido (backward fill)
        for col in ['Ordem Prod', 'Descrição Item', 'Roteiro']:
            df_preenchido[col] = df_preenchido.groupby('Centro Trabalho')[col].bfill()
        
    elif all(col in df_preenchido.columns for col in colunas_novas):
        # Usar o formato novo (dados já processados)
        print("Detectado formato de dados processado (colunas padronizadas).")

        # ================= TRATATIVA Qtd Retrabalhada (formato novo) =================
        retrab_cols = [c for c in df_preenchido.columns if c.strip().lower() in ('qtd retrabalhada','qtd_retrabalhada')]
        if not retrab_cols:
            retrab_cols = [c for c in df_preenchido.columns if 'retrabal' in c.lower()]
        if retrab_cols:
            col_retr = retrab_cols[0]
            item_col = 'item' if 'item' in df_preenchido.columns else None
            if item_col:
                try:
                    df_preenchido = df_preenchido.sort_values(by=['maquina','datetime_inicio'])
                except Exception:
                    pass
                df_preenchido = df_preenchido.reset_index(drop=True)
                to_drop = []
                for idx in range(len(df_preenchido)):
                    try:
                        q = df_preenchido.loc[idx, col_retr]
                        q_num = pd.to_numeric(q, errors='coerce')
                        if pd.isna(q_num) or q_num <= 0:
                            continue
                    except Exception:
                        continue
                    curr_item = str(df_preenchido.loc[idx, item_col]).strip().upper() if not pd.isna(df_preenchido.loc[idx, item_col]) else ''
                    # anterior
                    prev_item = ''
                    for j in range(idx-1, -1, -1):
                        if df_preenchido.loc[j, 'maquina'] != df_preenchido.loc[idx, 'maquina']:
                            break  # mudou máquina
                        val = df_preenchido.loc[j, item_col]
                        if pd.notna(val) and str(val).strip()!='':
                            prev_item = str(val).strip().upper(); break
                    # posterior
                    next_item = ''
                    for j in range(idx+1, len(df_preenchido)):
                        if df_preenchido.loc[j, 'maquina'] != df_preenchido.loc[idx, 'maquina']:
                            break
                        val = df_preenchido.loc[j, item_col]
                        if pd.notna(val) and str(val).strip()!='':
                            next_item = str(val).strip().upper(); break
                    cond_prev = (prev_item != '' and prev_item == curr_item)
                    cond_next = (next_item != '' and next_item == curr_item)
                    if not (cond_prev or cond_next):
                        to_drop.append(idx)
                if to_drop:
                    print(f"[Retrabalho] Removendo {len(to_drop)} linhas de retrabalho isoladas (formato novo).")
                    df_preenchido = df_preenchido.drop(index=to_drop).reset_index(drop=True)
        # ==========================================================================
        
        # Normalizar strings vazias nas colunas alvo novas
        alvo_new = ['ordem_servico','item','faca']
        for c in alvo_new:
            if c in df_preenchido.columns:
                df_preenchido[c] = df_preenchido[c].apply(lambda x: None if _eh_vazio(x) else x)

        # Resetar o índice para facilitar a iteração
        df_preenchido = df_preenchido.reset_index(drop=True)
        
        # Ordenar o DataFrame por máquina e data/hora
        try:
            df_preenchido = df_preenchido.sort_values(by=['maquina', 'datetime_inicio'])
        except:
            try:
                df_preenchido = df_preenchido.sort_values(by=['maquina', 'data'])
            except:
                print("Não foi possível ordenar o DataFrame por máquina e data/hora.")
        
        # Resetar o índice novamente após a ordenação
        df_preenchido = df_preenchido.reset_index(drop=True)

        # Regra extra 06:00: para cada máquina, se o primeiro registro do dia (ou qualquer registro com hora 06:00) tiver campos vazios,
        # preencher com o primeiro registro completo subsequente da mesma máquina.
        if 'datetime_inicio' in df_preenchido.columns:
            for maq, grupo in df_preenchido.groupby('maquina'):
                grupo_indices = grupo.index.tolist()
                for idx in grupo_indices:
                    dt = grupo.loc[idx, 'datetime_inicio']
                    try:
                        eh_seis = False
                        if isinstance(dt, list):
                            # Pode ser lista de datetimes; usar o primeiro
                            if dt and isinstance(dt[0], pd.Timestamp):
                                eh_seis = dt[0].hour == 6 and dt[0].minute == 0
                        elif isinstance(dt, pd.Timestamp):
                            eh_seis = dt.hour == 6 and dt.minute == 0
                        if not eh_seis:
                            continue
                    except Exception:
                        continue

                    # Verificar se falta algo
                    faltando = False
                    for c in ['ordem_servico', 'item', 'faca']:
                        if c in df_preenchido.columns:
                            val = df_preenchido.loc[idx, c]
                            if pd.isna(val) or (isinstance(val, str) and val.strip()==""):
                                faltando = True
                                break
                    if not faltando:
                        continue

                    # Buscar primeiro registro subsequente com todos completos
                    for prox_idx in grupo_indices[grupo_indices.index(idx)+1:]:
                        completo = True
                        for c in ['ordem_servico', 'item', 'faca']:
                            if c in df_preenchido.columns:
                                valp = df_preenchido.loc[prox_idx, c]
                                if pd.isna(valp) or (isinstance(valp, str) and valp.strip()==""):
                                    completo = False
                                    break
                        if completo:
                            for c in ['ordem_servico', 'item', 'faca']:
                                if c in df_preenchido.columns:
                                    val_atual = df_preenchido.loc[idx, c]
                                    if pd.isna(val_atual) or (isinstance(val_atual, str) and val_atual.strip()==""):
                                        df_preenchido.loc[idx, c] = df_preenchido.loc[prox_idx, c]
                            break
        
        # Lista de paradas que indicam troca de item
        paradas_troca_item = ['ACERTO', 'TROCA DE ÍTEM', 'TROCA DE ITEM', 'TROCA DE ÍTEM/OS', 'SETUP']
        
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
                if df_preenchido.loc[idx, 'maquina'] == maquina and registros_completos.iloc[idx]:
                    idx_proximo = idx
                    break
                    
            # Se encontrou um próximo registro completo
            if idx_proximo is not None:
                # Obter informações do próximo registro completo
                info_item = {
                    'ordem_servico': df_preenchido.loc[idx_proximo, 'ordem_servico'],
                    'item': df_preenchido.loc[idx_proximo, 'item'],
                    'faca': df_preenchido.loc[idx_proximo, 'faca']
                }
                
                # Preencher a própria linha da parada e todas as linhas seguintes até encontrar:
                # - Outra parada de troca de item
                # - Uma mudança de máquina
                # - O próximo registro completo
                idx_atual = idx_parada
                while idx_atual <= idx_proximo:
                    row_atual = df_preenchido.loc[idx_atual]

                    # Se for uma nova máquina, parar de preencher
                    if idx_atual > idx_parada and row_atual['nova_maquina']:
                        break

                    # Preencher ordem/item quando estiverem vazios
                    if esta_vazio(row_atual['ordem_servico']):
                        df_preenchido.at[idx_atual, 'ordem_servico'] = info_item['ordem_servico']
                    if esta_vazio(row_atual['item']):
                        df_preenchido.at[idx_atual, 'item'] = info_item['item']

                    # Só preencher 'faca' (Roteiro) quando for seguro:
                    # - a célula atual estiver vazia (não correr risco de sobrescrever)
                    # - E o item atual for igual ao item de origem ou estiver vazio
                    if esta_vazio(row_atual['faca']):
                        try:
                            item_atual = row_atual.get('item', None)
                        except Exception:
                            item_atual = None
                        # Normalizar comparações de string
                        try:
                            destino_ok = False
                            if item_atual is None or (isinstance(item_atual, str) and item_atual.strip() == ''):
                                destino_ok = True
                            elif isinstance(item_atual, str) and isinstance(info_item['item'], str) and item_atual.strip() == info_item['item'].strip():
                                destino_ok = True
                        except Exception:
                            destino_ok = False

                        if destino_ok:
                            df_preenchido.at[idx_atual, 'faca'] = info_item['faca']

                    idx_atual += 1
        
        # Remover a coluna auxiliar
        df_preenchido = df_preenchido.drop(columns=['nova_maquina'])
        
    else:
        print("Não foi possível identificar o formato de dados para preenchimento.")
        return df_preenchido
    
    return df_preenchido
