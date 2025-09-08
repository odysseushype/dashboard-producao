import dash
from dash import dcc, html, Input, Output, State, ctx, callback_context, ALL
import dash_bootstrap_components as dbc
import pandas as pd
import numpy as np
import os, time, json, base64, io
from datetime import datetime, timedelta, time as dtime
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
import unicodedata, html as html_stdlib, re
import warnings

# Suprimir avisos de deprecação específicos
warnings.filterwarnings('ignore', category=DeprecationWarning, message='.*scattermapbox.*')
warnings.filterwarnings('ignore', category=FutureWarning, message='.*fillna.*')
warnings.filterwarnings('ignore', category=FutureWarning, message='.*SeriesGroupBy.fillna.*')

# ========================= CONFIGURAÇÃO VISUAL GLOBAL =========================
# Nova paleta corporativa alinhada ao CSS (azul -> indigo -> roxo + auxiliares)
COLOR_SEQ = [
    '#1d4ed8',  # azul principal
    '#6366f1',  # indigo
    '#8b5cf6',  # purple
    '#a855f7',  # purple mais claro
    '#7c3aed',  # violet
    '#9333ea',  # purple intermediário
    '#5b21b6',  # purple escuro
    '#4c1d95',  # purple muito escuro
    '#6d28d9',  # violet escuro
    '#581c87'   # purple final
]

# Paletas específicas por tipo de gráfico
PIE_COLORS = ['#1d4ed8', '#6366f1', '#8b5cf6', '#a855f7', '#7c3aed', '#9333ea', '#5b21b6', '#4c1d95']
BAR_COLORS = ['#8b5cf6', '#a855f7', '#7c3aed', '#9333ea', '#5b21b6', '#6d28d9', '#581c87', '#4c1d95']
HEATMAP_COLORS = ['#f8fafc', '#e2e8f0', '#cbd5e1', '#94a3b8', '#64748b', '#475569', '#334155', '#1e293b']

# Template Plotly corporativo global (aplica a todos os gráficos, inclusive os que não passam por style_fig)
_corporate_template = go.layout.Template(
    layout=go.Layout(
        colorway=COLOR_SEQ,
        paper_bgcolor='white',
        plot_bgcolor='white',
        font=dict(family='Inter, Arial, Helvetica, sans-serif', size=12, color='#0f172a'),
        title=dict(font=dict(family='Inter, Arial', size=16, color='#0f172a'), x=0.01, xanchor='left'),
        hoverlabel=dict(font=dict(family='Inter, Arial', size=11)),
        legend=dict(bgcolor='rgba(0,0,0,0)', borderwidth=0, font=dict(size=11)),
        margin=dict(l=50, r=20, t=58, b=90, autoexpand=True)  # autoexpand para ajuste automático
    )
)
pio.templates['corporate'] = _corporate_template
try:
    # Combina com simple_white para herdar defaults limpos
    pio.templates.default = 'corporate+simple_white'
except Exception:
    pass

# Defaults extras para express
px.defaults.color_continuous_scale = ['#f8fafc', '#e2e8f0', '#cbd5e1', '#8b5cf6', '#7c3aed', '#5b21b6']  # Heatmap em tons purple
px.defaults.template = 'corporate'

def style_fig(fig: go.Figure, y_hours: bool = False, legend_bottom: bool = True, tight: bool = False):
    """Aplica o layout corporativo padronizado.
    Args:
        fig: figura plotly
        y_hours: se True, define label Y = Horas
        legend_bottom: se True, legenda horizontal inferior
        tight: margens horizontais menores
    """
    try:
        # Margens base - aumentar margem inferior significativamente para legendas
        base_margin = dict(l=48, r=16, t=58, b=90, autoexpand=True)  # autoexpand para ajuste automático
        if tight:
            base_margin = dict(l=40, r=12, t=54, b=95, autoexpand=True)  # autoexpand para ajuste automático

        fig.update_layout(
            template='corporate',
            margin=base_margin,
            hovermode='closest',
            autosize=True  # Permite ajuste automático de tamanho
        )

        # Legenda
        if legend_bottom:
            fig.update_layout(legend=dict(
                orientation='h', 
                y=-0.45,  # Mover legenda muito mais para baixo
                x=0, 
                xanchor='left', 
                yanchor='top',
                bgcolor='rgba(255,255,255,0.8)',  # Fundo semi-transparente
                bordercolor='rgba(0,0,0,0.1)',
                borderwidth=1
            ))
        else:
            fig.update_layout(legend=dict(
                orientation='v', 
                y=0.98, 
                x=0.99, 
                xanchor='right', 
                yanchor='top',
                bgcolor='rgba(255,255,255,0.9)',  # Fundo semi-transparente
                bordercolor='rgba(0,0,0,0.1)',
                borderwidth=1
            ))

        # Eixos
        grid_color = 'rgba(0,0,0,0.06)'
        fig.update_xaxes(showgrid=False, zeroline=False, linecolor='rgba(0,0,0,0.25)', ticks='', showspikes=False)
        fig.update_yaxes(showgrid=True, gridcolor=grid_color, zeroline=False, linecolor='rgba(0,0,0,0.12)', ticks='')
        if y_hours:
            fig.update_yaxes(title='Horas')
        else:
            fig.update_yaxes(title=fig.layout.yaxis.title.text if fig.layout.yaxis.title.text else None)

        # Título - configuração padrão sem barra
        if fig.layout.title and fig.layout.title.text:
            # Garantir configuração de título consistente
            fig.update_layout(
                title=dict(
                    text=fig.layout.title.text,
                    font=dict(family='Inter, Arial', size=16, color='#0f172a'),
                    x=0.01,
                    xanchor='left',
                    y=0.95,
                    yanchor='top'
                )
            )
    except Exception:
        pass
    return fig

def style_pie(fig: go.Figure, total_label: str = None):
    """Aplica estilo donut corporativo com paleta azul-roxo."""
    try:
        fig.update_traces(
            hole=0.55,
            textinfo='percent',
            textfont_size=12,
            marker=dict(
                line=dict(color='white', width=1),
                colors=PIE_COLORS
            )
        )
        fig.update_layout(
            template='corporate',
            margin=dict(l=24, r=24, t=54, b=80, autoexpand=True),  # autoexpand para ajuste automático
            legend=dict(orientation='h', y=-0.4, x=0, font=dict(size=11))  # Legenda muito mais baixa
        )
        if total_label:
            # Mantém outras anotações se existirem
            anns = list(fig.layout.annotations) if fig.layout.annotations else []
            anns.append(dict(text=total_label, x=0.5, y=0.5, showarrow=False,
                             font=dict(size=14, color='#0f172a', family='Inter, Arial')))
            fig.update_layout(annotations=anns)
    except Exception:
        pass
    return fig

def style_bar(fig: go.Figure):
    """Aplica cores purple específicas para gráficos de barras."""
    try:
        # Aplicar cores purple aos traces de barras
        for i, trace in enumerate(fig.data):
            if hasattr(trace, 'marker'):
                color_idx = i % len(BAR_COLORS)
                trace.marker.color = BAR_COLORS[color_idx]
    except Exception:
        pass
    return fig

def style_heatmap(fig: go.Figure):
    """Aplica escala de cores purple para heatmaps."""
    try:
        fig.update_traces(
            colorscale=[[0, '#f8fafc'], [0.2, '#e2e8f0'], [0.4, '#cbd5e1'], 
                       [0.6, '#8b5cf6'], [0.8, '#7c3aed'], [1, '#5b21b6']]
        )
        # Melhorar legibilidade dos labels do eixo Y
        fig.update_yaxes(
            tickfont=dict(size=10),
            automargin=True
        )
        fig.update_xaxes(
            tickfont=dict(size=10),
            automargin=True  # Adicionado automargin para eixo X também
        )
    except Exception:
        pass
    return fig

# Flag de debug (controla verbosidade)
DEBUG = False

def debug_print(msg:str):
    if DEBUG:
        print(msg)

# CSS custom minimal theme injected into the layout for improved visuals
# CSS_TEXT removido (estilos agora em assets/custom.css)

# DataFrames globais (inicialmente None) usados como fallback para descrições
df_original_global = None
df_proc_global = None

# ========================= NORMALIZAÇÃO DE TEXTO =========================
def _limpar_texto_basico(s: str) -> str:
    """Remove caracteres não imprimíveis e normaliza espaços."""
    # Remover controles
    s = ''.join(ch for ch in s if ch.isprintable())
    # Normalizar espaços múltiplos
    s = re.sub(r'\s+', ' ', s).strip()
    return s

# ========================= ATRIBUIÇÃO DE ROTEIROS (REGRAS) =========================
def atribuir_roteiros_generico(df: pd.DataFrame, df_vel: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Atribui roteiros faltantes seguindo regras específicas por centro de trabalho.
    Regras:
      CA05: se Roteiro vazio -> RAPIDO (Qtd Aprovada <=18000) senão LENTO ( >18000 )
      CA04: Pre Vincagem -> PREVINCAGEM (vel 120000); Aplic Ink-Jet / Pré-Vincagem -> INKJET_PREVINCAGEM (2x base 30000)
      CA16: Pre Vincagem -> PREVINCAGEM (vel 100000)
      CA15: Aplic Ink-Jet / Colagem -> INKJET (vel 10000)
      CA09: operação != Colagem -> GERAL (vel 12000)
      CA01: qualquer operação com roteiro vazio -> GERAL (vel 9000)
    Retorna df e df_vel (velocidades) atualizados.
    """
    if df is None or len(df) == 0:
        return df, df_vel

    df = df.copy()
    # garantir df_vel mínimo
    df_vel = df_vel.copy() if df_vel is not None else pd.DataFrame(columns=['Conc', 'Velocidade Padrão'])
    if 'Velocidade Padrão' not in df_vel.columns:
        for c in df_vel.columns:
            if 'vel' in str(c).lower():
                df_vel = df_vel.rename(columns={c: 'Velocidade Padrão'})
                break
    if 'Conc' not in df_vel.columns:
        df_vel['Conc'] = df_vel['Conc'] if 'Conc' in df_vel.columns else pd.Series(dtype='object')

    def add_velocidade_local(conc: str, valor: float):
        """Adiciona velocidade ao df_vel (se ainda não existir) e atualiza DICT_VELOCIDADE imediatamente."""
        conc_s = str(conc).strip()
        try:
            exists = conc_s in set(df_vel['Conc'].astype(str))
        except Exception:
            exists = False
        if not exists and conc_s:
            df_vel.loc[len(df_vel)] = {'Conc': conc_s, 'Velocidade Padrão': float(valor)}
        # atualizar global imediato para uso posterior
        try:
            if 'DICT_VELOCIDADE' in globals() and conc_s:
                DICT_VELOCIDADE[conc_s] = float(valor)
        except Exception:
            pass

    # Garantir colunas relevantes existam e trabalhar com strings normalizados
    for col in ['Centro Trabalho', 'Roteiro', 'Descrição Operação', 'Qtd Aprovada', 'Descrição Item']:
        if col not in df.columns:
            df[col] = ''

    # normalizar temporariamente colunas para comparação
    centro = df['Centro Trabalho'].astype(str).fillna('').str.strip()
    roteiro_raw = df['Roteiro'].fillna('').astype(str).str.strip()
    operacao = df['Descrição Operação'].fillna('').astype(str).str.strip()
    item_desc = df['Descrição Item'].fillna('').astype(str).str.strip()

    roteiro_vazio = roteiro_raw == ''

    # CA05: por item, baseado em Qtd Aprovada
    mask_ca05 = (centro == 'CA05') & roteiro_vazio
    if mask_ca05.any():
        sub = df.loc[mask_ca05, ['Descrição Item', 'Qtd Aprovada']].copy()
        for idx, row in sub.iterrows():
            try:
                qtd = float(row['Qtd Aprovada']) if row['Qtd Aprovada'] not in [None, ''] else 0
            except Exception:
                qtd = 0
            rote = 'RAPIDO' if qtd <= 18000 else 'LENTO'
            vel_pad = 50000 if rote == 'RAPIDO' else 70000
            item = str(row['Descrição Item']).strip()
            # aplicar somente em linhas do mesmo item e centro com roteiro vazio
            cond = (centro == 'CA05') & (item_desc == item) & (df['Roteiro'].fillna('').astype(str).str.strip() == '')
            if cond.any():
                df.loc[cond, 'Roteiro'] = rote
                add_velocidade_local(f"CA05-{rote}", vel_pad)
                debug_print(f"[Atribuir Roteiros] CA05: set {rote} para item '{item}' ({cond.sum()} registros)")

    # CA04 Pre Vincagem
    mask_ca04_prev = (centro == 'CA04') & (operacao == 'Pre Vincagem') & (df['Roteiro'].fillna('').astype(str).str.strip() == '')
    if mask_ca04_prev.any():
        df.loc[mask_ca04_prev, 'Roteiro'] = 'PREVINCAGEM'
        add_velocidade_local('CA04-PREVINCAGEM', 120000)
        debug_print(f"[Atribuir Roteiros] CA04-PREVINCAGEM aplicado a {mask_ca04_prev.sum()} registros")

    # CA04 Inkjet + Pré-Vincagem
    mask_ca04_ink = (centro == 'CA04') & (operacao == 'Aplic Ink-Jet / Pré-Vincagem') & (df['Roteiro'].fillna('').astype(str).str.strip() == '')
    if mask_ca04_ink.any():
        df.loc[mask_ca04_ink, 'Roteiro'] = 'INKJET_PREVINCAGEM'
        add_velocidade_local('CA04-INKJET_PREVINCAGEM', 60000)
        debug_print(f"[Atribuir Roteiros] CA04-INKJET_PREVINCAGEM aplicado a {mask_ca04_ink.sum()} registros")

    # CA16 Pre Vincagem
    mask_ca16 = (centro == 'CA16') & (operacao == 'Pre Vincagem') & (df['Roteiro'].fillna('').astype(str).str.strip() == '')
    if mask_ca16.any():
        df.loc[mask_ca16, 'Roteiro'] = 'PREVINCAGEM'
        add_velocidade_local('CA16-PREVINCAGEM', 100000)
        debug_print(f"[Atribuir Roteiros] CA16-PREVINCAGEM aplicado a {mask_ca16.sum()} registros")

    # CA16 Aplic Ink-Jet / Pré-Vincagem (mask específica solicitada)
    mask_ca16_ink = (centro == 'CA16') & (operacao == 'Aplic Ink-Jet / Pré-Vincagem') & (df['Roteiro'].fillna('').astype(str).str.strip() == '')
    if mask_ca16_ink.any():
        df.loc[mask_ca16_ink, 'Roteiro'] = 'INKJET_PREVINCAGEM'
        add_velocidade_local('CA16-INKJET_PREVINCAGEM', 36000)
        debug_print(f"[Atribuir Roteiros] CA16-INKJET_PREVINCAGEM aplicado a {mask_ca16_ink.sum()} registros")

    # CA15 Ink-Jet / Colagem
    mask_ca15 = (centro == 'CA15') & (operacao == 'Aplic Ink-Jet / Colagem') & (df['Roteiro'].fillna('').astype(str).str.strip() == '')
    if mask_ca15.any():
        df.loc[mask_ca15, 'Roteiro'] = 'INKJET'
        add_velocidade_local('CA15-INKJET', 10000)
        debug_print(f"[Atribuir Roteiros] CA15-INKJET aplicado a {mask_ca15.sum()} registros")

    # CA09 Geral (exceto Colagem)
    mask_ca09 = (centro == 'CA09') & (operacao != 'Colagem') & (df['Roteiro'].fillna('').astype(str).str.strip() == '')
    if mask_ca09.any():
        df.loc[mask_ca09, 'Roteiro'] = 'GERAL'
        add_velocidade_local('CA09-GERAL', 12000)
        debug_print(f"[Atribuir Roteiros] CA09-GERAL aplicado a {mask_ca09.sum()} registros")

    # CA01 Geral
    mask_ca01 = (centro == 'CA01') & (df['Roteiro'].fillna('').astype(str).str.strip() == '')
    if mask_ca01.any():
        df.loc[mask_ca01, 'Roteiro'] = 'GERAL'
        add_velocidade_local('CA01-GERAL', 9000)
        debug_print(f"[Atribuir Roteiros] CA01-GERAL aplicado a {mask_ca01.sum()} registros")

    # Recriar coluna Conc para o df e garantir df_vel tem entradas únicas
    df['Conc'] = df['Centro Trabalho'].astype(str).str.strip() + '-' + df['Roteiro'].fillna('').astype(str).str.strip()

    # Garantir df_vel tem as entradas geradas (normalize e dedupe)
    try:
        df_vel['Conc'] = df_vel['Conc'].astype(str).str.strip()
        df_vel = df_vel.drop_duplicates(subset=['Conc']).reset_index(drop=True)
    except Exception:
        pass

    # CA12: multiplicar velocidade padrão por 2 para todas as entradas CA12-*
    try:
        if 'Conc' in df_vel.columns and 'Velocidade Padrão' in df_vel.columns:
            mask_ca12 = df_vel['Conc'].astype(str).str.startswith('CA12-')
            if mask_ca12.any():
                df_vel.loc[mask_ca12, 'Velocidade Padrão'] = df_vel.loc[mask_ca12, 'Velocidade Padrão'].astype(float) * 2
                # Atualizar dicionário global imediatamente
                try:
                    if 'DICT_VELOCIDADE' in globals():
                        for _, r in df_vel.loc[mask_ca12].iterrows():
                            try:
                                DICT_VELOCIDADE[str(r['Conc']).strip()] = float(r['Velocidade Padrão'])
                            except Exception:
                                continue
                except Exception:
                    pass

        # Garantir que quaisquer "Conc" usados no df para CA12 existam em df_vel/DICT_VELOCIDADE
        try:
            concs_ca12 = set(df.loc[df['Centro Trabalho'].astype(str).str.strip() == 'CA12', 'Conc'].astype(str).unique())
            existing = set(df_vel['Conc'].astype(str)) if ('Conc' in df_vel.columns) else set()
            for conc in concs_ca12:
                if conc and conc not in existing:
                    # Tentar usar um valor base do DICT_VELOCIDADE se existir, senão usar 9000 como fallback
                    base = None
                    try:
                        if 'DICT_VELOCIDADE' in globals() and conc in DICT_VELOCIDADE:
                            base = float(DICT_VELOCIDADE[conc])
                    except Exception:
                        base = None
                    if base is None:
                        base = 9000.0
                    doubled = float(base) * 2
                    try:
                        df_vel.loc[len(df_vel)] = {'Conc': conc, 'Velocidade Padrão': doubled}
                    except Exception:
                        # se df_vel for imutável em formato, apenas garantir dicionário
                        pass
                    try:
                        if 'DICT_VELOCIDADE' in globals():
                            DICT_VELOCIDADE[conc] = doubled
                    except Exception:
                        pass
        except Exception:
            pass
    except Exception:
        pass

    return df, df_vel

def normalizar_texto(valor):
    """Corrige apresentação de caracteres especiais em qualquer campo textual.
    - Decodifica entidades HTML (&amp; -> &)
    - Normaliza Unicode (NFKC)
    - Remove caracteres de controle
    - Compacta espaços
    Mantém acentos (não faz transliteração) para preservar original.
    """
    if not isinstance(valor, str):
        return valor
    try:
        # Decodificar entidades HTML
        valor = html_stdlib.unescape(valor)
        # Unicode canonical/compat normalização
        valor = unicodedata.normalize('NFKC', valor)
        # Limpeza básica
        valor = _limpar_texto_basico(valor)
        # Ajustar espaços antes de pontuação final
        valor = re.sub(r'\s+([.,;:])', r'\1', valor)
        return valor
    except Exception:
        return valor

def normalizar_dataframe_texto(df: pd.DataFrame) -> pd.DataFrame:
    """Aplica normalização de texto a todas as colunas object relevantes."""
    if df is None or df.empty:
        return df
    df = df.copy()
    # Colunas alvo específicas primeiro (evita normalizar números serializados em string acidentalmente)
    col_prioritarias = [c for c in df.columns if any(tok in c.lower() for tok in ['descr', 'parada', 'item', 'roteiro', 'ordem', 'faca'])]
    # Demais object
    outras = [c for c in df.select_dtypes(include=['object']).columns if c not in col_prioritarias]
    for col in col_prioritarias + outras:
        try:
            df[col] = df[col].apply(normalizar_texto)
        except Exception:
            continue
    return df

# Tentativa de usar implementação externa mais robusta, se disponível
try:
    from preencher_dados_faltantes import preencher_dados_faltantes as _preencher_dados_faltantes_externo  # type: ignore
    EXTERNAL_FILL_FUNC = True
except Exception:
    _preencher_dados_faltantes_externo = None
    EXTERNAL_FILL_FUNC = False

# Preenchimento de dados faltantes restrito por máquina e dia produtivo (fallback interno simplificado)
def _preencher_dados_faltantes_interno(df: pd.DataFrame) -> pd.DataFrame:
    """Fallback simples: forward-fill por máquina e dia produtivo para colunas chave."""
    if df is None or df.empty:
        return df
    df = df.copy()
    col_maquina = None
    for c in ['maquina','Centro Trabalho','Centro_Trabalho']:
        if c in df.columns:
            col_maquina = c
            break
    if 'dia_produtivo' not in df.columns:
        if 'data' in df.columns:
            df['dia_produtivo'] = df['data']
        else:
            df['dia_produtivo'] = datetime.now().strftime('%Y-%m-%d')
    cols_fill = [c for c in ['ordem_servico','item','faca','Ordem Prod','Descrição Item','Roteiro'] if c in df.columns]
    if not col_maquina or not cols_fill:
        for c in cols_fill:
            df[c] = df[c].ffill()
        return df
    df.sort_values([col_maquina,'dia_produtivo'] + ([ 'datetime_inicio'] if 'datetime_inicio' in df.columns else []), inplace=True)
    for (_, _), idxs in df.groupby([col_maquina,'dia_produtivo']).groups.items():
        bloco = df.loc[list(idxs), cols_fill].ffill()
        df.loc[list(idxs), cols_fill] = bloco
    return df

def calcular_turno(dt: datetime):
    if not isinstance(dt, datetime):
        try:
            dt = pd.to_datetime(dt)
        except Exception:
            return '1'
    h = dt.hour + dt.minute/60
    if 6 <= h < 14 + 20/60:
        return '1'
    if 14 + 20/60 <= h < 22 + 40/60:
        return '2'
    return '3'

def format_time(value):
    try:
        v = float(value)
        if v <= 0 or pd.isna(v):
            return '0.0h'
        return f'{v:.1f}h'
    except Exception:
        return '0.0h'

# Alias público
if _preencher_dados_faltantes_externo:
    def preencher_dados_faltantes(df: pd.DataFrame) -> pd.DataFrame:  # type: ignore
        try:
            return _preencher_dados_faltantes_externo(df)
        except Exception:
            return _preencher_dados_faltantes_interno(df)
else:
    preencher_dados_faltantes = _preencher_dados_faltantes_interno
def calcular_duracao_turno(turno: str, weekday: int):
    # weekday: 0=segunda ... 5=sabado
    if turno == '1':
        return (14 + 20/60) - 6  # 8h20 = 8.333...
    if turno == '2':
        if weekday == 5:  # sábado
            # 14:20 a 22:13
            return (22 + 13/60) - (14 + 20/60)
        return (22 + 40/60) - (14 + 20/60)
    if turno == '3':
        if weekday == 5:  # sábado 22:13 -> 06:00 next day
            return (24 - (22 + 13/60)) + 6
        return (24 - (22 + 40/60)) + 6
    return 0

def get_turno_end(dt: datetime) -> datetime:
    """Retorna o datetime de término do turno ao qual 'dt' pertence.
    Turnos:
      1: 06:00–14:20
      2: 14:20–22:40 (22:13 sábado)
      3: 22:40–06:00 (dia seguinte)
    """
    if not isinstance(dt, datetime):
        raise ValueError("dt inválido para get_turno_end")
    d = dt.date()
    wd = dt.weekday()
    t = dt.time()
    t0600 = time(6,0)
    t1420 = time(14,20)
    t2240 = time(22,40)
    t2213 = time(22,13)
    fim_t2 = t2213 if wd == 5 else t2240
    if t >= t0600 and t < t1420:
        return datetime.combine(d, t1420)
    if t >= t1420 and t < fim_t2:
        return datetime.combine(d, fim_t2)
    return datetime.combine(d + timedelta(days=1), t0600)

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
        return "Ignorar"
    
    # Normaliza o código da máquina (remove espaços, converte para maiúsculo)
    maquina = maquina.strip().upper()
    
    # Casos especiais a ignorar - valores que não são máquinas reais
    valores_ignorar = [
        "REPORTE", "ALTERA EFICIÊNCIA", "ALTERA EFICIENCIA", 
        "DESTAQUE COLAGEM", "COL MANUAL", "DESTAQUE_COLAGEM", "COL_MANUAL",
        "PARADA", "SETUP", "MANUTENCAO", "MANUTENÇÃO", "TROCA", 
        "ACERTO", "LIMPEZA", "INTERVALO", "ALMOCO", "ALMOÇO",
        "", " ", "NAN", "NULL", "NONE"
    ]
    
    # Verificar se é um valor a ser ignorado
    if maquina in valores_ignorar or maquina.replace(" ", "_") in valores_ignorar:
        return "Ignorar"
    
    # Classificar por tipo
    if maquina.startswith("CA"):
        try:
            # Extrair o número da máquina
            num = int(maquina[2:])
            if 1 <= num <= 16:
                return "Planas"
            elif 17 <= num <= 24:
                return "Desativada"  # Ignorar
        except (ValueError, TypeError):
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
        except (ValueError, TypeError):
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
        except (ValueError, TypeError):
            pass
    elif maquina.startswith("JA"):
        try:
            # Extrair o número da máquina
            num = int(maquina[2:])
            if num == 1:
                return "Janela"
        except (ValueError, TypeError):
            pass
    
    # Se chegou até aqui e não foi classificado, verificar se parece com uma máquina válida
    # Se contém apenas letras e números e tem pelo menos 3 caracteres, pode ser uma máquina
    if len(maquina) >= 3 and maquina.replace(" ", "").replace("-", "").isalnum():
        return "Outros"
    else:
        return "Ignorar"

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

# Carregar dados de velocidade
def carregar_dados_velocidade():
    try:
        import traceback
        
        # Criar dados de velocidade de exemplo se o arquivo não existir
        dados_exemplo = {
            "CA01-A01": 1000,
            "CA01-B01": 1500,
            "CA02-A01": 1200,
            "CA02-B02": 1800,
            "GR01-G01": 800,
            "GR01-G02": 950,
            "MC01-M01": 600,
            "MC02-M01": 750,
            "SP01-S01": 2000,
            "CH01-C01": 500
        }
        
        # Primeiro tenta carregar do arquivo
        caminho_velocidade = os.path.join('static', 'Velocidade.xlsx')
        dict_velocidade = {}
        
        if os.path.exists(caminho_velocidade):
            print(f"Carregando dados de velocidade de {caminho_velocidade}...")
            try:
                # Carregar arquivo sem especificar colunas primeiro para inspecionar
                # Usar engine='openpyxl' para melhor compatibilidade com diferentes formatos de Excel
                df_velocidade = pd.read_excel(caminho_velocidade, engine='openpyxl')
                num_colunas = len(df_velocidade.columns)
                print(f"O arquivo de velocidades tem {num_colunas} colunas: {df_velocidade.columns.tolist()}")
                
                # Mapear colunas por posição e nome para máxima flexibilidade
                col_maquina = None  # Coluna da máquina (Conc)
                col_faca = None     # Coluna da faca (FC)
                col_velocidade = None  # Coluna da velocidade padrão
                
                # Identificar colunas pelo nome
                for i, col in enumerate(df_velocidade.columns):
                    col_name = str(col).lower().strip()
                    print(f"Verificando coluna: '{col}' (nome normalizado: '{col_name}')")
                    
                    # Verificar nome exato para 'Conc'
                    if col_name == 'conc' or col == 'Conc':
                        col_maquina = col
                        print(f"Coluna de máquina encontrada: {col}")
                    # Verificação para espaço em branco
                    elif col_name == ' ' or col == ' ':
                        print(f"Coluna em branco encontrada: {col}")
                    # Verificação para FC
                    elif col_name == 'fc' or col == 'FC':
                        col_faca = col
                        print(f"Coluna de faca encontrada: {col}")
                    # Verificação para Descrição Item
                    elif col_name == 'descrição item' or col == 'Descrição Item':
                        print(f"Coluna de descrição encontrada: {col}")
                    # Verificação para velocidade padrão
                    elif col_name == 'vel padrão/ideal' or col == 'Vel Padrão/Ideal' or ('vel' in col_name and ('pad' in col_name or 'ideal' in col_name)):
                        col_velocidade = col
                        print(f"Coluna de velocidade encontrada: {col}")
                
                # Verificação mais avançada de colunas baseada nos nomes e posições
                # Se não encontrou por nome, tentar por posição (formato específico com 5 colunas)
                if not col_maquina or not col_velocidade:
                    print("Tentando identificar colunas por posição...")
                    
                    # Verificar se existe um padrão típico de arquivo de velocidades
                    if num_colunas >= 3:
                        # Examinar os valores nas primeiras linhas para ajudar a identificar
                        primeiras_linhas = df_velocidade.head(5)
                        print("Primeiras linhas do arquivo:")
                        print(primeiras_linhas)
                        
                        # Tentar identificar a coluna de máquina (tipicamente contém códigos como CA01, CA02, etc.)
                        if col_maquina is None:
                            for i, col in enumerate(df_velocidade.columns):
                                # Verificar os primeiros valores da coluna
                                valores = df_velocidade[col].head(10).astype(str).str.strip()
                                # Verificar se algum valor parece um código de máquina (CA, GR, etc)
                                if valores.str.contains('CA|GR|MC|SP|CH').any():
                                    col_maquina = col
                                    print(f"Coluna de máquina detectada pela análise de valores: {col}")
                                    break
                            
                            # Se ainda não encontrou, usar a primeira coluna
                            if col_maquina is None and num_colunas >= 1:
                                col_maquina = df_velocidade.columns[0]
                                print(f"Coluna de máquina definida por posição: {col_maquina}")
                        
                        # Tentar identificar a coluna de velocidade (valores numéricos altos)
                        if col_velocidade is None:
                            for i, col in enumerate(df_velocidade.columns):
                                try:
                                    # Converter para numérico e verificar se há valores altos
                                    valores = pd.to_numeric(df_velocidade[col], errors='coerce')
                                    if valores.median() > 1000:  # Velocidades típicas estão na faixa de milhares
                                        col_velocidade = col
                                        print(f"Coluna de velocidade detectada pela análise de valores: {col}")
                                        break
                                except (ValueError, TypeError, KeyError):
                                    continue
                            
                            # Se ainda não encontrou e temos pelo menos 5 colunas, usar a quinta
                            if col_velocidade is None and num_colunas >= 5:
                                col_velocidade = df_velocidade.columns[4]
                                print(f"Coluna de velocidade definida por posição: {col_velocidade}")
                            # Se temos menos colunas, usar a última
                            elif col_velocidade is None and num_colunas >= 2:
                                col_velocidade = df_velocidade.columns[-1]
                                print(f"Coluna de velocidade definida como última coluna: {col_velocidade}")
                        
                        # Para a coluna de faca, podemos tentar a terceira coluna em um arquivo de 5 colunas
                        if col_faca is None and num_colunas >= 3:
                            # Tentar a terceira coluna para a faca
                            col_faca = df_velocidade.columns[2]
                            print(f"Coluna de faca definida por posição: {col_faca}")
                
                # Exibir as colunas que serão usadas
                print(f"Usando colunas: Máquina={col_maquina}, Faca={col_faca}, Velocidade={col_velocidade}")
                
                # Verificar se encontrou as colunas necessárias
                if col_maquina is None or col_velocidade is None:
                    print("ERRO: Não foi possível identificar colunas obrigatórias. Usando dados de exemplo.")
                    return dados_exemplo
                
                # Verificar se temos todas as colunas necessárias
                if col_maquina is None or col_velocidade is None:
                    print("ERRO: Não foi possível identificar as colunas necessárias no arquivo.")
                    return dados_exemplo
                
                # Limpar os dados: remover linhas de cabeçalho, nulas, etc.
                df_velocidade = df_velocidade.dropna(subset=[col_maquina, col_velocidade], how='all')
                
                # Converter valores para string e limpar espaços
                df_velocidade[col_maquina] = df_velocidade[col_maquina].astype(str).str.strip()
                df_velocidade[col_velocidade] = pd.to_numeric(df_velocidade[col_velocidade], errors='coerce')
                
                # Filtrar apenas linhas válidas (máquina não vazia e velocidade numérica)
                df_velocidade = df_velocidade[(df_velocidade[col_maquina] != '') & pd.notna(df_velocidade[col_velocidade])]
                
                # Filtrar cabeçalhos que possam ter sido incluídos
                df_velocidade = df_velocidade[~df_velocidade[col_maquina].str.lower().str.contains('conc')]
                
                if col_faca:
                    df_velocidade[col_faca] = df_velocidade[col_faca].astype(str).str.strip()
                    
                # Mostrar quantas linhas válidas temos
                debug_print(f"Processando {len(df_velocidade)} linhas válidas de dados de velocidade")
                
                # Processar dados linha a linha com limite de segurança
                max_rows = min(1000, len(df_velocidade))  # Limitar processamento a 1000 linhas para segurança
                for idx, row in df_velocidade.head(max_rows).iterrows():
                    try:
                        # Obter máquina (Conc)
                        maquina = row[col_maquina]
                        
                        # Obter velocidade padrão
                        velocidade = row[col_velocidade]
                        
                        # Verificar se a velocidade é válida e não muito pequena
                        if pd.isna(velocidade) or (isinstance(velocidade, (int, float)) and velocidade < 0.01):
                            continue
                        
                        # Converter para número
                        if isinstance(velocidade, str):
                            velocidade = velocidade.replace(',', '.').strip()
                            try:
                                velocidade = float(velocidade)
                            except (ValueError, TypeError):
                                print(f"Aviso: Não foi possível converter velocidade na linha {idx}: '{velocidade}'")
                                continue
                        
                        # Criar chave principal apenas com máquina
                        chave_maquina = str(maquina).strip()
                        if not chave_maquina:  # Ignorar se a máquina está vazia após strip()
                            continue
                            
                        # Adicionar ao dicionário
                        dict_velocidade[chave_maquina] = float(velocidade)
                        debug_print(f"Adicionado: {chave_maquina} = {velocidade}")
                        
                        # Se temos faca, criar chave combinada máquina-faca
                        if col_faca is not None:
                            faca = row[col_faca]
                            if pd.notna(faca) and str(faca).strip() != '':
                                faca_str = str(faca).strip()
                                chave_combinada = f"{chave_maquina}-{faca_str}"
                                dict_velocidade[chave_combinada] = float(velocidade)
                                debug_print(f"Adicionado com faca: {chave_combinada} = {velocidade}")
                                
                                # Adicionar também versão sem espaços para compatibilidade
                                chave_sem_espacos = chave_combinada.replace(" ", "")
                                if chave_sem_espacos != chave_combinada:
                                    dict_velocidade[chave_sem_espacos] = float(velocidade)
                                    debug_print(f"Adicionado sem espaços: {chave_sem_espacos} = {velocidade}")
                        
                    except Exception as e:
                        debug_print(f"Erro ao processar linha {idx}: {e}")
                
                # Mostrar estatísticas finais do dicionário carregado
                # Resumo compacto sem spam
                total_vel = len(dict_velocidade)
                if total_vel:
                    valores = list(dict_velocidade.values())
                    min_vel = min(valores); max_vel = max(valores); med_vel = sum(valores)/total_vel
                    print(f"Velocidades carregadas: {total_vel} (min={min_vel:.1f}, max={max_vel:.1f}, média={med_vel:.1f})")
                
            except Exception as e:
                print(f"Erro ao ler arquivo: {e}")
                print("Usando dados de velocidade de exemplo.")
                dict_velocidade = dados_exemplo
        else:
            debug_print(f"Arquivo de velocidades não encontrado em {caminho_velocidade}")
            debug_print("Usando dados de velocidade de exemplo.")
            dict_velocidade = dados_exemplo
        
        # Se não conseguiu carregar nenhum dado, use os exemplos
        if not dict_velocidade:
            debug_print("Nenhum dado de velocidade carregado. Usando dados de exemplo.")
            dict_velocidade = dados_exemplo
            
        return dict_velocidade
    except KeyboardInterrupt:
        debug_print("Carregamento de dados interrompido pelo usuário.")
        if 'dict_velocidade' in locals() and dict_velocidade:
            debug_print(f"Retornando {len(dict_velocidade)} velocidades carregadas até o momento.")
            return dict_velocidade
        else:
            debug_print("Nenhum dado carregado. Retornando dados de exemplo.")
            return dados_exemplo
    except Exception as e:
        debug_print(f"Erro ao carregar dados de velocidade: {e}")
        debug_print("Traceback suprimido em modo não-DEBUG")
        if 'dict_velocidade' in locals() and dict_velocidade:
            debug_print(f"Retornando {len(dict_velocidade)} velocidades carregadas apesar do erro.")
            return dict_velocidade
        else:
            return dados_exemplo

# Carregar dicionário de velocidades
try:
    DICT_VELOCIDADE = carregar_dados_velocidade()
    print(f"Dicionário de velocidades carregado com {len(DICT_VELOCIDADE)} itens")
except Exception as e:
    print(f"Erro ao carregar dicionário de velocidades: {e}")
    # Criar dados de exemplo se houver erro
    DICT_VELOCIDADE = {
        "CA01-A01": 1000, "CA01-B01": 1500, "CA02-A01": 1200,
        "GR01-G01": 800, "MC01-M01": 600, "SP01-S01": 2000
    }
    print("Usando dicionário de velocidades de exemplo")

# Inicializar o app Dash
app = dash.Dash(__name__, external_stylesheets=[
    dbc.themes.BOOTSTRAP,
    "https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css"
])
app.title = "Relatório de Produção"

 # Define a função para processar arquivos Excel (corrigida)
def parse_contents(contents, filename):
    content_type, content_string = contents.split(',')
    decoded = base64.b64decode(content_string)
    try:
        # Ler Excel
        df = pd.read_excel(io.BytesIO(decoded))
        df = normalizar_dataframe_texto(df)

        # Backup original preenchendo dados faltantes
        df_original = df.copy()
        df_original_preenchido = preencher_dados_faltantes(df_original)
        global df_original_global
        df_original_global = df_original_preenchido

        # Converter colunas numéricas
        df = converter_texto_para_numero(df)

        # Persistir upload
        with open(os.path.join('uploads', filename), 'wb') as f:
            f.write(decoded)

        # Processar e normalizar resultado final
        df_processado = processar_dados_producao(df)
        df_processado = normalizar_dataframe_texto(df_processado)
        global df_proc_global
        df_proc_global = df_processado

        return df_processado, f"Arquivo '{filename}' carregado com sucesso! Dados faltantes foram preenchidos automaticamente.", True
    except Exception as e:
        print(e)
        return None, f"Erro ao processar o arquivo: {e}", False

# A função preencher_dados_faltantes é importada de preenche_dados.py
    
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
            except (ValueError, TypeError):
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
    df_proc = normalizar_dataframe_texto(df.copy())

    # Aplicar regras de atribuição de roteiros ANTES do preenchimento genérico
    try:
        # df_velocidades inicial pode não existir ainda; criar estrutura vazia
        df_dummy_vel = pd.DataFrame(columns=['Conc','Velocidade Padrão'])
        df_proc, df_dummy_vel = atribuir_roteiros_generico(df_proc, df_dummy_vel)
    except Exception as e:
        print(f"Falha ao atribuir roteiros genéricos: {e}")
    
    # Preencher dados faltantes nas linhas de parada
    print("Preenchendo dados faltantes nas linhas de parada...")
    df_proc = preencher_dados_faltantes(df_proc)
    
    # Converter a data e hora para datetime
    try:
        # Tentar formatar a data assumindo formato DD/MM/YYYY
        df_proc['data_inicio_dt'] = pd.to_datetime(df_proc['Data Início'], format='%d/%m/%Y')
    except (ValueError, TypeError, KeyError):
        try:
            # Caso falhe, tentar formato padrão
            df_proc['data_inicio_dt'] = pd.to_datetime(df_proc['Data Início'])
        except (ValueError, TypeError, KeyError):
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
    
    # Calcular turno baseado na hora
    df_proc['turno'] = df_proc['datetime_inicio'].apply(calcular_turno)
    
    # Calcular duração do turno
    df_proc['duracao_turno'] = df_proc.apply(
        lambda row: calcular_duracao_turno(row['turno'], row['datetime_inicio'].weekday()) 
        if pd.notna(row['datetime_inicio']) and pd.notna(row['turno']) else 0.0, 
        axis=1
    )

    # Garantir coluna dia_produtivo antes de usar
    if 'dia_produtivo' not in df_proc.columns:
        try:
            # Regra: se hora < 06:00 pertence ao dia anterior
            df_proc['dia_produtivo'] = df_proc['datetime_inicio'].apply(
                lambda x: (x - timedelta(days=1)).strftime('%Y-%m-%d') if pd.notna(x) and x.hour < 6 else x.strftime('%Y-%m-%d') if pd.notna(x) else None
            )
        except Exception:
            df_proc['dia_produtivo'] = datetime.now().strftime('%Y-%m-%d')
    
    # Usar o dia produtivo como a data para agrupamento
    df_proc['data'] = df_proc['dia_produtivo']
    
    # Garantir que as colunas numéricas sejam realmente numéricas
    colunas_numericas = ['Qtd Aprovada', 'Parada Real Útil']
    for col in colunas_numericas:
        try:
            df_proc[col] = pd.to_numeric(df_proc[col], errors='coerce').fillna(0)
        except (ValueError, TypeError):
            print(f"Erro ao converter coluna {col} para numérico. Usando valores como estão.")
    
    # Criar colunas padronizadas para o dashboard conforme mapeamento definido
    # Verificar se a coluna Centro Trabalho existe antes de atribuir à coluna maquina
    if 'Centro Trabalho' in df_proc.columns:
        df_proc['maquina'] = df_proc['Centro Trabalho']
    else:
        print("AVISO: Coluna 'Centro Trabalho' não encontrada. Criando coluna 'maquina' vazia.")
        df_proc['maquina'] = "Desconhecido"
    
    # Classificar tipos e manter SOMENTE Planas conforme requisito atual
    df_proc['tipo_maquina'] = df_proc['maquina'].apply(classificar_tipo_maquina)
    df_proc = df_proc[df_proc['tipo_maquina'] == 'Planas'].copy()
    
    if len(df_proc) == 0:
        print("AVISO: Nenhuma máquina válida encontrada após filtragem.")
        # Criar um DataFrame mínimo para evitar erros
        df_proc = pd.DataFrame({
            'data': [datetime.now().strftime('%Y-%m-%d')],
            'maquina': ['CA01'],
            'tipo_maquina': ['Planas'],
            'producao': [0],
            'eficiencia': [1.0],
            'tempo_producao': [0.0]
        })
    
    df_proc['ordem_servico'] = df_proc['Ordem Prod']
    df_proc['item'] = df_proc['Descrição Item']
    df_proc['faca'] = df_proc['Roteiro']
    df_proc['producao'] = df_proc['Qtd Aprovada']
    df_proc['parada'] = df_proc['Descrição Parada']
    df_proc['tempo_parada'] = df_proc['Parada Real Útil']
    df_proc['tempo_producao'] = 0.0  # Inicializa a coluna
    
    # ================= NOVO CÁLCULO DE TEMPOS =================
    # Garantir datetime_fim
    if 'datetime_fim' not in df_proc.columns:
        if 'Data Fim' in df_proc.columns and 'Hora Fim' in df_proc.columns:
            try:
                df_proc['datetime_fim'] = pd.to_datetime(df_proc['Data Fim'].astype(str).str.strip() + ' ' + df_proc['Hora Fim'].astype(str).str.strip(), errors='coerce', dayfirst=True)
            except Exception:
                df_proc['datetime_fim'] = pd.NaT
        else:
            df_proc['datetime_fim'] = pd.NaT
    # Preencher datetime_fim ausente com próximo inicio da mesma máquina
    df_proc = df_proc.sort_values(['maquina','datetime_inicio'])
    for maq, gmaq in df_proc.groupby('maquina', sort=False):
        idxs = gmaq.index.tolist()
        for i, idx in enumerate(idxs):
            if pd.isna(df_proc.at[idx,'datetime_fim']):
                if i < len(idxs)-1:
                    prox_idx = idxs[i+1]
                    prox_ini = df_proc.at[prox_idx,'datetime_inicio']
                    if pd.notna(prox_ini) and pd.notna(df_proc.at[idx,'datetime_inicio']) and prox_ini > df_proc.at[idx,'datetime_inicio']:
                        df_proc.at[idx,'datetime_fim'] = prox_ini
                # Se ainda NaT, acrescentar 1 minuto para não zerar
                if pd.isna(df_proc.at[idx,'datetime_fim']) and pd.notna(df_proc.at[idx,'datetime_inicio']):
                    inicio_val = df_proc.at[idx,'datetime_inicio']
                    if isinstance(inicio_val, str):
                        inicio_val = pd.to_datetime(inicio_val, errors='coerce')
                    if pd.notna(inicio_val):
                        df_proc.at[idx,'datetime_fim'] = inicio_val + pd.Timedelta(minutes=1)
    # Calcular duração bruta por linha
    df_proc['duracao_h'] = (df_proc['datetime_fim'] - df_proc['datetime_inicio']).dt.total_seconds()/3600.0
    df_proc.loc[df_proc['duracao_h'] < 0, 'duracao_h'] = 0
    # Tempo de parada já está em 'tempo_parada'; garantir numérico
    df_proc['tempo_parada'] = pd.to_numeric(df_proc['tempo_parada'], errors='coerce').fillna(0.0)
    # Tempo efetivo linha = duração - parada (>=0)
    df_proc['tempo_producao_linha'] = (df_proc['duracao_h'] - df_proc['tempo_parada']).clip(lower=0)
    # Agrupar por maquina,item,turno (se existir) para obter tempo do item na máquina
    grp_cols = ['maquina','item'] + (['turno'] if 'turno' in df_proc.columns else [])
    tempos_agg = df_proc.groupby(grp_cols)['tempo_producao_linha'].sum().reset_index().rename(columns={'tempo_producao_linha':'_tempo_item_calc'})
    df_proc = df_proc.merge(tempos_agg, on=grp_cols, how='left')
    df_proc['_tempo_item_calc'] = df_proc['_tempo_item_calc'].fillna(0)
    df_proc['tempo_item_maquina'] = df_proc['_tempo_item_calc']
    df_proc['tempo_producao'] = df_proc['_tempo_item_calc']
    # ===========================================================

    # Preencher dados faltantes nas linhas de parada
    df_proc = preencher_dados_faltantes(df_proc)
    
    # Reforçar filtro somente Planas
    df_proc['tipo_maquina'] = df_proc['maquina'].apply(classificar_tipo_maquina)
    df_proc = df_proc[df_proc['tipo_maquina'] == 'Planas'].copy()
    
    # Armazenar hora inicio para filtragem posterior
    df_proc['hora_inicio'] = df_proc['datetime_inicio'].dt.hour
    
    # Criar identificador de velocidade (Máquina-Roteiro)
    df_proc['velocidade_id'] = df_proc['maquina'] + '-' + df_proc['faca'].astype(str)
    
    # Criar um dicionário para armazenar agregações por máquina e faca
    maquina_faca_agg = {}
    tipo_maquina_agg = {}
    
    # Filtrar máquinas que não devem ser ignoradas
    df_proc_filtrado = df_proc.copy()  # já somente Planas
    
    # Agrupar dados por máquina para análise
    maquinas = df_proc_filtrado['maquina'].unique()
    
    # Filtrar máquinas válidas (códigos de máquina reais)
    maquinas_validas = []
    for maquina in maquinas:
        if isinstance(maquina, str) and len(maquina.strip()) > 0:
            tipo = classificar_tipo_maquina(maquina)
            if tipo not in ['Ignorar', 'Desativada']:
                maquinas_validas.append(maquina)
    
    # Para cada máquina válida, criar uma coluna com a quantidade produzida apenas dessa máquina
    for maquina in maquinas_validas:
        if isinstance(maquina, str):  # Garantir que a máquina seja uma string
            col_name = maquina.replace(' ', '_')  # Formatar nome da coluna
            df_proc[col_name] = 0  # Inicializar coluna com zeros
            
            # Atribuir valor apenas para registros desta máquina específica
            mask = df_proc['maquina'] == maquina
            df_proc.loc[mask, col_name] = pd.to_numeric(df_proc.loc[mask, 'producao'], errors='coerce').fillna(0)
            
            # Adicionar ao dicionário de agregação
            maquina_faca_agg[col_name] = 'sum'
    
    # Agrupar por tipo de máquina
    tipos_maquina = ['Planas']
    for tipo in tipos_maquina:
        if tipo not in ['Ignorar', 'Desativada']:
            tipo_col = f"tipo_{tipo.replace(' ', '_')}"
            df_proc[tipo_col] = 0  # Inicializar coluna com zeros
            
            # Atribuir valor apenas para registros deste tipo específico
            mask = df_proc['tipo_maquina'] == tipo
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
    
    # Calcular eficiência com base nas velocidades padrão e produção real
    print("Calculando eficiência com base nas velocidades padrão...")
    df_proc['eficiencia'] = 1.0  # Valor padrão
    
    # Verificar se temos dados de velocidades carregados
    if DICT_VELOCIDADE:
        # Imprimir alguns exemplos do dicionário somente se DEBUG para reduzir I/O
        if DEBUG:
            print("Exemplos do dicionário de velocidades:")
            count = 0
            for k, v in DICT_VELOCIDADE.items():
                print(f"  {k}: {v}")
                count += 1
                if count >= 10:
                    print(f"  ... e mais {len(DICT_VELOCIDADE) - 10} valores")
                    break
        
        # Para cada linha no DataFrame
        for idx, row in df_proc.iterrows():
            # Verificar se temos a informação de máquina e faca
            if pd.notna(row.get('maquina')):
                # Verificar formato da faca (garantir string para evitar erros)
                faca = str(row.get('faca', '')) if pd.notna(row.get('faca', '')) else ''
                maquina = str(row['maquina'])
                
                # Criar várias opções de chave para busca no dicionário
                chave_principal = f"{maquina}-{faca}" if faca else maquina
                chave_sem_espacos = chave_principal.replace(" ", "")
                
                # Buscar velocidade padrão no dicionário, testando diferentes formatos
                vel_padrao = 0
                chave_usada = None
                
                # Tentar todas as combinações possíveis
                opcoes_chaves = [
                    (chave_principal, "chave principal"),
                    (chave_sem_espacos, "chave sem espaços"),
                    (maquina, "apenas máquina"),
                    (maquina.replace(" ", ""), "máquina sem espaços")
                ]
                
                for chave, descricao in opcoes_chaves:
                    if chave in DICT_VELOCIDADE:
                        vel_padrao = DICT_VELOCIDADE[chave]
                        chave_usada = f"{chave} ({descricao})"
                        break
                
                # Log para debug
                if DEBUG and idx < 20:  # Mostrar para as primeiras 20 linhas somente em modo debug
                    if vel_padrao > 0:
                        print(f"Linha {idx}: Velocidade encontrada {vel_padrao} usando {chave_usada}")
                    else:
                        print(f"Linha {idx}: Velocidade não encontrada para {chave_principal}")
                
                # Calcular a eficiência somente se encontrou velocidade padrão
                if vel_padrao > 0:
                    # Calcular tempo total de produção (em horas)
                    tempo_producao = float(row.get('tempo_producao', 0))
                    
                    # Calcular produção real por hora
                    if tempo_producao > 0:
                        producao_real = float(row.get('producao', 0))
                        producao_por_hora = producao_real / tempo_producao
                        
                        # Calcular eficiência (produção real / produção esperada)
                        eficiencia = producao_por_hora / vel_padrao
                        
                        # Limitar a eficiência entre 0 e 2 (0% a 200%) para visualização
                        eficiencia_limitada = min(max(eficiencia, 0), 2.0)
                        
                        # Salvar os valores no DataFrame
                        df_proc.loc[idx, 'eficiencia'] = eficiencia_limitada
                        df_proc.loc[idx, 'velocidade_padrao'] = vel_padrao
                        df_proc.loc[idx, 'velocidade_real'] = producao_por_hora
                        
                        # Log detalhado para debug
                        if DEBUG and idx < 5:
                            print(f"Linha {idx}: Item {row.get('item', 'N/A')} - Chave: {chave_usada}")
                            print(f"  Vel. Padrão: {vel_padrao}, Produção: {producao_real}")
                            print(f"  Tempo: {tempo_producao:.2f}h, Vel. Real: {producao_por_hora:.0f}/h")
                            print(f"  Eficiência: {eficiencia:.2f} ({eficiencia*100:.1f}%)")
                    else:
                        # Se não houver tempo de produção, definir valores padrão
                        df_proc.loc[idx, 'eficiencia'] = 1.0
                        df_proc.loc[idx, 'velocidade_padrao'] = vel_padrao
                        df_proc.loc[idx, 'velocidade_real'] = 0
                else:
                    # Se não encontrou a velocidade padrão, usar valores padrão
                    df_proc.loc[idx, 'eficiencia'] = 1.0
                    df_proc.loc[idx, 'velocidade_padrao'] = 0
                    df_proc.loc[idx, 'velocidade_real'] = 0
    else:
        print("Dados de velocidade não disponíveis para cálculo de eficiência.")
    
    # Incluir hora_inicio, tipo_maquina e tempo_producao no dicionário de agregação para filtragem posterior
    maquina_faca_agg['hora_inicio'] = list  # Lista para preservar todos os valores de hora
    maquina_faca_agg['datetime_inicio'] = list  # Lista para preservar todos os valores de datetime
    maquina_faca_agg['tipo_maquina'] = list  # Lista para preservar os tipos de máquina
    maquina_faca_agg['tempo_producao'] = 'sum'  # Somar tempo de produção
    
    # Combinar as agregações de máquina e tipo de máquina
    todas_agregacoes = {**maquina_faca_agg, **tipo_maquina_agg}
    
    # O DataFrame processado (df_proc) agora contém todas as informações detalhadas
    # e não será mais agrupado aqui. O agrupamento será feito no callback do dashboard.
    # Normalização final de segurança
    df_proc = normalizar_dataframe_texto(df_proc)
    return df_proc

    # Layout do aplicativo
app.layout = dbc.Container([
    # CSS agora está em assets/custom.css (Dash carrega automaticamente)
    dcc.Store(id='store-data'),
    dcc.Store(id='store-machine-details'),
    # NAVBAR SUPERIOR
    dbc.Navbar(
        dbc.Container([
            html.Span("Controle de Produção ", className="navbar-brand dashboard-title mb-0 h1"),
            dbc.Button("Filtros", id='btn-open-filtros', color='light', size='sm', className='ms-auto', style={'background':'#0ea5a4','color':'white','border':'none'}),
        ], fluid=True),
        color="transparent", dark=False, className="mb-3 py-2 navbar-transparent"
    ),
    # (Upload movido para dentro do Offcanvas de filtros)
    # Offcanvas com filtros
    dbc.Offcanvas(id='offcanvas-filtros', title='Filtros', placement='start', is_open=False, className='offcanvas-custom', children=[
        html.Div([
            # Header com ícone
            html.Div([
                html.Div([
                    html.I(className="fas fa-filter", style={'marginRight': '8px', 'color': '#1d4ed8'}),
                    html.H5("Configurações de Filtro", className="mb-0", style={'color': '#1d4ed8', 'fontWeight': '600'})
                ], className="d-flex align-items-center"),
                html.P("Configure os parâmetros para análise dos dados", 
                       className="text-muted mb-0", style={'fontSize': '0.85rem', 'marginTop': '4px'})
            ], className="mb-4", style={'paddingBottom': '16px', 'borderBottom': '1px solid #e2e8f0'}),
            
            # Seção Upload
            dbc.Card([
                dbc.CardHeader([
                    html.I(className="fas fa-upload me-2", style={'color': '#6366f1'}),
                    "Upload de Dados"
                ], className="py-2 d-flex align-items-center", style={'backgroundColor': '#f8fafc', 'border': 'none', 'fontWeight': '600'}),
                dbc.CardBody([
                    dcc.Upload(
                        id='upload-data',
                        children=html.Div([
                            html.I(className="fas fa-cloud-upload-alt", style={'fontSize': '2rem', 'color': '#8b5cf6', 'marginBottom': '8px'}),
                            html.Div('Arraste e solte ou ', style={'marginBottom': '4px'}),
                            html.A('selecione um arquivo Excel', style={'color': '#1d4ed8', 'fontWeight': '500', 'textDecoration': 'underline'})
                        ], className="text-center"),
                        className='upload-box-enhanced',
                        style={'width':'100%','minHeight':'80px','border':'2px dashed #cbd5e1','borderRadius':'12px','textAlign':'center','margin':'8px 0','fontSize':'0.85rem','background':'linear-gradient(135deg, #f8fafc, #ffffff)','padding':'16px','transition':'all 0.3s ease','cursor':'pointer'},
                        multiple=False
                    ),
                    html.Div(id='output-data-upload', className="mt-2", style={'fontSize':'0.75rem'}),
                    dbc.Alert("Nenhum arquivo carregado.", color="warning", id="alert-no-data", is_open=True, className="py-2 mb-2", style={'fontSize':'0.75rem', 'borderRadius': '8px'}),
                    dbc.Alert(id="alert-data-loaded", color="success", is_open=False, className="py-2 mb-2", style={'fontSize':'0.75rem', 'borderRadius': '8px'}),
                    html.Div([
                        html.Div([
                            html.I(className="fas fa-download me-1", style={'fontSize': '0.7rem'}),
                            html.Span("Exportar: ", className="me-2", style={'fontSize':'0.75rem', 'fontWeight': '500'})
                        ], className="d-flex align-items-center mb-2"),
                        html.Div([
                            dbc.Button([
                                html.I(className="fas fa-file-excel me-1"),
                                "XLSX"
                            ], id="btn-download-xlsx", color="success", size="sm", className="me-2", 
                            style={'padding':'6px 12px','fontSize':'0.7rem', 'borderRadius': '6px'}),
                            dbc.Button([
                                html.I(className="fas fa-file-csv me-1"),
                                "CSV" 
                            ], id="btn-download-csv", color="info", size="sm", 
                            style={'padding':'6px 12px','fontSize':'0.7rem', 'borderRadius': '6px'}),
                        ], className="d-flex"),
                        dcc.Download(id="download-dataframe-xlsx"),
                        dcc.Download(id="download-dataframe-csv"),
                    ], id="download-container", style={"display":"none","marginTop":"12px"})
                ], className="py-3")
            ], className="mb-4 shadow-sm", style={'border': '1px solid #e2e8f0', 'borderRadius': '12px'}),
            
            # Seção Período
            html.Div([
                html.Div([
                    html.I(className="fas fa-calendar-alt me-2", style={'color': '#8b5cf6'}),
                    html.H6("Período", className="mb-0", style={'fontWeight': '600', 'color': '#374151'})
                ], className="d-flex align-items-center mb-3"),
                dbc.Card([
                    dbc.CardBody([
                        dcc.RadioItems(
                            id='modo-data', 
                            options=[
                                {'label': html.Div([
                                    html.I(className="fas fa-calendar-day me-2"),
                                    "Data única"
                                ], className="d-flex align-items-center"), 'value':'single'},
                                {'label': html.Div([
                                    html.I(className="fas fa-calendar-week me-2"), 
                                    "Intervalo"
                                ], className="d-flex align-items-center"), 'value':'range'}
                            ], 
                            value='single', 
                            inline=True, 
                            className='mb-3',
                            style={'fontSize': '0.85rem'}
                        ),
                        html.Div(id='container-data-single', children=[
                            dcc.DatePickerSingle(
                                id='date-single', 
                                date=datetime.now().strftime("%Y-%m-%d"), 
                                display_format='DD/MM/YYYY', 
                                clearable=False,
                                style={'width': '100%'}
                            )
                        ], style={'marginBottom':'8px'}),
                        html.Div(id='container-data-range', children=[
                            dcc.DatePickerRange(
                                id='date-range', 
                                start_date=datetime.now().strftime("%Y-%m-%d"), 
                                end_date=datetime.now().strftime("%Y-%m-%d"), 
                                display_format='DD/MM/YYYY',
                                style={'width': '100%'}
                            )
                        ], style={'display':'none','marginBottom':'8px'}),
                    ], className="py-2")
                ], className="border-0", style={'backgroundColor': '#f8fafc'})
            ], className="mb-4"),
            
            # Seção Turno
            html.Div([
                html.Div([
                    html.I(className="fas fa-clock me-2", style={'color': '#10b981'}),
                    html.H6("Turno", className="mb-0", style={'fontWeight': '600', 'color': '#374151'})
                ], className="d-flex align-items-center mb-3"),
                dbc.Card([
                    dbc.CardBody([
                        dcc.Dropdown(
                            id='turno-filter', 
                            options=[
                                {'label':'🌅 Todos os turnos','value':'todos'},
                                {'label':'🌅 Turno 1 (Manhã)','value':'1'},
                                {'label':'🌞 Turno 2 (Tarde)','value':'2'},
                                {'label':'🌙 Turno 3 (Noite)','value':'3'}
                            ], 
                            value='todos', 
                            clearable=False, 
                            style={'fontSize':'0.85rem', 'zIndex': '10002'},
                            className="enhanced-dropdown dropdown-high-z"
                        )
                    ], className="py-2")
                ], className="border-0", style={'backgroundColor': '#f8fafc'})
            ], className="mb-4"),
            
            # Seção Visualização
            html.Div([
                html.Div([
                    html.I(className="fas fa-chart-bar me-2", style={'color': '#f59e0b'}),
                    html.H6("Visualização", className="mb-0", style={'fontWeight': '600', 'color': '#374151'})
                ], className="d-flex align-items-center mb-3"),
                dbc.Card([
                    dbc.CardBody([
                        dcc.RadioItems(
                            id='view-type', 
                            options=[
                                {'label': html.Div([
                                    html.I(className="fas fa-cogs me-2"),
                                    "Por Máquinas"
                                ], className="d-flex align-items-center"), 'value':'maquinas'},
                                {'label': html.Div([
                                    html.I(className="fas fa-layer-group me-2"),
                                    "Por Grupos"
                                ], className="d-flex align-items-center"), 'value':'grupos'}
                            ], 
                            value='maquinas', 
                            inline=True, 
                            className='mb-3',
                            style={'fontSize': '0.85rem'}
                        ),
                        html.Div(id='container-tipos-maquinas', children=[
                            dcc.Checklist(
                                id='tipo-checklist', 
                                options=[{'label':'Planas','value':'tipo_Planas'}], 
                                value=['tipo_Planas'], 
                                className='mb-2'
                            )
                        ], style={'display':'none'}),
                    ], className="py-2")
                ], className="border-0", style={'backgroundColor': '#f8fafc'})
            ], className="mb-4"),
            
            # Seção Máquinas
            html.Div(id='container-maquinas', children=[
                html.Div([
                    html.I(className="fas fa-industry me-2", style={'color': '#ef4444'}),
                    html.H6("Seleção de Máquinas", className="mb-0", style={'fontWeight': '600', 'color': '#374151'})
                ], className="d-flex align-items-center mb-3"),
                dbc.Card([
                    dbc.CardBody([
                        dcc.Dropdown(
                            id='setor-checklist', 
                            options=[], 
                            value=[], 
                            multi=True, 
                            placeholder='🔍 Selecione as máquinas para análise...', 
                            style={'fontSize':'0.8rem'},
                            className="enhanced-dropdown mb-3"
                        ),
                        dbc.Button([
                            html.I(className="fas fa-check-circle me-2"),
                            "Selecionar Todas"
                        ], id='btn-select-all-maquinas', size='sm', color='primary', outline=True, 
                        className='w-100', style={'borderRadius': '8px', 'fontWeight': '500'})
                    ], className="py-2")
                ], className="border-0", style={'backgroundColor': '#f8fafc'})
            ], className="mb-4"),
            
            # Footer com dica
            html.Div([
                html.Hr(style={'margin': '24px 0 16px 0', 'border': 'none', 'borderTop': '1px solid #e2e8f0'}),
                html.Div([
                    html.I(className="fas fa-info-circle me-2", style={'color': '#6b7280'}),
                    html.Span("Feche o painel para aplicar os filtros e atualizar a visualização", 
                             style={'fontSize':'0.75rem','color':'#6b7280', 'fontStyle': 'italic'})
                ], className="d-flex align-items-center justify-content-center text-center")
            ])
        ], className='filtros-card-enhanced')
    ]),
        
    # KPIs + TABS
    dbc.Row([
        dbc.Col([
            dbc.Row(className="g-3 kpi-row", children=[
                    dbc.Col(dbc.Card([
                        dbc.CardHeader("Produção", style={"padding":"4px 8px 0 8px"}),
                        dbc.CardBody(html.Div([
                            html.Div(className="kpi-icon-wrapper", children=[html.Img(src="/assets/icons/producao.png")]),
                            html.H3(id="producao-total", className="kpi-value mb-0", style={"marginBottom":"0"})
                        ], className="kpi-inline", style={"display":"flex","alignItems":"center","gap":"8px","flexWrap":"nowrap"}))
                    ], className="h-100 kpi-card card-custom kpi-prod"), xs=6, md=4, lg=2),
                    dbc.Col(dbc.Card([
                        dbc.CardHeader("Paradas (h)", style={"padding":"4px 8px 0 8px"}),
                        dbc.CardBody(html.Div([
                            html.Div(className="kpi-icon-wrapper", children=[html.Img(src="/assets/icons/paradas.png")]),
                            html.H3(id="total-paradas", className="kpi-value mb-0", style={"marginBottom":"0"})
                        ], className="kpi-inline", style={"display":"flex","alignItems":"center","gap":"8px","flexWrap":"nowrap"}))
                    ], className="h-100 kpi-card card-custom kpi-stop"), xs=6, md=4, lg=2),
                    dbc.Col(dbc.Card([
                        dbc.CardHeader("Máx Produtiva", style={"padding":"4px 8px 0 8px"}),
                        dbc.CardBody(html.Div([
                            html.Div(className="kpi-icon-wrapper", children=[html.Img(src="/assets/icons/maquina.png")]),
                            html.H3(id="maquina-produtiva", className="kpi-value mb-0", style={"marginBottom":"0"})
                        ], className="kpi-inline", style={"display":"flex","alignItems":"center","gap":"8px","flexWrap":"nowrap"}))
                    ], className="h-100 kpi-card card-custom kpi-max"), xs=6, md=4, lg=2),
                    dbc.Col(dbc.Card([
                        dbc.CardHeader("Turno", style={"padding":"4px 8px 0 8px"}),
                        dbc.CardBody(html.Div([
                            html.Div(className="kpi-icon-wrapper", children=[html.Img(src="/assets/icons/turno.png")]),
                            html.H3(id="turno-atual", className="kpi-value mb-0", style={"marginBottom":"0"})
                        ], className="kpi-inline", style={"display":"flex","alignItems":"center","gap":"8px","flexWrap":"nowrap"}))
                    ], className="h-100 kpi-card card-custom kpi-shift"), xs=6, md=4, lg=2),
                    dbc.Col(dbc.Card([
                        dbc.CardHeader("Acertos", style={"padding":"4px 8px 0 8px"}),
                        dbc.CardBody(html.Div([
                            html.Div(className="kpi-icon-wrapper", children=[html.Img(src="/assets/icons/acertos.png")]),
                            html.H3(id="total-acertos", className="kpi-value mb-0", style={"marginBottom":"0"})
                        ], className="kpi-inline", style={"display":"flex","alignItems":"center","gap":"8px","flexWrap":"nowrap"}))
                    ], className="h-100 kpi-card card-custom kpi-acerto"), xs=6, md=4, lg=2),
                    dbc.Col(dbc.Card([
                        dbc.CardHeader("Eficiência", style={"padding":"4px 8px 0 8px"}),
                        dbc.CardBody(html.Div([
                            html.Div(className="kpi-icon-wrapper", children=[html.Img(src="/assets/icons/eficiencia.png")]),
                            html.H3(id="eficiencia-media", className="kpi-value mb-0", style={"marginBottom":"0"})
                        ], className="kpi-inline", style={"display":"flex","alignItems":"center","gap":"8px","flexWrap":"nowrap"}))
                    ], className="h-100 kpi-card card-custom kpi-efi"), xs=6, md=4, lg=2),
                ]),
            # Adiciona classe 'dash-tabs' para aplicar estilos personalizados de abas definidos em assets/custom.css
            dcc.Tabs(id='tabs-conteudo', value='tab-visao', className='mt-3 dash-tabs', children=[
                dcc.Tab(label='Visão Geral', value='tab-visao', className='tab', selected_className='tab--selected', children=[
                    dbc.Card([
                        dbc.CardHeader(id="header-producao-maquina", children="Produção por Máquina"),
                        dbc.CardBody(dcc.Graph(id="grafico-producao-maquina"))
                    ], className="mb-4 card-custom graph-card"),
                    dbc.Card([
                        dbc.CardHeader("Resumo por Máquina (Produção, Velocidade Média, Eficiência Média)"),
                        dbc.CardBody([
                            dcc.Graph(id="grafico-resumo-maquinas"),
                            html.Div(id="tabela-resumo-maquinas", style={"overflowX": "auto", "marginTop": "15px"})
                        ])
                    ], className="mb-4 card-custom graph-card"),
                ]),
                dcc.Tab(label='Paradas & Tempo', value='tab-paradas', className='tab', selected_className='tab--selected', children=[
                    # Linha com três gráficos (Manutenção, Aguardando e Ajustes) lado a lado
                    dbc.Row([
                        dbc.Col(
                            dbc.Card([
                                dbc.CardHeader("Paradas - Manutenção"),
                                dbc.CardBody([
                                    dcc.Graph(id="grafico-parada-manutencao"),
                                    html.Small("Soma de horas de paradas classificadas como manutenção por máquina no período filtrado.", className="text-muted")
                                ])
                            ], className="mb-4 card-custom graph-card"), md=4, width=12
                        ),
                        dbc.Col(
                            dbc.Card([
                                dbc.CardHeader("Paradas - Aguardando"),
                                dbc.CardBody([
                                    dcc.Graph(id="grafico-parada-aguardando"),
                                    html.Small("Paradas contendo 'AGUARD' (ex: aguardando material/liberação) agregadas por máquina.", className="text-muted")
                                ])
                            ], className="mb-4 card-custom graph-card"), md=4, width=12
                        ),
                        dbc.Col(
                            dbc.Card([
                                dbc.CardHeader("Paradas - Ajustes"),
                                dbc.CardBody([
                                    dcc.Graph(id="grafico-parada-ajuste"),
                                    html.Small("Paradas contendo 'AJUSTE' (ex: ajuste mec/eletrônico) agregadas por máquina.", className="text-muted")
                                ])
                            ], className="mb-4 card-custom graph-card"), md=4, width=12
                        )
                    ], className="g-3"),
                    dbc.Row([
                        dbc.Col(
                            dbc.Card([
                                dbc.CardHeader("Densidade de Paradas (Máquina x Turno)"),
                                dbc.CardBody([
                                    dcc.Graph(id="grafico-parada-densidade"),
                                    html.Small("Número de ocorrências de paradas por máquina e turno (quantidade, não horas). Quanto mais intenso, mais paradas.", className="text-muted")
                                ])
                            ], className="mb-4 card-custom graph-card"), md=6, width=12
                        ),
                        dbc.Col(
                            dbc.Card([
                                dbc.CardHeader("Frequência Paradas (Máquina x Descrição) por Turno"),
                                dbc.CardBody([
                                    dcc.Graph(id="grafico-parada-densidade3d"),
                                    html.Small("Bolhas (tamanho=quantidade) nas facetas de turno; top 12 descrições mais recorrentes.", className="text-muted")
                                ])
                            ], className="mb-4 card-custom graph-card"), md=6, width=12
                        )
                    ], className="g-3"),
                    dbc.Card([
                        dbc.CardHeader("Paradas - Acertos"),
                        dbc.CardBody([
                            dcc.Graph(id="grafico-parada-acerto"),
                            html.Small("Tempo de paradas do tipo ACERTO (ajuste/setup) consolidado por máquina.", className="text-muted")
                        ])
                    ], className="mb-4 card-custom graph-card"),
                    dbc.Card([
                        dbc.CardHeader("Paradas - Outros"),
                        dbc.CardBody([
                            dcc.Graph(id="grafico-parada-outros"),
                            html.Small("Demais paradas (excluídas Manutenção/Acerto/Aguardando).", className="text-muted")
                        ])
                    ], className="mb-4 card-custom graph-card"),
                    dbc.Row([
                        dbc.Col(
                            dbc.Card([
                                dbc.CardHeader("Acertos por Máquina"),
                                dbc.CardBody([
                                    dcc.Graph(id="grafico-acerto-maquina"),
                                    html.Small("Tempo total de acertos (setup/ajuste) por máquina.", className="text-muted")
                                ])
                            ], className="mb-4 card-custom graph-card"), md=6, width=12
                        ),
                        dbc.Col(
                            dbc.Card([
                                dbc.CardHeader("Acertos por Turno"),
                                dbc.CardBody([
                                    dcc.Graph(id="grafico-acerto-turno"),
                                    html.Small("Distribuição das horas de acerto por turno (1/2/3).", className="text-muted")
                                ])
                            ], className="mb-4 card-custom graph-card"), md=6, width=12
                        )
                    ], className="g-3 mb-2"),
                    dbc.Card([
                        dbc.CardHeader("Acertos por Item, Máquina e Turno"),
                        dbc.CardBody([
                            html.Div([
                                html.Span("Dimensão:", style={'fontSize':'0.75rem','marginRight':'6px'}),
                                dcc.RadioItems(
                                    id='acertos-dim-mode',
                                    options=[
                                        {'label':'Item+Máquina','value':'item_maquina'},
                                        {'label':'Máquina (agregado)','value':'maquina'}
                                    ],
                                    value='item_maquina',
                                    inline=True,
                                    style={'fontSize':'0.7rem'}
                                )
                            ], style={'marginBottom':'6px'}),
                            html.Div(id='tabela-acertos', style={'overflowX':'auto'})
                        ])
                    ], className='mb-4'),
                    dbc.Card([
                        dbc.CardHeader("Tempo de Parada por Máquina (horas)"),
                        dbc.CardBody(dcc.Graph(id="grafico-tempo-parada-maquina"))
                    ], className="mb-4 card-custom graph-card"),
                    dbc.Card([
                        dbc.CardHeader("Detalhe das Paradas por Descrição"),
                        dbc.CardBody(html.Div(id='tabela-detalhe-paradas', style={'maxHeight':'320px','overflowY':'auto','overflowX':'auto'}))
                    ], className='mb-4')
                ]),
                dcc.Tab(label='Velocidades', value='tab-vel', className='tab', selected_className='tab--selected', children=[
                    dbc.Card([
                        dbc.CardHeader("Velocidade por Máquina-Faca"),
                        dbc.CardBody(dcc.Graph(id="grafico-velocidade"))
                    ], className="mb-4 card-custom graph-card"),
                ]),
                dcc.Tab(label='Detalhes', value='tab-detalhes', className='tab', selected_className='tab--selected', children=[
                    dbc.Card([
                        dbc.CardHeader("Detalhes por Máquina: Itens, Produção e Velocidade Média"),
                        dbc.CardBody([
                            dbc.Row([
                                dbc.Col([
                                    html.Label("Selecione a Máquina:"),
                                    dcc.Dropdown(
                                        id="maquina-detalhe-dropdown",
                                        options=[],
                                        value=None,
                                        clearable=False
                                    )
                                ], width=12, md=6),
                                dbc.Col([
                                    html.Label("Ordenar por:"),
                                    dcc.RadioItems(
                                        id="ordenar-detalhe-radio",
                                        options=[
                                            {"label": "Produção Total", "value": "producao"},
                                            {"label": "Velocidade Média", "value": "velocidade"},
                                            {"label": "Item (A-Z)", "value": "item"}
                                        ],
                                        value="producao",
                                        inline=True
                                    )
                                ], width=12, md=6),
                            ], className="mb-3"),
                            dcc.Graph(id="grafico-detalhes-maquina"),
                            html.Hr(),
                            html.H5("Tabela de Detalhes", className="mt-3"),
                            dbc.Row([
                                dbc.Col([
                                    html.Div(id="tabela-detalhes-container", className="mt-3", style={"overflowX": "auto"})
                                ], width=12)
                            ])
                        ])
                    ], className="card-custom graph-card")
                ])
            ])
        ], width=12)
    ]),
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

# Callback para selecionar todas as máquinas (Dropdown multi)
@app.callback(
    Output('setor-checklist','value', allow_duplicate=True),
    Input('btn-select-all-maquinas','n_clicks'),
    State('setor-checklist','options'),
    prevent_initial_call=True
)
def select_all_maquinas(n, options):
    if not options:
        return dash.no_update
    return [o['value'] for o in options]

# Offcanvas open/close
@app.callback(
    Output('offcanvas-filtros','is_open'),
    Input('btn-open-filtros','n_clicks'),
    State('offcanvas-filtros','is_open'),
    prevent_initial_call=True
)
def toggle_offcanvas(n, is_open):
    if n:
        return not is_open
    return is_open

# Callback para atualizar a lista de máquinas (agora Dropdown multi) com base no tipo selecionado
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
            
            # IMPORTANTE: As máquinas são fornecidas exclusivamente pela coluna Centro Trabalho/maquina
            
            # Na versão processada, as máquinas estão na coluna 'maquina'
            if 'maquina' in df.columns:
                # Pegamos os valores únicos da coluna maquina
                maquinas_candidatas = df['maquina'].dropna().unique().tolist()
            # Na versão original, as máquinas estão na coluna 'Centro Trabalho'
            elif 'Centro Trabalho' in df.columns:
                # Pegamos os valores únicos da coluna Centro Trabalho
                maquinas_candidatas = df['Centro Trabalho'].dropna().unique().tolist()
            else:
                # Não encontramos a coluna de máquinas
                print("AVISO: Coluna de máquinas (maquina/Centro Trabalho) não encontrada.")
                maquinas_candidatas = []
            
            # Verificar se cada candidata é uma máquina válida
            maquinas = []
            for maquina in maquinas_candidatas:
                tipo = classificar_tipo_maquina(maquina)
                if tipo not in ['Ignorar', 'Desativada']:
                    maquinas.append(maquina)
                       
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
                     'eh_dia_atual', 'primeira_data_hora', 'tipo_maquina', 'tempo_producao', 'maquina', 
                     'ordem_servico', 'item', 'faca', 'parada', 'tempo_parada', 'turno', 'dia_produtivo', 
                     'velocidade_real', 'velocidade_padrao', 'duracao_turno', 'id_velocidade', 
                     'descricao_item', 'tempo_item_maquina', '_data_norm']
        
        # IMPORTANTE: As máquinas são fornecidas exclusivamente pela coluna Centro Trabalho
        # As demais colunas não devem ser consideradas como máquinas
        
        # Na versão processada, as máquinas estão na coluna 'maquina'
        if 'maquina' in df.columns:
            # Pegamos os valores únicos da coluna maquina
            maquinas_candidatas = df['maquina'].dropna().unique().tolist()
        # Na versão original, as máquinas estão na coluna 'Centro Trabalho'
        elif 'Centro Trabalho' in df.columns:
            # Pegamos os valores únicos da coluna Centro Trabalho
            maquinas_candidatas = df['Centro Trabalho'].dropna().unique().tolist()
        else:
            # Não encontramos a coluna de máquinas
            print("AVISO: Coluna de máquinas (maquina/Centro Trabalho) não encontrada.")
            maquinas_candidatas = []
            
        # Adicionar outras colunas especiais conhecidas que não são máquinas
        for col in df.columns:
            try:
                if isinstance(df[col].iloc[0], (list, dict, object)) or pd.isna(df[col].iloc[0]):
                    if col not in colunas_nao_maquinas:
                        colunas_nao_maquinas.append(col)
            except Exception:
                pass
        
        # Verificar se cada candidata é uma máquina válida
        maquinas = []
        for maquina in maquinas_candidatas:
            tipo = classificar_tipo_maquina(maquina)
            if tipo not in ['Ignorar', 'Desativada']:
                maquinas.append(maquina)
        
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
            except (ValueError, TypeError, KeyError):
                # Em caso de erro, apenas adicionar sem informação de soma
                nome_tipo = tipo.replace('tipo_', '').replace('_', ' ')
                opcoes_tipos.append({'label': f"{nome_tipo}", 'value': tipo})
        
        today = datetime.now().strftime("%Y-%m-%d")
        data_inicial = df['data'].min() if len(df) > 0 else today
        data_final = df['data'].max() if len(df) > 0 else today
        # Garantir ordenação das datas para pegar a mais recente corretamente
        try:
            datas_convertidas = pd.to_datetime(df['data'], errors='coerce')
            data_mais_recente = datas_convertidas.max().strftime('%Y-%m-%d') if datas_convertidas.notna().any() else today
        except Exception:
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
     Output("total-acertos", "children"),
     Output("turno-atual", "children"),
     Output("eficiencia-media", "children"),
     Output("grafico-producao-maquina", "figure"),
     Output("grafico-parada-manutencao", "figure"),
     Output("grafico-parada-ajuste", "figure"),
    Output("grafico-parada-densidade", "figure"),
    Output("grafico-parada-densidade3d", "figure"),
     Output("grafico-parada-acerto", "figure"),
     Output("grafico-parada-aguardando", "figure"),
     Output("grafico-parada-outros", "figure"),
     Output("grafico-acerto-maquina", "figure"),
     Output("grafico-acerto-turno", "figure"),
    Output("grafico-velocidade", "figure"),
    Output("grafico-tempo-parada-maquina", "figure"),
     Output("tabela-acertos", "children"),
    Output("tabela-detalhe-paradas", "children"),
     Output("store-machine-details", "data"),
     Output("maquina-detalhe-dropdown", "options")],
    [Input("modo-data", "value"),
     Input("date-single", "date"),
     Input("date-range", "start_date"),
     Input("date-range", "end_date"),
     Input("turno-filter", "value"),
     Input("view-type", "value"),
     Input("setor-checklist", "value"),
     Input("tipo-checklist", "value"),
    Input("store-data", "data"),
    Input("acertos-dim-mode","value")]
)
def update_dashboard(modo_data, selected_date, start_date, end_date, turno_filter, view_type, maquinas_selecionadas, tipos_selecionados, data_json, acertos_dim_mode):
    t0 = time.perf_counter()
    # Converter os dados JSON de volta para DataFrame
    df = pd.DataFrame(data_json)
    
    # Debug: mostrar estrutura dos dados recebidos
    debug_print(f"[DEBUG] DataFrame recebido: {len(df)} linhas")
    debug_print(f"[DEBUG] Colunas disponíveis: {list(df.columns)}")
    if len(df) > 0:
        debug_print(f"[DEBUG] Primeiras 3 colunas: {df.columns[:3].tolist()}")
        # Verificar se há colunas que parecem máquinas (CA01, CA02, etc)
        cols_maquina = [col for col in df.columns if col.startswith('CA') and len(col) <= 4]
        debug_print(f"[DEBUG] Colunas que parecem máquinas: {cols_maquina[:10]}")  # mostrar apenas primeiras 10

    # Capturar lista global de máquinas disponíveis no dataset completo (antes de qualquer filtro)
    todas_maquinas_orig = []
    if 'maquina' in df.columns:
        try:
            serie_maqs_orig = df['maquina']
            if len(serie_maqs_orig) > 0:
                if isinstance(serie_maqs_orig.iloc[0], list):
                    flat = []
                    for lst in serie_maqs_orig:
                        if isinstance(lst, list):
                            flat.extend([v.strip() for v in lst if isinstance(v,str) and v.strip()])
                    todas_maquinas_orig = sorted(list(set(flat)))
                else:
                    todas_maquinas_orig = sorted(list(set([v.strip() for v in serie_maqs_orig.dropna().astype(str) if v.strip() != ''])))
        except Exception:
            pass

    # Garantir presença das máquinas CA01 a CA08 mesmo que não apareçam no filtro atual
    base_defaults = [f"CA{str(i).zfill(2)}" for i in range(1,9)]
    if todas_maquinas_orig:
        for m in base_defaults:
            if m not in todas_maquinas_orig:
                todas_maquinas_orig.append(m)
        todas_maquinas_orig = sorted(list(set(todas_maquinas_orig)))
    else:
        todas_maquinas_orig = base_defaults.copy()
    
    # Converter as listas armazenadas de volta para objetos Python
    if 'datetime_inicio' in df.columns:
        # Se os datetimes estão armazenados como listas de strings, converter de volta para datetime
        try:
            if isinstance(df['datetime_inicio'].iloc[0], list):
                # Criar uma nova coluna com o primeiro datetime de cada linha para facilitar a filtragem
                df['primeira_data_hora'] = df['datetime_inicio'].apply(
                    lambda x: pd.to_datetime(x[0]) if x and len(x) > 0 else None
                )
        except (ValueError, TypeError):
            pass
    
    # ================== NOVA NORMALIZAÇÃO DE DATAS ==================
    def _parse_date(val):
        if val is None or (isinstance(val, float) and pd.isna(val)):
            return None
        for dayfirst in (False, True):
            try:
                return pd.to_datetime(val, dayfirst=dayfirst).date()
            except Exception:
                continue
        return None

    # Criar coluna normalizada robusta
    raw_series = df['data'] if 'data' in df.columns else pd.Series([], dtype=object)
    try:
        df['_data_norm'] = raw_series.apply(lambda x: _parse_date(x))
    except Exception:
        df['_data_norm'] = pd.NaT

    sel_date_obj = _parse_date(selected_date)
    start_obj = _parse_date(start_date)
    end_obj = _parse_date(end_date)

    # Se todas as datas viraram None tentar segunda estratégia: assumir formato '%Y-%m-%d'
    if df['_data_norm'].notna().sum() == 0 and 'data' in df.columns:
        try:
            parsed_alt = pd.to_datetime(df['data'].astype(str).str.strip(), format='%Y-%m-%d', errors='coerce').dt.date
            if parsed_alt.notna().sum() > 0:
                df['_data_norm'] = parsed_alt
        except Exception:
            pass

    # Terceira estratégia: usar dia_produtivo se existir
    if df['_data_norm'].notna().sum() == 0 and 'dia_produtivo' in df.columns:
        try:
            parsed_dia_prod = pd.to_datetime(df['dia_produtivo'].astype(str).str[:10], errors='coerce').dt.date
            if parsed_dia_prod.notna().sum() > 0:
                df['_data_norm'] = parsed_dia_prod
        except Exception:
            pass

    # Quarta estratégia: derivar a partir de datetime_inicio aplicando regra 06:00
    if df['_data_norm'].notna().sum() == 0 and 'datetime_inicio' in df.columns:
        try:
            base_dt = df['datetime_inicio'].apply(lambda v: v[0] if isinstance(v, list) and v else v)
            base_dt = pd.to_datetime(base_dt, errors='coerce')
            if base_dt.notna().sum() > 0:
                # Aplicar regra: hora < 06 => dia anterior
                ajust = base_dt - pd.to_timedelta((base_dt.dt.hour < 6).astype(int), unit='D')
                df['_data_norm'] = ajust.dt.date
        except Exception as e:
            debug_print(f"[DEBUG] Falha derivar de datetime_inicio: {e}")

    # Fallback final: se ainda não temos datas válidas, NÃO somar tudo; retornar vazio para evitar somatório incorreto
    if df['_data_norm'].notna().sum() == 0:
        debug_print('[DEBUG] Nenhuma data válida encontrada após todas as estratégias; retornando figuras vazias.')
        empty_fig = px.line(title="Datas indisponíveis no conjunto de dados")
        empty_fig.update_layout(xaxis_visible=False, yaxis_visible=False,
                                annotations=[{"text":"Datas indisponíveis no conjunto de dados",
                                              "showarrow":False,"font":{"size":16}}])
        empty_fig = style_fig(empty_fig)
        return (
            "0", "0", "N/A", "0",
            "Todos os Turnos", "0%",
            # 10 figuras (producao, parada-manutencao, parada-acerto, parada-aguardando, parada-outros,
            # acerto-maquina, acerto-turno, velocidade, tempo-item, eficiencia)
            empty_fig, empty_fig, empty_fig, empty_fig, empty_fig, empty_fig, empty_fig,
            empty_fig, empty_fig, empty_fig,
            html.Div("Sem acertos"), html.Div("Sem paradas"), {}, []
        )
    else:
        if modo_data == 'single':
            if sel_date_obj is not None:
                mask_data = df['_data_norm'] == sel_date_obj
                periodo_str = f"no Dia {sel_date_obj}"
            else:
                # Se não foi possível interpretar a data selecionada, não retornar nada para deixar claro
                mask_data = pd.Series(False, index=df.index)
                periodo_str = f"no Dia (inválido)"
        else:  # range
            if start_obj is None:
                start_obj = df['_data_norm'].min()
            if end_obj is None:
                end_obj = df['_data_norm'].max()
            mask_data = (df['_data_norm'] >= start_obj) & (df['_data_norm'] <= end_obj)
            periodo_str = f"no Período de {start_obj} a {end_obj}"

    if DEBUG:
        try:
            exemplos = df['_data_norm'].dropna().unique().tolist()[:10]
            debug_print(f"[DEBUG] Datas únicas (até 10): {exemplos}")
            debug_print(f"[DEBUG] selected={sel_date_obj} start={start_obj} end={end_obj} modo={modo_data} linhas_filtradas={mask_data.sum()} total={len(df)}")
        except Exception:
            pass
    # ================================================================
    # Reforço: usar datetime_inicio para janela produtiva 06:00-06:00 se disponível
    try:
        if 'datetime_inicio' in df.columns and df['datetime_inicio'].notna().any():
            serie_dt = df['datetime_inicio'].apply(lambda v: v[0] if isinstance(v, list) and v else v)
            serie_dt = pd.to_datetime(serie_dt, errors='coerce')
            if modo_data == 'single' and sel_date_obj is not None:
                ini = datetime.combine(sel_date_obj, dtime(6,0))
                fim = ini + timedelta(days=1)
                mask_prod = (serie_dt >= ini) & (serie_dt < fim)
                if mask_prod.sum() > 0:
                    mask_data = mask_prod
                    periodo_str = f"no Dia Produtivo {sel_date_obj} (06:00-06:00)"
            elif modo_data == 'range' and start_obj is not None and end_obj is not None:
                ini = datetime.combine(start_obj, dtime(6,0))
                fim = datetime.combine(end_obj, dtime(6,0)) + timedelta(days=1)
                mask_prod = (serie_dt >= ini) & (serie_dt < fim)
                if mask_prod.sum() > 0:
                    mask_data = mask_prod
                    periodo_str = f"no Período Produtivo de {start_obj} a {end_obj} (06:00-06:00)"
    except Exception as e:
        debug_print(f"[DEBUG] Falha reforço datetime_inicio: {e}")

    # Se o filtro single está retornando todas as linhas (ou zero) e não estamos em DEBUG, habilitar diagnóstico mínimo
    if not DEBUG and modo_data == 'single' and len(df) > 0:
        if mask_data.sum() == len(df) or mask_data.sum() == 0:
            # Mostrar 5 primeiras datas brutas e normalizadas
            try:
                print("[INFO] Diagnóstico filtro de data:")
                print("  selected_date=", selected_date)
                print("  _data_norm primeiras:")
                print(df[['data','_data_norm']].head(5))
                print(f"  Linhas correspondentes: {mask_data.sum()} / {len(df)}")
            except Exception:
                pass
    
    # Aplicar filtro de turno se não for 'todos'
    if turno_filter != 'todos':
        if 'turno' in df.columns:
            # Verificar se a coluna turno existe e contém valores como listas
            if isinstance(df['turno'].iloc[0], list):
                # Filtrar registros que contêm o turno selecionado na lista
                mask_turno = df['turno'].apply(lambda x: turno_filter in x if isinstance(x, list) else False)
            else:
                # Filtrar registros com turno igual ao selecionado
                mask_turno = df['turno'] == turno_filter
            
            # Combinar máscaras de data e turno
            mask = mask_data & mask_turno
            periodo_str += f" - Turno {turno_filter}"
        else:
            # Se não houver coluna de turno, usar apenas a máscara de data
            mask = mask_data
    else:
        # Se o filtro for 'todos', usar apenas a máscara de data
        mask = mask_data
        periodo_str += " - Todos os Turnos"
    
    df_filtrado = df.loc[mask].copy()
    
    # Calcular turno atual baseado na hora atual ou usar o turno filtrado
    if turno_filter == 'todos':
        turno_atual = "Todos os Turnos"
    else:
        # Usar o turno do filtro
        turno_atual = f"Turno {turno_filter}"
    
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
        empty_fig = style_fig(empty_fig)
        return (
            "0", "0", "N/A", "0", 
            "Todos os Turnos", "0%",
            # 10 figuras vazias
            empty_fig, empty_fig, empty_fig, empty_fig, empty_fig, empty_fig, empty_fig,
            empty_fig, empty_fig, empty_fig,
            html.Div("Sem acertos"), html.Div("Sem paradas"), {}, []
        )
    
    # Calcular métricas principais com base no modo de visualização e seleções
    if view_type == 'maquinas' and maquinas_selecionadas:
        # Verificar quais máquinas selecionadas existem como registros de máquina na coluna 'maquina' ou 'Centro Trabalho'
        producao_total = 0
        
        # Verificar se temos a coluna 'maquina' ou 'Centro Trabalho' para identificar registros
        if 'maquina' in df_filtrado.columns:
            # Filtrar registros que correspondem às máquinas selecionadas
            for maq in maquinas_selecionadas:
                df_maq = df_filtrado[df_filtrado['maquina'] == maq]
                if not df_maq.empty:
                    producao_total += df_maq['producao'].sum()
        elif 'Centro Trabalho' in df_filtrado.columns:
            # Filtrar registros que correspondem às máquinas selecionadas
            for maq in maquinas_selecionadas:
                df_maq = df_filtrado[df_filtrado['Centro Trabalho'] == maq]
                if not df_maq.empty:
                    producao_total += df_maq['producao'].sum()
        else:
            # Fallback para o caso de não termos as colunas de identificação de máquina
            producao_total = df_filtrado['producao'].sum()
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
    
    # Total de paradas (em horas) - versão otimizada vetorizada
    colunas_paradas = [col for col in df_filtrado.columns if col.startswith('parada_')]
    total_paradas = 0
    if colunas_paradas:
        try:
            if view_type == 'maquinas' and maquinas_selecionadas:
                maquinas_existentes = [c for c in maquinas_selecionadas if c in df_filtrado.columns]
                if maquinas_existentes:
                    bloc = df_filtrado[maquinas_existentes].apply(pd.to_numeric, errors='coerce').fillna(0)
                    mascara = bloc.gt(0).any(axis=1)
                    if mascara.any():
                        total_paradas = df_filtrado.loc[mascara, colunas_paradas].sum().sum()
            elif view_type == 'grupos' and tipos_selecionados:
                tipos_validos = [c for c in tipos_selecionados if c in df_filtrado.columns]
                if tipos_validos:
                    bloc = df_filtrado[tipos_validos].apply(pd.to_numeric, errors='coerce').fillna(0)
                    mascara = bloc.gt(0).any(axis=1)
                    if mascara.any():
                        total_paradas = df_filtrado.loc[mascara, colunas_paradas].sum().sum()
            else:
                total_paradas = df_filtrado[colunas_paradas].sum().sum()
        except Exception as e:
            print(f"Erro (paradas vetorizado): {e}")
            try:
                total_paradas = df_filtrado[colunas_paradas].sum().sum()
            except (ValueError, TypeError, KeyError):
                total_paradas = 0
    
    # Identificar a máquina mais produtiva
    # Se foram selecionadas máquinas específicas, usar essas
    if maquinas_selecionadas:
        colunas_maquinas = maquinas_selecionadas
    # Caso contrário, obter máquinas da coluna 'maquina'
    elif 'maquina' in df_filtrado.columns:
        # Obter valores únicos da coluna maquina
        maquinas_unicas = df_filtrado['maquina'].dropna().unique().tolist()
        # Filtramos apenas as máquinas que também são colunas no DataFrame
        colunas_maquinas = [m for m in maquinas_unicas if m in df_filtrado.columns]
    # Verificar coluna Centro Trabalho como backup
    elif 'Centro Trabalho' in df_filtrado.columns:
        # Obter valores únicos da coluna Centro Trabalho
        maquinas_unicas = df_filtrado['Centro Trabalho'].dropna().unique().tolist()
        # Filtramos apenas as máquinas que também são colunas no DataFrame
        colunas_maquinas = [m for m in maquinas_unicas if m in df_filtrado.columns]
    # Se não encontrou de nenhuma forma, usar lista vazia
    else:
        colunas_maquinas = []
        
    # Se não estamos usando máquinas selecionadas manualmente, filtrar apenas máquinas válidas
    if not maquinas_selecionadas and colunas_maquinas:
        colunas_maquinas_validas = []
        for col in colunas_maquinas:
            tipo = classificar_tipo_maquina(col)
            if tipo not in ['Ignorar', 'Desativada']:
                colunas_maquinas_validas.append(col)
        colunas_maquinas = colunas_maquinas_validas
    
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
    
    # Total de ACERTOS (1 por item conforme regra solicitada)
    total_acertos = 0
    try:
        if 'parada' in df_filtrado.columns:
            serie_parada = df_filtrado['parada'].astype(str)
            mask_acerto = serie_parada.str.contains('ACERTO', case=False, na=False)
            acertos_base = df_filtrado[mask_acerto]
            if not acertos_base.empty:
                if 'item' in acertos_base.columns:
                    total_acertos = acertos_base['item'].dropna().astype(str).str.strip().nunique()
                else:
                    total_acertos = 0
    except Exception as e:
        print(f"Erro ao calcular total de acertos: {e}")
    
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
                
                # Calcular produção total para exibir como anotação
                producao_total_formatada = f"{producao_total:,.0f}".replace(',', '.')
                
                fig_producao_diaria = px.bar(df_hora, x='Hora', y='Produção',
                                           title=f'Produção por Hora {periodo_str}',
                                           labels={'Produção': 'Quantidade', 'Hora': 'Hora do Dia'})
                
                # Adicionar anotação com o valor total de produção ANTES do style_fig
                fig_producao_diaria.add_annotation(
                    text=f"Produção total: {producao_total_formatada}",
                    xref="paper", yref="paper",
                    x=0.5, y=0.95,  # Posiciona dentro do gráfico para não interferir com header
                    showarrow=False,
                    font=dict(size=14, color="black"),
                    bgcolor="rgba(255, 255, 255, 0.8)",
                    bordercolor="rgba(0, 0, 0, 0.5)",
                    borderwidth=1,
                    borderpad=4
                )
                fig_producao_diaria = style_bar(fig_producao_diaria)  # Aplicar cores purple
                fig_producao_diaria = style_fig(fig_producao_diaria, legend_bottom=True, tight=True)
            else:
                fig_producao_diaria = go.Figure()
                fig_producao_diaria.update_layout(title=f'Produção Total {periodo_str}')
                fig_producao_diaria.add_annotation(text=f"Produção total: {producao_total:,.0f}".replace(',', '.'),
                                                   x=0.5,y=0.5,showarrow=False,font=dict(size=16))
                fig_producao_diaria = style_fig(fig_producao_diaria, legend_bottom=True, tight=True)
        except Exception as e:
            print(f"Erro ao criar gráfico por hora: {e}")
            fig_producao_diaria = go.Figure()
            fig_producao_diaria.update_layout(title=f'Produção Total {periodo_str}')
            fig_producao_diaria.add_annotation(text=f"Produção total: {producao_total:,.0f}".replace(',', '.'),
                                               x=0.5,y=0.5,showarrow=False,font=dict(size=16))
            fig_producao_diaria = style_fig(fig_producao_diaria, legend_bottom=True, tight=True)
    elif modo_data == 'range' and len(df_filtrado) > 0:
        # Para intervalo de datas: mostrar produção diária
        df_dia = df_filtrado.groupby('data')['producao'].sum().reset_index()
        if len(df_dia) > 0:
            # Calcular produção total para exibir como anotação
            producao_total_formatada = f"{producao_total:,.0f}".replace(',', '.')
            
            fig_producao_diaria = px.line(df_dia, x='data', y='producao',
                                        title=f'Produção Diária {periodo_str}',
                                        labels={'producao': 'Quantidade', 'data': 'Data'})
            
            # Adicionar anotação com o valor total de produção ANTES do style_fig
            fig_producao_diaria.add_annotation(
                text=f"Produção total: {producao_total_formatada}",
                xref="paper", yref="paper",
                x=0.5, y=0.95,  # Posiciona dentro do gráfico para não interferir com header
                showarrow=False,
                font=dict(size=14, color="black"),
                bgcolor="rgba(255, 255, 255, 0.8)",
                bordercolor="rgba(0, 0, 0, 0.5)",
                borderwidth=1,
                borderpad=4
            )
            fig_producao_diaria = style_fig(fig_producao_diaria, legend_bottom=True, tight=True)
        else:
            fig_producao_diaria = go.Figure()
            fig_producao_diaria.update_layout(title=f'Produção Total {periodo_str}')
            fig_producao_diaria.add_annotation(text="Sem dados disponíveis para o período", x=0.5,y=0.5,showarrow=False,font=dict(size=16))
            fig_producao_diaria = style_fig(fig_producao_diaria, legend_bottom=True, tight=True)
    else:
        # Gráfico padrão quando não houver dados
            fig_producao_diaria = go.Figure()
            fig_producao_diaria.update_layout(title=f'Produção Total {periodo_str}')
            fig_producao_diaria.add_annotation(text=f"Produção total: {producao_total:,.0f}".replace(',', '.'), x=0.5,y=0.5,showarrow=False,font=dict(size=16))
            fig_producao_diaria = style_fig(fig_producao_diaria, legend_bottom=True, tight=True)
    
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
            # Renomear a coluna para manter comp
            # m o resto do código
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
    
    # Validar dados antes de criar gráfico para evitar ValueError do Plotly
    try:
        if len(df_maquinas) == 0 or df_maquinas['Produção'].isna().all():
            raise ValueError("DataFrame vazio ou sem dados válidos")
            
        if modo_data == 'single':
            # Para data única: mostrar barras por máquina/grupo
            fig_producao_maquina = px.bar(df_maquinas, x='Máquina', y='Produção', 
                                        title=titulo_grafico)
            fig_producao_maquina = style_bar(fig_producao_maquina)  # Aplicar cores purple
            fig_producao_maquina = style_fig(fig_producao_maquina, legend_bottom=False, tight=True)
        else:
            # Para intervalo de datas: mostrar barras agrupadas por data e máquina/grupo
            if 'data' in df_maquinas.columns:
                fig_producao_maquina = px.bar(df_maquinas, x='data', y='Produção', color='Máquina', barmode='stack',
                                           title=titulo_grafico)
                fig_producao_maquina = style_bar(fig_producao_maquina)  # Aplicar cores purple
                fig_producao_maquina = style_fig(fig_producao_maquina, legend_bottom=True, tight=True)
            else:
                fig_producao_maquina = px.bar(df_maquinas, x='Máquina', y='Produção', 
                                           title=titulo_grafico)
                fig_producao_maquina = style_bar(fig_producao_maquina)  # Aplicar cores purple
                fig_producao_maquina = style_fig(fig_producao_maquina, legend_bottom=False, tight=True)
    except Exception as e:
        # Fallback seguro com go.Figure se px.bar falhar
        print(f"[WARN] Erro no gráfico de máquinas, usando fallback: {e}")
        fig_producao_maquina = go.Figure()
        fig_producao_maquina.update_layout(
                title=titulo_grafico,
                annotations=[{"text": "Dados indisponíveis para gráfico", "showarrow": False, "font": {"size": 16}}],
                xaxis={"visible": False},
                yaxis={"visible": False}
            )
        fig_producao_maquina = style_fig(fig_producao_maquina, legend_bottom=False, tight=True)
    
    # 3. Gráfico de análise de paradas
    colunas_paradas = [col for col in df_filtrado.columns if col.startswith('parada_')]
    if colunas_paradas:
        # Calcular total de paradas por tipo
        df_paradas_sum = df_filtrado[colunas_paradas].sum().reset_index()
        df_paradas_sum.columns = ['Tipo de Parada', 'Tempo (horas)']
        df_paradas_sum['Tipo de Parada'] = df_paradas_sum['Tipo de Parada'].str.replace('parada_', '')
        
        fig_paradas = px.pie(df_paradas_sum, values='Tempo (horas)', names='Tipo de Parada',
                          title=f'Distribuição de Paradas {periodo_str}')
        fig_paradas = style_pie(fig_paradas)
    else:
        # Criar gráfico vazio se não houver dados de paradas
        fig_paradas = px.pie(title="Sem dados de paradas disponíveis")
        fig_paradas.update_layout(
            annotations=[{"text": "Sem dados de paradas disponíveis", "showarrow": False, "font": {"size": 16}}]
        )
        fig_paradas = style_pie(fig_paradas)
    
    # 5. Gráfico de tempo de parada por máquina (stacked bars por descrição de parada)
    try:
        if 'maquina' in df_filtrado.columns and 'parada' in df_filtrado.columns and 'tempo_parada' in df_filtrado.columns:
            # Normalizar colunas e garantir numérico
            df_paradas_det = df_filtrado[['maquina', 'parada', 'tempo_parada']].copy()
            df_paradas_det['parada'] = df_paradas_det['parada'].fillna('Desconhecida').astype(str)
            df_paradas_det['maquina'] = df_paradas_det['maquina'].fillna('Desconhecido').astype(str)
            df_paradas_det['tempo_parada'] = pd.to_numeric(df_paradas_det['tempo_parada'], errors='coerce').fillna(0.0)

            # Agrupar soma do tempo de parada por máquina e descrição
            df_paradas_agg = df_paradas_det.groupby(['maquina', 'parada'], as_index=False)['tempo_parada'].sum()
            if len(df_paradas_agg) > 0:
                # Criar gráfico de barras empilhadas (por máquina, color = descrição da parada)
                fig_tempo_parada_maquina = px.bar(
                    df_paradas_agg,
                    x='maquina',
                    y='tempo_parada',
                    color='parada',
                    title=f'Tempo de Parada por Máquina {periodo_str}',
                    labels={'tempo_parada': 'Tempo de Parada (h)', 'maquina': 'Máquina', 'parada': 'Descrição da Parada'},
                    color_discrete_sequence=BAR_COLORS
                )
                fig_tempo_parada_maquina.update_layout(barmode='stack', legend_title_text='Descrição da Parada', margin=dict(l=50, r=20, t=60, b=80))
                fig_tempo_parada_maquina = style_bar(fig_tempo_parada_maquina)  # Aplicar cores purple
                fig_tempo_parada_maquina = style_fig(fig_tempo_parada_maquina, legend_bottom=True, tight=True)
            else:
                fig_tempo_parada_maquina = px.bar(title="Sem dados de paradas para o período")
                fig_tempo_parada_maquina.update_layout(annotations=[{"text": "Sem dados de paradas para o período", "showarrow": False, "font": {"size": 16}}])
                fig_tempo_parada_maquina = style_fig(fig_tempo_parada_maquina, legend_bottom=True, tight=True)
        else:
            fig_tempo_parada_maquina = px.bar(title="Sem dados de paradas disponíveis")
            fig_tempo_parada_maquina.update_layout(annotations=[{"text": "Sem dados de paradas disponíveis", "showarrow": False, "font": {"size": 16}}])
            fig_tempo_parada_maquina = style_fig(fig_tempo_parada_maquina, legend_bottom=True, tight=True)
    except Exception as e:
        fig_tempo_parada_maquina = px.bar(title=f"Erro ao gerar gráfico de paradas: {e}")
        fig_tempo_parada_maquina.update_layout(annotations=[{"text": f"Erro: {e}", "showarrow": False, "font": {"size": 14}}])
        fig_tempo_parada_maquina = style_fig(fig_tempo_parada_maquina, legend_bottom=True, tight=True)
    
    # 4. Gráfico de velocidade por máquina-faca
    if 'velocidade_id' in df_filtrado.columns and len(df_filtrado['velocidade_id'].unique()) > 1:
        # Agrupar por velocidade_id e calcular métricas
        df_velocidade = df_filtrado.groupby('velocidade_id').agg({'producao': 'sum'}).reset_index()
        df_velocidade = df_velocidade.sort_values('producao', ascending=False).head(10)
        fig_velocidade = px.bar(
            df_velocidade,
            x='velocidade_id', y='producao',
            title=f'Produção por Máquina-Faca {periodo_str} (Top 10)',
            labels={'velocidade_id': 'Máquina-Faca', 'producao': 'Produção Total'}
        )
        fig_velocidade = style_bar(fig_velocidade)  # Aplicar cores purple
        fig_velocidade = style_fig(fig_velocidade, legend_bottom=False, tight=True)
    else:
        fig_velocidade = px.bar(title="Sem dados de velocidade disponíveis")
        fig_velocidade.update_layout(annotations=[{"text": "Sem dados de velocidade disponíveis", "showarrow": False, "font": {"size": 16}}])
        fig_velocidade = style_fig(fig_velocidade, legend_bottom=False, tight=True)
    
    # Calcular eficiência média global consistente com tabela resumo (ponderada por tempo_producao dos detalhes)
    eficiencia_media = 0.0
    try:
        # Se já existir detalhes_maquinas calculados anteriormente nesta função podemos usá-los
        # (caso contrário caímos para cálculo direto no df_filtrado)
        if 'detalhes_maquinas' in locals() and detalhes_maquinas:
            soma_num = 0.0; soma_den = 0.0
            for regs in detalhes_maquinas.values():
                for r in regs:
                    t = float(r.get('tempo_producao',0) or 0)
                    e = r.get('eficiencia', None)
                    if e is None:
                        continue
                    try:
                        e = float(e)
                    except Exception:
                        continue
                    if t>0 and e>0:
                        soma_num += e * t
                        soma_den += t
            if soma_den>0:
                eficiencia_media = soma_num / soma_den
            elif 'eficiencia' in df_filtrado.columns:
                eficiencia_media = pd.to_numeric(df_filtrado['eficiencia'], errors='coerce').fillna(0).mean()
        elif 'eficiencia' in df_filtrado.columns:
            if 'tempo_producao' in df_filtrado.columns and df_filtrado['tempo_producao'].gt(0).any():
                pesos = pd.to_numeric(df_filtrado['tempo_producao'], errors='coerce').fillna(0)
                efics = pd.to_numeric(df_filtrado['eficiencia'], errors='coerce').fillna(0)
                if pesos.sum()>0:
                    eficiencia_media = (efics*pesos).sum()/pesos.sum()
                else:
                    eficiencia_media = efics.mean()
            else:
                eficiencia_media = pd.to_numeric(df_filtrado['eficiencia'], errors='coerce').fillna(0).mean()
    except Exception as e:
        print(f"Erro ao calcular eficiência média: {e}")
    # Formatar para exibição como porcentagem
    eficiencia_media_str = f"{eficiencia_media * 100:.1f}%"
    
    # Criar gráfico de eficiência por máquina-item
    if 'eficiencia' in df_filtrado.columns and 'item' in df_filtrado.columns and 'maquina' in df_filtrado.columns and len(df_filtrado) > 0:
        # Agrupar por máquina e item, calculando a eficiência média
        df_eficiencia = df_filtrado.groupby(['maquina', 'item'])['eficiencia'].mean().reset_index()
        
        # Ordenar por eficiência para destacar os extremos
        df_eficiencia = df_eficiencia.sort_values('eficiencia', ascending=False)
        
        # Limitar para os top 20 para melhor visualização
        df_eficiencia = df_eficiencia.head(20)
        
        # Converter eficiência para porcentagem para visualização
        df_eficiencia['eficiencia_pct'] = df_eficiencia['eficiencia'] * 100
        
        # Criar o gráfico de barras
        fig_eficiencia = px.bar(
            df_eficiencia,
            x='item', y='eficiencia_pct', color='maquina',
            title=f'Eficiência por Máquina-Item {periodo_str} (Top 20)',
            labels={'eficiencia_pct': 'Eficiência (%)', 'item': 'Item', 'maquina': 'Máquina'}
        )
        fig_eficiencia = style_fig(fig_eficiencia, legend_bottom=True, tight=True)
        
        # Ajuste adicional para legendas com muitos itens
        fig_eficiencia.update_layout(margin=dict(b=85))  # Margem extra para legenda com máquinas
        
        # Adicionar linha de 100% para referência
        try:
            fig_eficiencia.add_shape(
                type="line",
                x0=-0.5,
                y0=100,
                x1=len(df_eficiencia) - 0.5,
                y1=100,
                line=dict(color="#d62728", width=2, dash="dash")
            )
        except Exception:
            pass
    else:
        # Criar gráfico vazio se não houver dados de eficiência
        fig_eficiencia = px.bar(title="Dados de eficiência não disponíveis")
        fig_eficiencia = style_fig(fig_eficiencia, legend_bottom=True, tight=True)
        
        # Ajuste adicional para legendas com muitos itens
        fig_eficiencia.update_layout(margin=dict(b=85))  # Margem extra para legenda com máquinas
        fig_eficiencia.update_layout(
            annotations=[
                {
                    "text": "Dados de eficiência não disponíveis",
                    "showarrow": False,
                    "font": {"size": 16}
                }
            ]
        )
    
    # Preparar os detalhes de máquinas para o armazenamento (lazy / limitado)
    detalhes_maquinas = {}
    
    # Verificar se temos os dados necessários ou criar dados baseados no que está disponível
    dados_insuficientes = False
    
    if len(df_filtrado) == 0:
        # Criar dados básicos para visualização quando não há nenhum dado
        print("DataFrame vazio. Criando visualização básica...")
        dados_insuficientes = True
    elif ('item' not in df_filtrado.columns and 'Descrição Item' not in df_filtrado.columns and df_original_global is None):
        # Não temos informações de item nem acesso ao DataFrame original
        print("Não há informações de itens disponíveis. Criando visualização básica...")
        dados_insuficientes = True
    elif 'maquina' not in df_filtrado.columns and not any(col for col in df_filtrado.columns if col not in ['data', 'producao', 'eficiencia', 'hora_inicio', 'datetime_inicio', 'eh_dia_atual', 'primeira_data_hora', 'tipo_maquina', 'turno'] and not col.startswith('parada_') and not col.startswith('tipo_') and not col.startswith('maquinas_tipo_')):
        # Não temos informações de máquinas
        print("Não há informações de máquinas disponíveis. Criando visualização básica...")
        dados_insuficientes = True
    else:
        # Temos os dados necessários, continuar com o processamento normal
        print("Dados suficientes encontrados para detalhes de máquinas.")
    
    if dados_insuficientes:
        # Lista inicial de máquinas (garante existência para uso posterior)
        maquinas_no_df = []
        # Verificar se há alguma coluna de máquina no DataFrame
        maquinas_exemplo = []
        # Primeiro tentar encontrar máquinas no dataframe filtrado
        for col in df_filtrado.columns:
            if col not in ['data', 'producao', 'eficiencia', 'velocidade_id', 'hora_inicio', 'datetime_inicio',
                           'eh_dia_atual', 'primeira_data_hora', 'tipo_maquina', 'turno'] and \
               not col.startswith('parada_') and \
               not col.startswith('tipo_') and \
               not col.startswith('maquinas_tipo_') and \
               not col.startswith('_'):
                maquinas_exemplo.append(col)
        # Se não encontrou máquinas, verificar se temos "Centro Trabalho" no dataframe original
        if not maquinas_exemplo and df_original_global is not None and 'Centro Trabalho' in df_original_global.columns:
            maquinas_exemplo = df_original_global['Centro Trabalho'].dropna().unique().tolist()
            print(f"Encontradas {len(maquinas_exemplo)} máquinas no dataframe original.")
        # Se ainda não encontrou máquinas, usar exemplos padrão
        if not maquinas_exemplo:
            maquinas_exemplo = ["CA01", "CA02", "GR01", "GR02", "MC01", "SP01", "CH01"]
            print("Usando máquinas de exemplo padrão.")
        # Garantir inclusão de CA01-CA08 sempre
        for d in base_defaults:
            if d not in maquinas_exemplo:
                maquinas_exemplo.append(d)
        debug_print(f"[DEBUG] maquinas_exemplo (dados insuficientes) => {maquinas_exemplo}")
        for maquina in maquinas_exemplo:
            # Tentar encontrar dados reais para esta máquina
            producao_total = 0
            if len(df_filtrado) > 0 and maquina in df_filtrado.columns:
                try:
                    producao_total = pd.to_numeric(df_filtrado[maquina], errors='coerce').fillna(0).sum()
                except Exception:
                    producao_total = 0
            if producao_total == 0:
                producao_total = 1000
            itens_detalhes = [{
                'item': f"Produção da máquina {maquina}",
                'producao': producao_total,
                'velocidade': 0,
                'eficiencia': 1.0,
                'descricao_item': f"Dados da máquina {maquina}",
                'centro_trabalho': maquina,
                'tempo_producao': 0
            }]
            detalhes_maquinas[maquina] = itens_detalhes
        # Garantir variável utilizada depois
        maquinas_no_df = maquinas_exemplo
    else:
        # Identificar todas as máquinas no conjunto de dados
        maquinas_no_df = []
        
        # Usar lista global de máquinas originais para garantir presença mesmo sem registros filtrados
        if todas_maquinas_orig:
            maquinas_no_df = list(todas_maquinas_orig)
        
        # Também verificar se temos 'maquina' na coluna
        if 'maquina' in df_filtrado.columns and not todas_maquinas_orig:
            try:
                serie_maqs = df_filtrado['maquina']
                if len(serie_maqs) > 0 and isinstance(serie_maqs.iloc[0], list):
                    explode_vals = []
                    for lst in serie_maqs:
                        if isinstance(lst, list):
                            explode_vals.extend([v for v in lst if isinstance(v,str) and v.strip()])
                    maquinas_no_df.extend(explode_vals)
                else:
                    maquinas_no_df.extend([v for v in serie_maqs.dropna().unique() if isinstance(v,str) and v.strip()])
            except Exception as e:
                debug_print(f"[DEBUG] Falha ao extrair maquinas da coluna 'maquina': {e}")

        # Eliminar duplicatas e fallback
        maquinas_no_df = sorted(list(set(maquinas_no_df)))
        if not maquinas_no_df:
            maquinas_no_df = base_defaults.copy()
            debug_print("[DEBUG] Fallback para base_defaults CA01-CA08 (lista vazia após filtros)")
        
    # =================== NOVA LÓGICA DE DETALHES POR MÁQUINA ===================
    debug_print(f"[DEBUG] Iniciando processamento de detalhes por máquina")
    debug_print(f"[DEBUG] DataFrame filtrado: {len(df_filtrado)} linhas, colunas: {list(df_filtrado.columns)}")
    
    detalhes_maquinas = {}
    
    # Detectar estrutura dos dados
    has_maquina_column = 'maquina' in df_filtrado.columns
    has_centro_trabalho = 'Centro Trabalho' in df_filtrado.columns
    
    # Encontrar colunas que representam máquinas (CA01, CA02, etc.)
    machine_columns = [col for col in df_filtrado.columns if 
                      col.startswith(('CA', 'GR', 'MC', 'SP', 'CH')) and 
                      len(col) <= 5 and col not in ['data', 'Data']]
    
    debug_print(f"[DEBUG] Estrutura detectada:")
    debug_print(f"[DEBUG] - has_maquina_column: {has_maquina_column}")
    debug_print(f"[DEBUG] - has_centro_trabalho: {has_centro_trabalho}")
    debug_print(f"[DEBUG] - machine_columns: {machine_columns[:10]}")  # mostrar apenas 10
    
    # ======================= CENÁRIOS DE ESTRUTURA =======================
    if has_maquina_column and len(df_filtrado) > 0:
        # -------- CENÁRIO 1: já existe coluna 'maquina' --------
        debug_print("[DEBUG] Cenário 1 - coluna 'maquina' detectada")
        if 'item' not in df_filtrado.columns and 'Descrição Item' in df_filtrado.columns:
            df_filtrado['item'] = df_filtrado['Descrição Item']
        if 'item' not in df_filtrado.columns:
            df_filtrado['item'] = 'ITEM_GENERICO'
        if 'turno' not in df_filtrado.columns:
            if 'datetime_inicio' in df_filtrado.columns:
                df_filtrado['turno'] = df_filtrado['datetime_inicio'].apply(calcular_turno)
            else:
                df_filtrado['turno'] = '1'

        # --- Remover linhas de correção isoladas (similar à tratativa de retrabalho) ---
        def remover_correcoes_isoladas(df_base: pd.DataFrame) -> pd.DataFrame:
            """Remove blocos de correção (item diferente que aparece apenas uma vez entre o mesmo item anterior/posterior).
            Regras:
              - Ordena por maquina, turno (se existir), datetime_inicio.
              - Identifica blocos contíguos de mesmo item (run-length).
              - Se um bloco (item X) está cercado (antes/depois) pelo mesmo item Y (Y!=X) e X ocorre apenas neste bloco dentro
                do (maquina, turno) então considera correção e remove todas as suas linhas.
              - Também remove bloco se a soma de produção == 0 e está completamente cercado por mesmo item diferente.
            """
            try:
                req = {'maquina','item','datetime_inicio'}
                if not req.issubset(df_base.columns):
                    return df_base
                df_work = df_base.copy()
                turno_col = 'turno' if 'turno' in df_work.columns else None
                order_cols = ['maquina'] + ([turno_col] if turno_col else []) + ['datetime_inicio']
                df_work = df_work.sort_values(order_cols)
                to_drop_idx = set()
                group_keys = ['maquina'] + ([turno_col] if turno_col else [])
                for gvals, gdf in df_work.groupby(group_keys, sort=False):
                    # Run-length encoding por item
                    itens = gdf['item'].tolist()
                    idxs = gdf.index.tolist()
                    producoes = pd.to_numeric(gdf.get('producao', 0), errors='coerce').fillna(0).tolist()
                    # Construir blocos
                    blocos = []  # cada: (start_i, end_i, item, soma_producao)
                    start = 0
                    for i in range(1, len(itens)+1):
                        if i==len(itens) or itens[i]!=itens[start]:
                            bloco_indices = idxs[start:i]
                            soma_prod = float(sum(producoes[start:i]))
                            blocos.append((start, i-1, itens[start], bloco_indices, soma_prod))
                            start = i
                    # Frequência de cada item no conjunto (quantos blocos)
                    freq_blocos = {}
                    for b in blocos:
                        freq_blocos[b[2]] = freq_blocos.get(b[2],0)+1
                    for bi, bloco in enumerate(blocos):
                        _, _, item_b, indices_b, soma_prod = bloco
                        if bi==0 or bi==len(blocos)-1:
                            continue  # precisa de anterior e posterior
                        item_prev = blocos[bi-1][2]
                        item_next = blocos[bi+1][2]
                        if item_prev == item_next and item_prev != item_b:
                            cond_freq = freq_blocos.get(item_b,0)==1
                            cond_zero = soma_prod == 0.0
                            # se bloco único OU produção zero -> remover
                            if cond_freq or cond_zero:
                                to_drop_idx.update(indices_b)
                    # fim loop blocos
                if to_drop_idx:
                    debug_print(f"[DEBUG] Correções isoladas removidas: {len(to_drop_idx)} linhas")
                    return df_work.drop(index=list(to_drop_idx))
                return df_work
            except Exception as e:
                debug_print(f"[WARN] Falha remover correções isoladas: {e}")
                return df_base

        df_filtrado = remover_correcoes_isoladas(df_filtrado)

        for maquina_nome in maquinas_no_df:
            debug_print(f"[DEBUG] Processando máquina (contiguo somado): {maquina_nome}")
            df_maq = df_filtrado[df_filtrado['maquina'] == maquina_nome].copy()
            if len(df_maq) == 0:
                detalhes_maquinas[maquina_nome] = create_empty_machine_data(maquina_nome)
                continue
            df_maq['__dt_inicio'] = pd.to_datetime(df_maq.get('datetime_inicio'), errors='coerce') if 'datetime_inicio' in df_maq.columns else pd.NaT
            df_maq['__dt_fim'] = pd.to_datetime(df_maq.get('datetime_fim'), errors='coerce') if 'datetime_fim' in df_maq.columns else pd.NaT
            df_maq = df_maq[pd.notna(df_maq['__dt_inicio'])].copy().sort_values('__dt_inicio')
            if len(df_maq)==0:
                detalhes_maquinas[maquina_nome] = create_empty_machine_data(maquina_nome)
                continue
            agg = {}
            current = None
            def fechar_chunk():
                nonlocal current, agg
                if not current:
                    return
                try:
                    turno_end = get_turno_end(current['inicio'])
                    if current['fim']>turno_end:
                        current['fim']=turno_end
                except Exception:
                    pass
                if current['fim']<=current['inicio']:
                    current=None; return
                dur_h = (current['fim']-current['inicio']).total_seconds()/3600.0
                key = (current['turno'], current['item'])
                if key not in agg:
                    agg[key] = {'dur_total':0.0,'producao_total':0.0,'paradas_total':0.0,'efics':[],
                                 'inicio_primeiro':current['inicio'],'fim_ultimo':current['fim'],'roteiro':current['roteiro'],
                                 'paradas_list':[]}
                a=agg[key]
                a['dur_total'] += dur_h
                a['producao_total'] += current['producao']
                a['paradas_total'] += current['paradas']
                a['efics'].extend(current['efics'])
                try:
                    a['paradas_list'].extend(current.get('paradas_lista', []))
                except Exception:
                    pass
                if current['inicio']<a['inicio_primeiro']: a['inicio_primeiro']=current['inicio']
                if current['fim']>a['fim_ultimo']: a['fim_ultimo']=current['fim']
                if (not a['roteiro']) and current['roteiro']: a['roteiro']=current['roteiro']
                current=None
            for _, row in df_maq.iterrows():
                turno = str(row.get('turno','1'))
                item_val = str(row.get('item','ITEM_GENERICO'))
                inicio = row['__dt_inicio']; fim = row['__dt_fim']
                if pd.isna(fim): fim = inicio + pd.Timedelta(minutes=1)
                if (current is None) or (item_val!=current['item']) or (turno!=current['turno']):
                    fechar_chunk()
                    roteiro_val=None
                    for col_r in ['Roteiro','roteiro','faca','Faca']:
                        if col_r in df_maq.columns and pd.notna(row.get(col_r)):
                            roteiro_val=str(row.get(col_r)).strip(); break
                    tempo_parada_row = float(pd.to_numeric(row.get('tempo_parada',0), errors='coerce') or 0)
                    parada_desc = row.get('parada') if pd.notna(row.get('parada')) else None
                    current = {'turno':turno,'item':item_val,'inicio':inicio,'fim':fim,'roteiro':roteiro_val,
                               'producao':float(pd.to_numeric(row.get('producao',0), errors='coerce') or 0),
                               'paradas':tempo_parada_row,
                               'efics':[float(pd.to_numeric(row.get('eficiencia',1.0), errors='coerce') or 1.0)],
                               'paradas_lista':([] if (not parada_desc or tempo_parada_row<=0) else [{'descricao':str(parada_desc).strip(), 'horas':tempo_parada_row}])}
                else:
                    if fim>current['fim']: current['fim']=fim
                    current['producao'] += float(pd.to_numeric(row.get('producao',0), errors='coerce') or 0)
                    tempo_parada_row = float(pd.to_numeric(row.get('tempo_parada',0), errors='coerce') or 0)
                    current['paradas'] += tempo_parada_row
                    parada_desc = row.get('parada') if pd.notna(row.get('parada')) else None
                    if parada_desc and tempo_parada_row>0:
                        try:
                            current['paradas_lista'].append({'descricao':str(parada_desc).strip(), 'horas':tempo_parada_row})
                        except Exception:
                            pass
                    try: current['efics'].append(float(pd.to_numeric(row.get('eficiencia',1.0), errors='coerce') or 1.0))
                    except Exception: pass
                    if (not current['roteiro']) or current['roteiro']=='':
                        for col_r in ['Roteiro','roteiro','faca','Faca']:
                            if col_r in df_maq.columns and pd.notna(row.get(col_r)):
                                current['roteiro']=str(row.get(col_r)).strip(); break
            fechar_chunk()
            itens_maquina=[]
            for (turno_str,item_str),info in agg.items():
                dur_total = info['dur_total']
                if dur_total<=0: continue
                prod_total = info['producao_total']; paradas_total = info['paradas_total']
                tempo_efetivo = max(0.0, dur_total - paradas_total)
                vel = prod_total/tempo_efetivo if tempo_efetivo>0 else 0.0
                efics_list = info['efics']; eficiencia_media = float(sum(efics_list)/len(efics_list)) if efics_list else 1.0
                # Agregar lista de paradas por descrição (somar horas e contar ocorrências)
                paradas_item_lista = []
                try:
                    from collections import defaultdict
                    horas_map = defaultdict(float); occ_map = defaultdict(int)
                    for p in info.get('paradas_list', []):
                        if isinstance(p, dict):
                            desc = p.get('descricao') or p.get('parada') or p.get('desc')
                            h = float(p.get('horas') or p.get('tempo') or 0)
                            if desc:
                                horas_map[desc]+=h; occ_map[desc]+=1
                    for dsc in horas_map:
                        paradas_item_lista.append({'descricao': dsc, 'horas': horas_map[dsc], 'ocorrencias': occ_map[dsc]})
                except Exception:
                    pass
                reg = {'item': item_str,'descricao_item': item_str,'producao': float(prod_total),'velocidade': float(vel),
                       'eficiencia': eficiencia_media,'tempo_producao': float(tempo_efetivo),'tempo_span_h': float(dur_total),
                       'tempo_efetivo_h': float(tempo_efetivo),'turno': turno_str,'paradas_item_h': float(paradas_total),
                       'paradas_turno_h': float(paradas_total),'paradas_item_lista': paradas_item_lista,'centro_trabalho': maquina_nome,
                       'id_velocidade': f"{maquina_nome}-01",'velocidade_padrao': 0,'roteiro': info['roteiro'] if info['roteiro'] else '',
                       'inicio_primeira_aparicao': info['inicio_primeiro'],'fim_ultima_aparicao': info['fim_ultimo'],
                       'inicio_clip': info['inicio_primeiro'],'fim_clip': info['fim_ultimo']}
                try:
                    if 'DICT_VELOCIDADE' in globals():
                        chave1 = maquina_nome.strip(); chave2 = None
                        if info['roteiro']: chave2 = f"{maquina_nome.strip()}-{info['roteiro'].strip()}"
                        velp=None
                        if chave2 and chave2 in DICT_VELOCIDADE: velp=DICT_VELOCIDADE[chave2]
                        elif chave1 in DICT_VELOCIDADE: velp=DICT_VELOCIDADE[chave1]
                        if velp:
                            reg['velocidade_padrao']=float(velp)
                            if reg['tempo_producao']>0: reg['eficiencia']=(reg['producao']/reg['tempo_producao'])/velp
                except Exception: pass
                itens_maquina.append(reg)
            detalhes_maquinas[maquina_nome]= itens_maquina if itens_maquina else create_empty_machine_data(maquina_nome)
        # fim cenário 1
    elif machine_columns and len(df_filtrado) > 0:
        # -------- CENÁRIO 2: colunas de máquinas separadas --------
        debug_print("[DEBUG] Cenário 2 - colunas individuais de máquinas")
        for maquina_col in machine_columns:
            if maquina_col not in maquinas_no_df:
                continue
            mask_producao = pd.to_numeric(df_filtrado[maquina_col], errors='coerce').fillna(0) > 0
            df_maq = df_filtrado[mask_producao].copy()
            if len(df_maq) == 0:
                detalhes_maquinas[maquina_col] = create_empty_machine_data(maquina_col)
                continue
            if 'item' not in df_maq.columns and 'Descrição Item' in df_maq.columns:
                df_maq['item'] = df_maq['Descrição Item']
            if 'item' not in df_maq.columns:
                df_maq['item'] = 'PRODUCAO_GERAL'
            if 'turno' not in df_maq.columns:
                df_maq['turno'] = '1'
            # Para cenário 2, não temos datas por máquina; manter heurística first-last com base em qualquer datetime disponível
            # Se existir uma coluna de tempo comum (ex: 'Data/Hora' ou 'datetime_inicio'), usar para span; caso contrário, usar registros * 0.5h
            dt_col = None
            for cposs in ['datetime_inicio','Data/Hora','data_hora']:
                if cposs in df_maq.columns:
                    dt_col = cposs
                    break
            if dt_col:
                df_maq['__dt_inicio'] = pd.to_datetime(df_maq[dt_col], errors='coerce')
                df_maq = df_maq[pd.notna(df_maq['__dt_inicio'])].copy()
            spans = {}
            for _, row in df_maq.iterrows():
                turno = row.get('turno','1')
                item_val = row.get('item','PRODUCAO_GERAL')
                key = (str(turno), str(item_val))
                if key not in spans:
                    spans[key] = {
                        'inicio': row['__dt_inicio'] if dt_col else None,
                        'fim': row['__dt_inicio'] if dt_col else None,
                        'producao_total': 0.0
                    }
                else:
                    if dt_col and row['__dt_inicio'] and row['__dt_inicio'] > spans[key]['fim']:
                        spans[key]['fim'] = row['__dt_inicio']
                spans[key]['producao_total'] += float(pd.to_numeric(row.get(maquina_col,0), errors='coerce') or 0)
            itens_maquina = []
            for (turno_str,item_str), info in spans.items():
                producao_total = info['producao_total']
                if producao_total <= 0:
                    continue
                if dt_col and info['inicio'] and info['fim']:
                    # assume última leitura representa fim aproximado (adiciona 1 min para dar duração mínima)
                    span_h_total = max(0.0, (info['fim'] - info['inicio'] + pd.Timedelta(minutes=1)).total_seconds()/3600.0)
                else:
                    # fallback: quantidade de registros * 0.5h
                    span_h_total = 0.5  # mínimo
                velocidade = producao_total/span_h_total if span_h_total>0 else 0.0
                itens_maquina.append({
                    'item': item_str,
                    'descricao_item': item_str,
                    'producao': float(producao_total),
                    'velocidade': float(velocidade),
                    'eficiencia': 1.0,
                    'tempo_producao': float(span_h_total),
                    'turno': turno_str,
                    'paradas_item_h': 0.0,
                    'paradas_turno_h': 0.0,
                    'paradas_item_lista': [],
                    'centro_trabalho': maquina_col,
                    'id_velocidade': f"{maquina_col}-01",
                    'velocidade_padrao': 0
                })
            detalhes_maquinas[maquina_col] = itens_maquina if itens_maquina else create_empty_machine_data(maquina_col)

    else:
        # -------- CENÁRIO 3: fallback --------
        debug_print("[DEBUG] Cenário 3 - fallback, criando dados vazios")
        for maquina_nome in maquinas_no_df:
            detalhes_maquinas[maquina_nome] = create_empty_machine_data(maquina_nome)
    # =====================================================================
    
    debug_print(f"[DEBUG] Detalhes criados para {len(detalhes_maquinas)} máquinas")
    # =================== FIM NOVA LÓGICA DE DETALHES ===================
    # Garantir que temos pelo menos uma máquina para o dropdown
    if not detalhes_maquinas:
        debug_print("[DEBUG] detalhes_maquinas vazio após processamento; criando fallback")
        for m in base_defaults:
            detalhes_maquinas[m] = create_empty_machine_data(m)

    # Enriquecer cada registro com velocidade_padrao (se presente no dicionário) e recalcular eficiência
    try:
        if 'DICT_VELOCIDADE' in globals():
            for maq, registros in detalhes_maquinas.items():
                for reg in registros:
                    if not isinstance(reg, dict):
                        continue
                    item_reg = (reg.get('item') or '').strip()
                    roteiro_reg = (reg.get('roteiro') or '').strip()
                    # Chaves possíveis: MAQ-ROTEIRO, MAQ ROTEIRO, MAQ-ITEM, MAQ ITEM, MAQ
                    candidatos = []
                    if roteiro_reg:
                        candidatos += [f"{maq}-{roteiro_reg}", f"{maq} {roteiro_reg}"]
                    if item_reg:
                        candidatos += [f"{maq}-{item_reg}", f"{maq} {item_reg}"]
                    candidatos.append(maq)
                    vel_padrao = None
                    for ch in candidatos:
                        if ch in DICT_VELOCIDADE:
                            vel_padrao = DICT_VELOCIDADE[ch]
                            break
                    if vel_padrao is not None:
                        reg['velocidade_padrao'] = float(vel_padrao)
                        # Recalcular eficiência segmentar se possível
                        tempo_eff = reg.get('tempo_producao', 0) or 0
                        prod = reg.get('producao', 0) or 0
                        if tempo_eff > 0 and vel_padrao > 0:
                            velocidade_media = prod / tempo_eff
                            reg['eficiencia'] = (velocidade_media / vel_padrao)
    except Exception as _e:
        debug_print(f"[DEBUG] Falha ao atribuir velocidades padrão: {_e}")

    # Opções para dropdown (ordenadas)
    opcoes_dropdown = [
        {"label": maquina, "value": maquina} for maquina in sorted(detalhes_maquinas.keys())
    ]
    if not opcoes_dropdown:
        # Fallback mínimo
        opcoes_dropdown = [{"label": "SEM DADOS", "value": "SEM DADOS"}]

    # Formatação da produção total
    producao_total_int = int(producao_total) if not pd.isna(producao_total) else 0
    producao_total_str = f"{producao_total_int:,}".replace(",", ".")

    if DEBUG:
        debug_print(f"[DEBUG] Produção Total exibida: {producao_total_str} (valor original: {producao_total})")
        debug_print(f"[DEBUG] Máquinas finais para dropdown: { [o['value'] for o in opcoes_dropdown] }")

    # Métrica de performance
    t_elapsed = (time.perf_counter() - t0) * 1000
    print(f"[PERF] update_dashboard em {t_elapsed:.1f} ms (linhas={len(df_filtrado)})")

    # ---------- Categorização detalhada das paradas ----------
    if 'parada' in df_filtrado.columns and 'tempo_parada' in df_filtrado.columns:
        # Incluir coluna 'item' para permitir reconstruir tabela de acertos (1 por item/turno)
        _cols_base_parada = ['maquina','parada','tempo_parada','datetime_inicio']
        if 'item' in df_filtrado.columns:
            _cols_base_parada.append('item')
        try:
            df_paradas_base = df_filtrado[pd.to_numeric(df_filtrado['tempo_parada'], errors='coerce').fillna(0) > 0][_cols_base_parada].copy()
        except Exception:
            # Fallback sem item caso algo falhe
            df_paradas_base = df_filtrado[pd.to_numeric(df_filtrado['tempo_parada'], errors='coerce').fillna(0) > 0][['maquina','parada','tempo_parada','datetime_inicio']].copy()
        df_paradas_base['parada'] = df_paradas_base['parada'].astype(str).str.upper().str.strip()
        manut_pats = ['MANUT','PREVENT','CORRET','LUBR','AJUSTE MEC','AJUSTE ELETR','TROCA ROL','TROCA CORREIA']
        ajuste_pats = ['AJUSTE']
        acerto_pats = ['ACERTO','SETUP','TROCA DE ÍTEM','TROCA DE ITEM','TROCA ITEM','TROCA ÍTEM']
        aguard_pats = ['AGUARD','ESPERA','FALTA','SEM MATER','AGEND']
        def _cat(desc:str):
            if any(p in desc for p in manut_pats):
                return 'MANUTENCAO'
            if any(p in desc for p in ajuste_pats):
                return 'AJUSTE'
            if any(p in desc for p in acerto_pats):
                return 'ACERTO'
            if any(p in desc for p in aguard_pats):
                return 'AGUARDANDO'
            return 'OUTROS'
        df_paradas_base['categoria_parada'] = df_paradas_base['parada'].apply(_cat)
    else:
        df_paradas_base = pd.DataFrame(columns=['maquina','parada','tempo_parada','datetime_inicio','categoria_parada'])

    def _cat_filter(cat_key):
        mapa = {
            'MANUT':'MANUTENCAO',
            'AJUSTE':'AJUSTE',
            'ACERTO':'ACERTO',
            'AGUARD':'AGUARDANDO',
            'OUTROS':'OUTROS'
        }
        alvo = mapa.get(cat_key, cat_key)
        return df_paradas_base[df_paradas_base['categoria_parada']==alvo]

    figs_cat = {}
    cat_def = [ ('MANUT','Paradas de Manutenção','grafico-parada-manutencao'),
                ('AGUARD','Paradas Aguardando','grafico-parada-aguardando'),
                ('AJUSTE','Paradas de Ajuste','grafico-parada-ajuste'),
                ('ACERTO','Paradas de Acerto','grafico-parada-acerto'),
                ('OUTROS','Paradas - Outros','grafico-parada-outros') ]
    for key,label,idfig in cat_def:
        dfx = _cat_filter(key)
        if len(dfx)>0:
            agg = dfx.groupby('maquina')['tempo_parada'].sum().reset_index()
            top_desc = dfx.groupby(['maquina','parada'])['tempo_parada'].sum().reset_index()
            top_desc = top_desc.sort_values(['maquina','tempo_parada'], ascending=[True,False])
            hover_map = {}
            for maq, sub in top_desc.groupby('maquina'):
                linhas = []
                for _, r in sub.head(3).iterrows():
                    linhas.append(f"{r['parada'][:30]}: {r['tempo_parada']:.1f}h")
                hover_map[maq] = '<br>'.join(linhas)
            agg['hover'] = agg['maquina'].map(hover_map)
            if key in ('MANUT','AGUARD'):
                # Pie chart formato donut
                total_h = agg['tempo_parada'].sum()
                figc = px.pie(agg, names='maquina', values='tempo_parada', title=label, hover_data={'hover':True})
                figc.update_traces(hovertemplate='<b>%{label}</b><br>%{percent:.1%} | %{value:.2f}h<br>%{customdata[0]}', texttemplate='%{label}<br>%{percent:.0%}', textposition='inside')
                figc.update_layout(legend_title_text='Máquina')
                figc = style_pie(figc, total_label=f"{total_h:.1f}h")  # Aplicar cores azul-roxo
            elif key == 'AJUSTE':
                # Barra horizontal ordenada (melhor leitura com muitas máquinas) mantendo descrições (top3) em hover
                agg_sorted = agg.sort_values('tempo_parada', ascending=True)
                figc = px.bar(agg_sorted, y='maquina', x='tempo_parada', orientation='h', title=label,
                               text='tempo_parada', color='maquina',
                               hover_data={'hover':True})
                figc.update_traces(texttemplate='%{text:.1f}h',
                                   hovertemplate='<b>%{y}</b><br>Horas: %{x:.2f}h<br>%{customdata[0]}')
                figc.update_layout(xaxis_title='Horas', yaxis_title='Máquina')
                figc = style_bar(figc)  # Aplicar cores purple
                figc = style_fig(figc, y_hours=True, legend_bottom=False, tight=True)
            else:
                figc = px.bar(
                    agg,
                    x='maquina',
                    y='tempo_parada',
                    title=label,
                    text='tempo_parada',
                    hover_data={'hover': True, 'tempo_parada': True, 'maquina': True},
                    color='maquina'
                )
                figc.update_traces(
                    texttemplate='%{text:.1f}h',
                    textposition='outside',
                    hovertemplate='<b>%{x}</b><br>Horas: %{y:.2f}h<br>%{customdata[0]}'
                )
                figc.update_layout(yaxis_title='Horas', xaxis_title='Máquina')
                figc = style_bar(figc)  # Aplicar cores purple
                figc = style_fig(figc, y_hours=True, legend_bottom=False, tight=True)
        else:
            if key in ('MANUT','AGUARD'):
                figc = style_pie(px.pie(title=label), total_label='0.0h')
            elif key == 'AJUSTE':
                figc = px.bar(title=label)
                figc = style_bar(figc)  # Aplicar cores purple
                figc = style_fig(figc, y_hours=True)
            else:
                figc = px.bar(title=label)
                figc = style_bar(figc)  # Aplicar cores purple
                figc = style_fig(figc)
        figs_cat[idfig] = figc

    # Tabela de acertos
    acertos_df = _cat_filter('ACERTO')
    tabela_acertos_comp = html.Div('Sem acertos')
    if len(acertos_df) > 0 and {'item', 'tempo_parada', 'datetime_inicio'}.issubset(acertos_df.columns):
        acertos_df = acertos_df.copy()

        # Robust normalization of tempo_parada into hours (tempo_h).
        # Handles numeric values (minutes or hours), comma decimals, and time strings like 'HH:MM:SS' or 'MM:SS'.
        # Detect dataset minute-mode: majority of numeric values are small integers (<60)
        numeric_vals = []
        for v in acertos_df['tempo_parada']:
            if isinstance(v, (int, float, np.integer, np.floating)) and not pd.isna(v):
                numeric_vals.append(float(v))
            elif isinstance(v, str) and ':' not in v:
                try:
                    numeric_vals.append(float(v.strip().replace(',', '.')))
                except Exception:
                    pass
        minute_mode = False
        if numeric_vals:
            arr = pd.Series(numeric_vals)
            # Ratios
            is_int_ratio = (arr.apply(lambda x: abs(x - round(x)) < 1e-6)).mean()
            lt60_ratio = (arr < 60).mean()
            # If almost all are integers and less than 60 and there are no obviously-hour-format strings (with colon), assume minutes
            if is_int_ratio > 0.85 and lt60_ratio > 0.85:
                minute_mode = True
        def parse_tempo_to_hours(v):
            # Return hours
            # 1) Missing
            try:
                if pd.isna(v):
                    return 0.0
            except Exception:
                pass
            # 2) Timedelta
            if isinstance(v, pd.Timedelta):
                return v.total_seconds()/3600.0
            # 3) String
            if isinstance(v, str):
                s = v.strip().replace(',', '.')
                if not s:
                    return 0.0
                if ':' in s:  # time-like
                    parts = s.split(':')
                    try:
                        parts_int = [int(p) for p in parts]
                        if len(parts_int) == 3:
                            h, m, sec = parts_int
                        elif len(parts_int) == 2:
                            h = 0; m, sec = parts_int
                        else:
                            return float(s)
                        return (h*3600 + m*60 + sec)/3600.0
                    except Exception:
                        pass
                try:
                    num = float(s)
                except Exception:
                    return 0.0
                # If in minute_mode treat EVERY numeric (even <50) as minutes; else keep prior >50 heuristic
                if minute_mode:
                    return num/60.0
                return num/60.0 if num > 50 else num
            # 4) Numeric
            if isinstance(v, (int, float, np.integer, np.floating)):
                num = float(v)
                if minute_mode:
                    return num/60.0
                return num/60.0 if num > 50 else num
            # 5) Fallback conversion
            try:
                num = float(v)
                if minute_mode:
                    return num/60.0
                return num/60.0 if num > 50 else num
            except Exception:
                return 0.0

        acertos_df['tempo_h'] = acertos_df['tempo_parada'].apply(parse_tempo_to_hours)
        origem_em_minutos = minute_mode

        # Debug: mostrar algumas linhas que entrarão no cálculo de acertos
        if DEBUG:
            try:
                debug_print(f"[ACERTOS] linhas usadas: {len(acertos_df)}; origem_em_minutos={origem_em_minutos}")
                debug_print(acertos_df[['maquina','item','parada','tempo_parada','tempo_h','datetime_inicio']].head(10))
            except Exception:
                pass

        # Turno calculado a partir do datetime_inicio quando disponível
        try:
            if 'datetime_inicio' in acertos_df.columns:
                acertos_df['turno_calc'] = acertos_df['datetime_inicio'].apply(calcular_turno)
            elif 'turno' in acertos_df.columns:
                acertos_df['turno_calc'] = acertos_df['turno']
            else:
                acertos_df['turno_calc'] = '1'
        except Exception:
            acertos_df['turno_calc'] = acertos_df.get('turno', '1')

        # Garantir colunas 'item' e 'maquina' para agrupamento
        if 'item' not in acertos_df.columns:
            achou_item = False
            for c in acertos_df.columns:
                if 'item' in c.lower():
                    try:
                        acertos_df['item'] = acertos_df[c].astype(str)
                        achou_item = True
                        break
                    except Exception:
                        continue
            if not achou_item:
                acertos_df['item'] = 'N/D'
        if 'maquina' not in acertos_df.columns:
            # tentar mapear a partir de possíveis nomes
            mapped = False
            for alt in ['Centro Trabalho','centro_trabalho','centro trabalho','machine','máquina']:
                for col in acertos_df.columns:
                    if col.lower().strip() == alt.lower():
                        try:
                            acertos_df['maquina'] = acertos_df[col].astype(str)
                            mapped = True
                            break
                        except Exception:
                            continue
                if mapped:
                    break
            if not mapped:
                acertos_df['maquina'] = 'N/D'

        # Agrupamentos usando a coluna normalizada 'tempo_h' (horas)
        try:
            if acertos_dim_mode == 'maquina':
                gturno = acertos_df.groupby(['maquina','turno_calc'])['tempo_h'].sum().reset_index()
                ordem_totais = (gturno.groupby('maquina')['tempo_h'].sum().sort_values(ascending=False))
                heat_pivot = gturno.pivot_table(index='maquina', columns='turno_calc', values='tempo_h', aggfunc='sum').fillna(0)
                cols_turnos = [c for c in ['1','2','3'] if c in heat_pivot.columns]
                heat_pivot = heat_pivot[cols_turnos]
                heat_pivot = heat_pivot.loc[ordem_totais.index.intersection(heat_pivot.index)]
                z_vals = heat_pivot.values
                x_vals = [f"Turno {c}" for c in heat_pivot.columns]
                y_vals = heat_pivot.index.tolist()
                text_vals = [[f"{v:.2f}h" for v in row] for row in z_vals]
                fig_acertos_heatmap = go.Figure(data=go.Heatmap(
                    z=z_vals, x=x_vals, y=y_vals, colorbar=dict(title='Horas'),
                    text=text_vals, texttemplate="%{text}",
                    hovertemplate='Máquina: %{y}<br>%{x}<br>Horas: %{z:.2f}<extra></extra>'
                ))
                fig_acertos_heatmap.update_layout(title='Acertos por Máquina e Turno (Heatmap Horas)', xaxis_title='Turno', yaxis_title='Máquina', margin=dict(l=120,r=20,t=60,b=90))  # Margem inferior maior
                fig_acertos_heatmap = style_heatmap(fig_acertos_heatmap)  # Aplicar cores purple
            else:
                gturno = acertos_df.groupby(['item','maquina','turno_calc'])['tempo_h'].sum().reset_index()
                gturno['item_maquina'] = gturno['item'].astype(str).str.strip() + ' | ' + gturno['maquina'].astype(str).str.strip()
                ordem_totais = (gturno.groupby('item_maquina')['tempo_h'].sum().sort_values(ascending=False))
                top_lista = ordem_totais.head(60).index.tolist()
                gturno_top = gturno[gturno['item_maquina'].isin(top_lista)].copy()
                heat_pivot = gturno_top.pivot_table(index='item_maquina', columns='turno_calc', values='tempo_h', aggfunc='sum').fillna(0)
                cols_turnos = [c for c in ['1','2','3'] if c in heat_pivot.columns]
                heat_pivot = heat_pivot[cols_turnos]
                heat_pivot = heat_pivot.loc[ordem_totais.index.intersection(heat_pivot.index)]
                z_vals = heat_pivot.values
                x_vals = [f"Turno {c}" for c in heat_pivot.columns]
                y_vals = heat_pivot.index.tolist()
                text_vals = [[f"{v:.2f}h" for v in row] for row in z_vals]
                fig_acertos_heatmap = go.Figure(data=go.Heatmap(
                    z=z_vals, x=x_vals, y=y_vals, colorbar=dict(title='Horas'),
                    text=text_vals, texttemplate="%{text}",
                    hovertemplate='Item|Máquina: %{y}<br>%{x}<br>Horas: %{z:.2f}<extra></extra>'
                ))
                fig_acertos_heatmap.update_layout(title='Acertos por Item, Máquina e Turno (Heatmap Horas)', xaxis_title='Turno', yaxis_title='Item | Máquina', margin=dict(l=250,r=20,t=60,b=90))  # Margem esquerda maior para mostrar nomes completos + margem inferior maior
                fig_acertos_heatmap = style_heatmap(fig_acertos_heatmap)  # Aplicar cores purple
            fig_acertos_heatmap = style_fig(fig_acertos_heatmap, y_hours=False)
            tabela_acertos_comp = html.Div([
                dcc.Graph(figure=fig_acertos_heatmap, style={'height': f'{max(600, 25*len(y_vals) + 100)}px'})  # Altura maior + espaço para legenda
            ])
        except Exception as _e_ac:
            debug_print(f"[ACERTOS] Falha ao agrupar: {_e_ac}")
            tabela_acertos_comp = html.Div("Erro ao agrupar acertos")

    # Garantir cálculo atualizado de total de paradas se coluna tempo_parada existir
    if 'tempo_parada' in df_filtrado.columns:
        try:
            total_paradas = float(pd.to_numeric(df_filtrado['tempo_parada'], errors='coerce').fillna(0).sum())
        except Exception:
            pass

    # Recalcular eficiência média agora que detalhes_maquinas está completo (ponderada por tempo)
    try:
        soma_num = 0.0; soma_den = 0.0
        for regs in detalhes_maquinas.values():
            for r in regs:
                t = float(r.get('tempo_producao',0) or 0)
                e = r.get('eficiencia',0) or 0
                # Filtrar valores não numéricos e outliers extremos (>10 = 1000%)
                try:
                    e = float(e)
                except Exception:
                    continue
                if t>0 and 0 < e < 10:
                    soma_num += e * t
                    soma_den += t
        if soma_den>0:
            eficiencia_media = soma_num / soma_den
            eficiencia_media_str = f"{eficiencia_media*100:.1f}%"
    except Exception as e:
        print(f"Erro ao recalcular eficiência média (detalhes): {e}")

    # Gráficos de acerto por máquina e turno
    if len(acertos_df)>0 and 'tempo_parada' in acertos_df.columns:
        try:
            fig_acerto_maquina = px.bar(acertos_df.groupby('maquina')['tempo_parada'].sum().reset_index(),
                                        x='maquina', y='tempo_parada', title='Acertos por Máquina', text='tempo_parada', color='maquina')
            fig_acerto_maquina.update_traces(texttemplate='%{text:.1f}h', textposition='outside')
            fig_acerto_maquina = style_bar(fig_acerto_maquina)  # Aplicar cores purple
            fig_acerto_maquina = style_fig(fig_acerto_maquina, y_hours=True)
        except Exception:
            fig_acerto_maquina = px.bar(title='Acertos por Máquina')
            fig_acerto_maquina = style_bar(fig_acerto_maquina)  # Aplicar cores purple
        try:
            if 'datetime_inicio' in acertos_df.columns:
                acertos_df['turno_calc'] = acertos_df['datetime_inicio'].apply(calcular_turno)
            if 'turno_calc' not in acertos_df.columns and 'turno' in acertos_df.columns:
                acertos_df['turno_calc'] = acertos_df['turno']
            agrup_turno = acertos_df.groupby('turno_calc')['tempo_parada'].sum().reset_index()
            total_turno = agrup_turno['tempo_parada'].sum() if len(agrup_turno)>0 else 0
            fig_acerto_turno = px.pie(agrup_turno, values='tempo_parada', names='turno_calc',
                                      title='Acertos por Turno', color='turno_calc')
            fig_acerto_turno = style_pie(fig_acerto_turno, total_label=f"{total_turno:.1f}h")  # Aplicar cores azul-roxo
        except Exception:
            fig_acerto_turno = style_pie(px.pie(title='Acertos por Turno'), total_label='0.0h')
    else:
        fig_acerto_maquina = px.bar(title='Acertos por Máquina')
        fig_acerto_maquina = style_bar(fig_acerto_maquina)  # Aplicar cores purple
        fig_acerto_maquina = style_fig(fig_acerto_maquina, y_hours=True)
        fig_acerto_turno = style_pie(px.pie(title='Acertos por Turno'), total_label='0.0h')

    # Detalhe das Paradas por Descrição - voltar para tabela completa
    detalhe_paradas_comp = html.Div("Sem paradas")
    try:
        if len(df_paradas_base) > 0:
            agg_desc = df_paradas_base.groupby('parada').agg(
                tempo_parada=('tempo_parada','sum'),
                qtde=('parada','count')
            ).reset_index()
            max_val = agg_desc['tempo_parada'].max() if len(agg_desc) > 0 else 0
            origem_em_minutos = max_val > 50
            if origem_em_minutos:
                agg_desc['tempo_horas'] = agg_desc['tempo_parada'] / 60.0
            else:
                agg_desc['tempo_horas'] = agg_desc['tempo_parada']
            agg_desc = agg_desc.sort_values('tempo_horas', ascending=False)
            tabela_full = agg_desc.rename(columns={
                'parada':'Descrição Parada',
                'qtde':'Qtde',
                'tempo_horas':'Tempo (h)'
            })[['Descrição Parada','Qtde','Tempo (h)']]
            tabela_full['Tempo (h)'] = tabela_full['Tempo (h)'].apply(lambda v: f"{v:.2f}")
            detalhe_paradas_comp = dbc.Table.from_dataframe(tabela_full, striped=True, bordered=True, hover=True, size='sm')
    except Exception as e:
        detalhe_paradas_comp = html.Div(f"Erro ao gerar detalhe de paradas: {e}")

    # ---- Gráfico de densidade de paradas (quantidade) Máquina x Turno ----
    try:
        if len(df_paradas_base)>0:
            base_dens = df_paradas_base.copy()
            # Derivar turno da parada pelo datetime_inicio (fallback turno calculado)
            if 'datetime_inicio' in base_dens.columns:
                base_dens['turno_plot'] = base_dens['datetime_inicio'].apply(calcular_turno)
            else:
                base_dens['turno_plot'] = '1'
            contagem = base_dens.groupby(['maquina','turno_plot'])['parada'].count().reset_index().rename(columns={'parada':'qtde'})
            # Pivot para heatmap
            pivot = contagem.pivot(index='maquina', columns='turno_plot', values='qtde').fillna(0)
            pivot = pivot.reindex(sorted(pivot.index), axis=0)
            pivot = pivot[['1','2','3']] if all(c in pivot.columns for c in ['1','2','3']) else pivot
            fig_parada_densidade = go.Figure(data=go.Heatmap(
                z=pivot.values,
                x=pivot.columns,
                y=pivot.index,
                colorbar=dict(title='Qtd'),
                hovertemplate='Máquina: %{y}<br>Turno: %{x}<br>Paradas: %{z}<extra></extra>'
            ))
            fig_parada_densidade.update_layout(title='Densidade de Paradas (Quantidade)', xaxis_title='Turno', yaxis_title='Máquina',
                                               margin=dict(l=60,r=20,t=60,b=90))  # Margem inferior maior para legendas
            fig_parada_densidade = style_heatmap(fig_parada_densidade)  # Aplicar cores purple
            fig_parada_densidade = style_fig(fig_parada_densidade, legend_bottom=False, tight=True)

            # Mapa de frequência (bolhas) Máquina x Descrição segmentado por Turno (facetas)
            base_bolha = base_dens.copy()
            top_desc = base_bolha['parada'].value_counts().head(12).index.tolist()
            base_bolha = base_bolha[base_bolha['parada'].isin(top_desc)]
            if len(base_bolha) > 0:
                # Truncar descrições muito longas para melhor visualização
                base_bolha['parada_display'] = base_bolha['parada'].apply(lambda x: x[:40] + '...' if len(str(x)) > 40 else str(x))
                agr_bolha = base_bolha.groupby(['maquina','turno_plot','parada','parada_display'])['parada'].count().rename('qtde').reset_index()
                
                # Ordenar descrições por total descendente para legibilidade
                ordem_desc = agr_bolha.groupby('parada_display')['qtde'].sum().sort_values(ascending=False).index.tolist()
                
                fig_parada_densidade3d = px.scatter(
                    agr_bolha,
                    x='maquina', y='parada_display', size='qtde', color='qtde',
                    facet_col='turno_plot', facet_col_spacing=0.06,
                    title='Frequência de Paradas (Máquina x Descrição) por Turno',
                    size_max=40,
                    hover_data={'parada': True, 'parada_display': False}  # Mostrar descrição completa no hover
                )
                fig_parada_densidade3d.update_layout(
                    margin=dict(l=300, r=30, t=90, b=80),  # Margem esquerda muito maior para descrições longas
                    coloraxis_colorbar=dict(title='Qtd', x=1.02),  # Mover colorbar para direita
                    height=max(500, len(top_desc) * 35),  # Altura ainda maior para melhor espaçamento
                    font=dict(size=10),  # Fonte menor para otimizar espaço
                    showlegend=False  # Remover legenda para dar mais espaço
                )
                # Aplicar cores purple
                fig_parada_densidade3d.update_traces(
                    marker=dict(
                        colorscale=[[0, '#f3e8ff'], [0.2, '#e9d5ff'], [0.4, '#d8b4fe'], 
                                   [0.6, '#8b5cf6'], [0.8, '#7c3aed'], [1, '#5b21b6']]
                    )
                )
                # Configurar eixos para melhor visualização dos textos
                fig_parada_densidade3d.update_yaxes(
                    categoryorder='array', 
                    categoryarray=ordem_desc,
                    tickfont=dict(size=9),  # Fonte menor nos ticks do eixo Y
                    tickmode='linear',
                    automargin=True  # Ajuste automático da margem
                )
                fig_parada_densidade3d.update_xaxes(
                    tickangle=45,  # Rotacionar nomes das máquinas para economizar espaço
                    tickfont=dict(size=9)
                )
                # Hover customizado com descrição completa
                for tr in fig_parada_densidade3d.data:
                    tr.hovertemplate = 'Máquina: %{x}<br>Descrição: %{customdata[0]}<br>Qtd: %{marker.size}<extra></extra>'
                fig_parada_densidade3d = style_fig(fig_parada_densidade3d, legend_bottom=False, tight=True)
            else:
                fig_parada_densidade3d = style_fig(px.bar(title='Frequência Paradas (sem dados)'))
        else:
            fig_parada_densidade = style_fig(px.bar(title='Densidade de Paradas'))
            # ranking removido
            fig_parada_densidade3d = style_fig(px.bar(title='Frequência Paradas'))
    except Exception as e:
        import traceback
        error_msg = f"Erro densidade: {str(e)}\n{traceback.format_exc()}"
        print(f"[ERROR] {error_msg}")
        fig_parada_densidade = style_fig(px.bar(title=f'Erro densidade: {str(e)}'))
        fig_parada_densidade3d = style_fig(px.bar(title=f'Erro frequência: {str(e)}'))

    return (
        producao_total_str,                 # producao-total
        f"{total_paradas:.1f}",            # total-paradas
        maquina_produtiva,                 # maquina-produtiva
        str(total_acertos),                # total-acertos
        turno_atual,                       # turno-atual
        eficiencia_media_str,              # eficiencia-media
        fig_producao_maquina,              # grafico-producao-maquina
        figs_cat['grafico-parada-manutencao'], # grafico-parada-manutencao
        figs_cat['grafico-parada-ajuste'],     # grafico-parada-ajuste
        fig_parada_densidade,              # grafico-parada-densidade
        fig_parada_densidade3d,            # grafico-parada-densidade3d
        figs_cat['grafico-parada-acerto'],     # grafico-parada-acerto
        figs_cat['grafico-parada-aguardando'], # grafico-parada-aguardando
        figs_cat['grafico-parada-outros'],     # grafico-parada-outros
        fig_acerto_maquina,                # grafico-acerto-maquina
        fig_acerto_turno,                  # grafico-acerto-turno
        fig_velocidade,                    # grafico-velocidade
        fig_tempo_parada_maquina,          # grafico-tempo-parada-maquina
        tabela_acertos_comp,               # tabela-acertos
        detalhe_paradas_comp,              # tabela-detalhe-paradas
        detalhes_maquinas,                 # store-machine-details
        opcoes_dropdown                    # maquina-detalhe-dropdown
    )


def create_empty_machine_data(maquina_nome):
    """Cria dados vazios padronizados para uma máquina."""
    return [{
        'item': 'Sem dados no período',
        'descricao_item': 'Sem registros para esta máquina no intervalo selecionado',
        'producao': 0,
        'velocidade': 0,
        'eficiencia': 0,
        'tempo_producao': 0,
        'turno': 'N/D',
        'paradas_item_h': 0,
        'paradas_turno_h': 0,
        'paradas_item_lista': [],
        'centro_trabalho': maquina_nome,
        'id_velocidade': f"{maquina_nome}-01",
        'velocidade_padrao': 0
    }]

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

# Callback para inicializar o valor do dropdown com a primeira máquina
@app.callback(
    Output("maquina-detalhe-dropdown", "value"),
    [Input("maquina-detalhe-dropdown", "options")],
    [State("maquina-detalhe-dropdown", "value")]
)
def initialize_dropdown_value(options, current_value):
    # Se já houver um valor selecionado e ele estiver nas opções, manter
    if current_value and any(opt["value"] == current_value for opt in options):
        return current_value
    
    # Caso contrário, selecionar a primeira opção se disponível
    if options and len(options) > 0:
        return options[0]["value"]
    
    # Se não houver opções, retornar None
    return None

# Callback para gerar o gráfico de detalhes das máquinas
@app.callback(
    Output("grafico-detalhes-maquina", "figure"),
    [Input("maquina-detalhe-dropdown", "value"),
     Input("ordenar-detalhe-radio", "value"),
     Input("store-machine-details", "data")]
)
def update_machine_details_graph(maquina_selecionada, ordenar_por, machine_details):
    try:
        # Verificar se temos dados e uma máquina selecionada
        if not machine_details or not maquina_selecionada or maquina_selecionada not in machine_details:
            # Retornar gráfico vazio com instruções claras
            fig = px.bar(title="Selecione uma máquina para ver detalhes")
            fig.update_layout(
                annotations=[{"text": "Selecione uma máquina para ver detalhes", "showarrow": False, "font": {"size": 16}}],
                xaxis={"visible": False},
                yaxis={"visible": False}
            )
            fig = style_fig(fig)
            return fig
        
        # Obter dados da máquina selecionada
        detalhes = machine_details[maquina_selecionada]
        
        # Verificar se temos detalhes
        if not detalhes or len(detalhes) == 0:
            fig = px.bar(title=f"Sem dados disponíveis para a máquina {maquina_selecionada}")
            fig.update_layout(
                annotations=[{"text": f"Sem dados disponíveis para a máquina {maquina_selecionada}", "showarrow": False, "font": {"size": 16}}],
                xaxis={"visible": False},
                yaxis={"visible": False}
            )
            fig = style_fig(fig)
            return fig
            
        # Converter para DataFrame para manipulação
        df_detalhes = pd.DataFrame(detalhes)
        # Filtro de sanidade: garantir que só itens dessa máquina estejam presentes (caso contaminação)
        if 'id_velocidade' in df_detalhes.columns:
            df_detalhes = df_detalhes[df_detalhes['id_velocidade'].astype(str).str.startswith(str(maquina_selecionada)) | (df_detalhes['id_velocidade']=='' )]
        
        # Garantir que as colunas necessárias existam
        for col in ['item', 'producao', 'velocidade', 'eficiencia']:
            if col not in df_detalhes.columns:
                df_detalhes[col] = 0
                
        # Limpar valores NaN
        df_detalhes = df_detalhes.fillna(0)
        
        # Ordenar conforme solicitado
        if ordenar_por == "producao":
            df_detalhes = df_detalhes.sort_values(by="producao", ascending=False)
        elif ordenar_por == "velocidade":
            df_detalhes = df_detalhes.sort_values(by="velocidade", ascending=False)
        elif ordenar_por == "item":
            df_detalhes = df_detalhes.sort_values(by="item")
        
        # Limitar a 15 itens para melhor visualização
        df_detalhes = df_detalhes.head(15)

        # Extrair listas de paradas por item e por turno se disponíveis
        # Espera-se que 'paradas_item_lista' contenha lista de dicts com chaves: descricao/desc/parada, horas/tempo/tempo_h/tempo_horas ou similar
        def extrair_paradas(res):
            saida = []
            if isinstance(res, list):
                for p in res:
                    if isinstance(p, dict):
                        desc = p.get('parada') or p.get('descricao') or p.get('desc') or ''
                        tempo = p.get('tempo_h') or p.get('tempo_horas') or p.get('horas') or p.get('tempo') or 0
                        try:
                            tempo = float(tempo)
                        except Exception:
                            tempo = 0
                        if desc:
                            saida.append((desc, tempo))
            return saida

        df_detalhes['paradas_extra'] = df_detalhes.get('paradas_item_lista', []).apply(extrair_paradas) if 'paradas_item_lista' in df_detalhes.columns else [[] for _ in range(len(df_detalhes))]
        # Montar tooltip de paradas por item (top 5 por tempo)
        tooltips_paradas = {}
        for _, row in df_detalhes.iterrows():
            lst = sorted(row['paradas_extra'], key=lambda x: x[1], reverse=True)[:5]
            if lst:
                txt = '<br>'.join([f"{d}: {t:.2f}h" for d,t in lst])
            else:
                txt = 'Sem paradas registradas'
            tooltips_paradas[row['item']] = txt
        
        # Converter dados para o formato necessário para o gráfico de barras agrupadas
        df_melt = df_detalhes.melt(
            id_vars=['item', 'eficiencia'],
            value_vars=['producao', 'velocidade'],
            var_name='Métrica',
            value_name='Valor'
        )
        
        # Criar o gráfico
        fig = px.bar(
            df_melt,
            x="item",
            y="Valor",
            color="Métrica",
            barmode="group",
            title=f"Detalhes da Máquina: {maquina_selecionada}",
            labels={
                "item": "Item",
                "Valor": "Produção / Velocidade",
                "Métrica": "Métrica"
            },
            color_discrete_map={'producao': '#8b5cf6','velocidade': '#7c3aed'}  # Purple colors
        )

        # Custom hover com paradas (injeção)
        try:
            for tr in fig.data:
                if tr.type == 'bar':
                    new_hover = []
                    for x in tr.x:
                        info_paradas = tooltips_paradas.get(x, 'Sem paradas')
                        new_hover.append(f"<b>{x}</b><br>Métrica: {tr.name}<br>Valor: %{tr.y}<br><br><b>Paradas (Top)</b><br>{info_paradas}")
                    tr.hovertemplate = '%{customdata}'
                    tr.customdata = np.array(new_hover)
        except Exception:
            pass
        
        # Adicionar linha de eficiência como marcadores
        fig.add_trace(
            go.Scatter(
                x=df_detalhes["item"],
                y=df_detalhes["eficiencia"] * 100,  # Converter para percentual
                mode="markers+lines",
                name="Eficiência (%)",
                yaxis="y2",
                line=dict(color="red", width=2),
                marker=dict(size=8, symbol="diamond")
            )
        )
        
        # Configurar eixo Y secundário para eficiência
        fig.update_layout(
            yaxis2=dict(
                title="Eficiência (%)",
                overlaying="y",
                side="right",
                range=[0, 200],  # Limite máximo de 200%
                tickformat=".0f"
            ),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            xaxis=dict(title="Item", tickangle=45),
            yaxis=dict(title="Produção / Velocidade"),
            height=600  # Altura maior para melhor visualização
        )
        
        # Adicionar linha de referência para eficiência de 100%
        fig.add_shape(
            type="line",
            x0=-0.5,
            y0=100,
            x1=len(df_detalhes) - 0.5,
            y1=100,
            yref="y2",
            line=dict(color="red", width=1, dash="dash")
        )
        fig = style_fig(fig, legend_bottom=False, tight=True)
        return fig
    except Exception as e:
        print(f"Erro ao gerar gráfico de detalhes: {e}")
        # Em caso de erro, retornar gráfico simples
        fig = px.bar(title="Erro ao gerar o gráfico de detalhes")
        fig.update_layout(
            annotations=[{"text": f"Erro: {str(e)}", "showarrow": False, "font": {"size": 16}}]
        )
        fig = style_fig(fig)
        return fig

# Callback para gerar a tabela de detalhes separada por turno - NOVA VERSÃO
@app.callback(
    Output("tabela-detalhes-container", "children"),
    [Input("maquina-detalhe-dropdown", "value"),
     Input("ordenar-detalhe-radio", "value"),
     Input("store-machine-details", "data")]
)
def update_machine_details_table(maquina_selecionada, ordenar_por, machine_details):
    """Nova versão simplificada da tabela de detalhes"""
    
    debug_print(f"[DEBUG] Tabela - máquina: {maquina_selecionada}")
    
    # Validações básicas
    if not machine_details:
        return create_message_div("Dados de máquinas não disponíveis")
    
    if not maquina_selecionada:
        return create_message_div("Selecione uma máquina para ver detalhes")
    
    if maquina_selecionada not in machine_details:
        available = list(machine_details.keys())[:5]
        msg = f"Máquina '{maquina_selecionada}' não encontrada. Disponíveis: {', '.join(available)}"
        return create_message_div(msg)
    
    # Obter dados da máquina
    detalhes = machine_details[maquina_selecionada]
    debug_print(f"[DEBUG] Tabela - {len(detalhes)} itens encontrados")
    
    if not detalhes:
        return create_message_div(f"Sem dados para a máquina {maquina_selecionada}")
    
    # Verificar se são dados placeholder
    if len(detalhes) == 1 and detalhes[0].get('producao', 0) == 0:
        item_name = detalhes[0].get('item', '')
        if 'Sem dados' in item_name or 'sem registros' in item_name.lower():
            return create_no_data_div(maquina_selecionada)
    
    # Converter para DataFrame e processar
    try:
        df_detalhes = pd.DataFrame(detalhes)
        debug_print(f"[DEBUG] Tabela - colunas: {list(df_detalhes.columns)}")
        
        # Garantir colunas essenciais
        required_cols = ['item', 'producao', 'velocidade', 'eficiencia', 'tempo_producao', 'turno']
        for col in required_cols:
            if col not in df_detalhes.columns:
                if col == 'turno':
                    df_detalhes[col] = '1'
                else:
                    df_detalhes[col] = 0
        
        # Limpar e normalizar dados
        df_detalhes = df_detalhes.fillna({
            'item': 'Item sem nome',
            'turno': '1',
            'descricao_item': '',
            'paradas_item_h': 0,
            'paradas_turno_h': 0
        })
        
        # Aplicar ordenação
        if ordenar_por == "producao":
            df_detalhes = df_detalhes.sort_values('producao', ascending=False)
        elif ordenar_por == "velocidade":
            df_detalhes = df_detalhes.sort_values('velocidade', ascending=False)
        elif ordenar_por == "item":
            df_detalhes = df_detalhes.sort_values('item')
        
        # Criar tabelas por turno
        return create_tables_by_turno(df_detalhes, maquina_selecionada)
        
    except Exception as e:
        debug_print(f"[DEBUG] Erro ao processar tabela: {e}")
        return create_message_div(f"Erro ao processar dados: {str(e)}")

# Resumo por Máquina: produção, velocidade média, eficiência média
@app.callback(
    [Output("grafico-resumo-maquinas", "figure"),
     Output("tabela-resumo-maquinas", "children")],
    [Input("store-machine-details", "data")]
)
def update_resumo_maquinas(machine_details):
    try:
        if not machine_details:
            fig = px.bar(title="Carregue dados para ver o resumo por máquina")
            fig.update_layout(annotations=[{"text": "Sem dados", "showarrow": False}])
            fig = style_fig(fig)
            return fig, html.Div("Sem dados")
        linhas = []
        for maq, registros in machine_details.items():
            if not registros:
                continue
            prod_total = 0.0; tempo_total = 0.0; soma_efic_peso = 0.0
            for r in registros:
                prod_total += float(r.get('producao',0) or 0)
                t = float(r.get('tempo_producao',0) or 0)
                tempo_total += t
                e = r.get('eficiencia')
                try:
                    if e is not None and t>0:
                        soma_efic_peso += float(e) * t
                except: 
                    pass
            vel_media = (prod_total/tempo_total) if tempo_total>0 else 0
            eficiencia_media = (soma_efic_peso/tempo_total) if tempo_total>0 else 0
            linhas.append({
                'maquina': maq,
                'producao_total': prod_total,
                'velocidade_media': vel_media,
                'eficiencia_media': eficiencia_media
            })
        if not linhas:
            fig = px.bar(title="Sem registros de produção")
            fig = style_fig(fig)
            return fig, html.Div("Sem registros")
        df_resumo = pd.DataFrame(linhas).sort_values('producao_total', ascending=False)
        # Gráfico: barras produção; linhas velocidade e eficiência
        fig = go.Figure()
        fig.add_bar(x=df_resumo['maquina'], y=df_resumo['producao_total'], name='Produção', marker_color='#8b5cf6')  # Purple
        fig.add_trace(go.Scatter(x=df_resumo['maquina'], y=df_resumo['velocidade_media'], name='Velocidade Média (cart/h)', mode='lines+markers', yaxis='y2', line=dict(color='#7c3aed')))  # Purple escuro
        fig.add_trace(go.Scatter(x=df_resumo['maquina'], y=df_resumo['eficiencia_media']*100, name='Eficiência (%)', mode='lines+markers', yaxis='y3', line=dict(color='#a855f7', dash='dot')))  # Purple claro
        fig.update_layout(
            title='Resumo por Máquina',
            xaxis=dict(title='Máquina', tickangle=45),
            yaxis=dict(title='Produção'),
            yaxis2=dict(title='Velocidade Média (cart/h)', overlaying='y', side='right', position=0.92),
            yaxis3=dict(title='Eficiência (%)', anchor='free', overlaying='y', side='right', position=0.98, range=[0,200]),
            legend=dict(orientation='h', y=1.15, x=0),
            margin=dict(l=40,r=90,b=80,t=60), height=480
        )
        fig = style_fig(fig, legend_bottom=False, tight=True)
        # Tabela formatada
        def fmt_int(v):
            try: return f"{int(round(v)):,}".replace(',', '.')
            except: return '0'
        def fmt_float(v):
            try: return f"{v:,.1f}".replace(',', '.')
            except: return '0.0'
        header = html.Tr([
            html.Th('Máquina'), html.Th('Produção Total'), html.Th('Velocidade Média (cart/h)'), html.Th('Eficiência Média (%)')
        ])
        body = []
        for _, row in df_resumo.iterrows():
            body.append(html.Tr([
                html.Td(row['maquina']),
                html.Td(fmt_int(row['producao_total'])),
                html.Td(fmt_int(row['velocidade_media'])),
                html.Td(fmt_float(row['eficiencia_media']*100))
            ]))
        tabela = html.Table([html.Thead(header), html.Tbody(body)], className='table table-sm table-striped table-bordered')
        return fig, tabela
    except Exception as e:
        fig = px.bar(title='Erro no resumo')
        fig.update_layout(annotations=[{"text": str(e), "showarrow": False}])
        fig = style_fig(fig)
        return fig, html.Div(f"Erro: {e}")


def create_message_div(message):
    """Cria div com mensagem centralizada"""
    return html.Div(message, style={"textAlign": "center", "padding": "20px"})


def create_no_data_div(maquina_nome):
    """Cria div para máquina sem dados"""
    return html.Div([
        html.H5(f"Máquina {maquina_nome}", style={"textAlign": "center"}),
        html.P("Sem dados de produção para esta máquina no período selecionado.", 
               style={"textAlign": "center", "padding": "20px", "fontStyle": "italic"}),
        html.P("Carregue um arquivo de dados ou ajuste o filtro de período.", 
               style={"textAlign": "center", "color": "gray"})
    ])


def create_tables_by_turno(df_detalhes, maquina_nome):
    """Cria tabelas agrupadas por turno"""
    
    # Turnos disponíveis nos dados
    turnos_dados = sorted(df_detalhes['turno'].unique())
    turnos_padrao = ['1', '2', '3']
    
    # Garantir que todos os turnos padrão apareçam
    turnos_exibir = ['Todos'] + turnos_padrao
    
    tabelas = []
    
    for turno in turnos_exibir:
        if turno == 'Todos':
            # Tabela consolidada
            df_turno = df_detalhes.copy()
            
            # Agrupar por item
            df_agrupado = df_turno.groupby('item').agg({
                'producao': 'sum',
                'tempo_producao': 'sum',
                'velocidade': 'mean',  # placeholder; será recalculada abaixo de forma ponderada
                'velocidade_padrao': 'first',
                'eficiencia': 'mean',
                'paradas_turno_h': 'sum'
            }).reset_index()
            # Recalcular velocidade média global: produção total / tempo_producao total (ponderada)
            df_agrupado['velocidade'] = df_agrupado.apply(
                lambda r: (r['producao'] / r['tempo_producao']) if r['tempo_producao'] > 0 else 0,
                axis=1
            )
            # Recalcular eficiência agregada com base em produção total / tempo total / vel padrão
            if 'velocidade_padrao' in df_agrupado.columns:
                df_agrupado['eficiencia'] = df_agrupado.apply(lambda r: ( (r['producao'] / r['tempo_producao']) / r['velocidade_padrao'] ) if r.get('velocidade_padrao',0)>0 and r.get('tempo_producao',0)>0 else r.get('eficiencia',0), axis=1)
            
            df_agrupado['descricao_item'] = df_agrupado['item']
            
            total_paradas = df_agrupado['paradas_turno_h'].sum() if 'paradas_turno_h' in df_agrupado.columns else 0
            header_text = f"Todos os Turnos (Produção Total) - Paradas: {total_paradas:.2f}h"
            
        else:
            # Turno específico
            df_turno_filtrado = df_detalhes[df_detalhes['turno'] == turno]
            
            if len(df_turno_filtrado) == 0:
                # Criar linha vazia para turnos sem dados
                df_agrupado = pd.DataFrame([{
                    'item': f'Sem produção no Turno {turno}',
                    'descricao_item': f'Sem dados para o Turno {turno}',
                    'producao': 0,
                    'tempo_producao': 0,
                    'velocidade': 0,
                    'eficiencia': 0,
                    'paradas_turno_h': 0
                }])
            else:
                # Agrupar por item dentro do turno
                df_agrupado = df_turno_filtrado.groupby('item').agg({
                    'producao': 'sum',
                    'tempo_producao': 'sum', 
                    'velocidade': 'mean',  # placeholder
                    'velocidade_padrao': 'first',
                    'eficiencia': 'mean',
                    'paradas_turno_h': 'sum'
                }).reset_index()
                # Recalcular velocidade média ponderada por item no turno
                df_agrupado['velocidade'] = df_agrupado.apply(
                    lambda r: (r['producao'] / r['tempo_producao']) if r['tempo_producao'] > 0 else 0,
                    axis=1
                )
                if 'velocidade_padrao' in df_agrupado.columns:
                    df_agrupado['eficiencia'] = df_agrupado.apply(lambda r: ( (r['producao'] / r['tempo_producao']) / r['velocidade_padrao'] ) if r.get('velocidade_padrao',0)>0 and r.get('tempo_producao',0)>0 else r.get('eficiencia',0), axis=1)
                
                df_agrupado['descricao_item'] = df_agrupado['item']
            
            # Ajuste: antes usávamos max() (mostrava apenas o maior bloco de paradas de um item)
            # Agora somamos todas as paradas dos itens do turno para refletir o total solicitado pelo usuário
            total_paradas = df_agrupado['paradas_turno_h'].sum() if len(df_agrupado) > 0 else 0
            header_text = f"Turno: {turno} (Paradas: {total_paradas:.2f}h)"
        
        # Resumo de paradas (agora por horas totais por descrição em vez de contagem)
        bloco_paradas_html = None
        if 'paradas_item_lista' in df_detalhes.columns:
            if turno == 'Todos':
                subset = df_detalhes
            else:
                subset = df_detalhes[df_detalhes['turno'] == turno]
            from collections import defaultdict
            horas_por_desc = defaultdict(float)
            for _, r in subset.iterrows():
                lst = r.get('paradas_item_lista')
                if isinstance(lst, list):
                    for p in lst:
                        if isinstance(p, dict):
                            desc = p.get('descricao') or p.get('parada') or p.get('desc')
                            h = p.get('horas') or p.get('tempo') or 0
                            try:
                                h = float(h)
                            except Exception:
                                h = 0
                            if desc and h>0:
                                horas_por_desc[desc.strip()] += h
            if horas_por_desc:
                top = sorted(horas_por_desc.items(), key=lambda x: x[1], reverse=True)[:30]
                lista_formatada = [f"{d} ({v:.2f}h)" for d,v in top]
                bloco_paradas_html = html.Div([
                    html.Small("Paradas (Top descrições - horas)", style={'fontWeight':'bold'}),
                    html.Div(', '.join(lista_formatada), style={'fontSize':'0.7rem','whiteSpace':'normal'})
                ], style={'marginTop':'4px','marginBottom':'4px'})

        # Criar tabela HTML
        tabela = create_html_table(df_agrupado, header_text, turno == 'Todos')
        if bloco_paradas_html is not None:
            tabela.children.append(bloco_paradas_html)
        tabelas.append(tabela)
    
    return html.Div(tabelas)


def create_html_table(df_agrupado, header_text, is_consolidated):
    """Cria tabela HTML a partir do DataFrame.
    Eficiência por item = (produção / tempo_producao) / velocidade_padrao
    onde:
      produção = cartuchos no segmento (somada se consolidado)
      tempo_producao = horas efetivas (duração do(s) segmento(s) - paradas proporcionais)
      velocidade_padrao = primeira velocidade padrão associada (máquina-faca) no intervalo
    Se velocidade_padrao ausente ou zero -> eficiência exibida é fallback anterior.
    """

    # Estilo do cabeçalho
    header_style = {
        "backgroundColor": "#e3f2fd" if is_consolidated else "#f2f2f2",
        "textAlign": "center",
        "fontWeight": "bold" if is_consolidated else "normal"
    }

    # Cabeçalho da tabela
    header = html.Thead([
        html.Tr([
            html.Th(header_text, colSpan=7, style=header_style)
        ]),
        html.Tr([
            html.Th("Item/Descrição"),
            html.Th("Produção Total"),
            html.Th("Tempo (horas)"),
            html.Th("Velocidade Média (cart/h)"),
            html.Th("Velocidade Padrão"),
            html.Th("Eficiência (%)"),
            html.Th("Paradas Turno (h)")
        ])
    ])

    # Linhas de dados
    rows = []
    for _, row in df_agrupado.iterrows():
        producao = format_number(row.get('producao', 0))
        tempo = format_time(row.get('tempo_producao', 0))
        velocidade = format_velocity(row.get('velocidade', 0))
        vel_padrao_val = row.get('velocidade_padrao', 0)
        vel_padrao_fmt = format_velocity(vel_padrao_val)
        eficiencia = format_efficiency(row.get('eficiencia', 0))
        paradas_turno = format_time(row.get('paradas_turno_h', 0))
        descricao = row.get('descricao_item', '') or row.get('item', 'Item sem nome')
        rows.append(html.Tr([
            html.Td(descricao),
            html.Td(producao),
            html.Td(tempo),
            html.Td(velocidade),
            html.Td(vel_padrao_fmt),
            html.Td(eficiencia, style={"color": "green" if row.get('eficiencia', 0) >= 0.9 else "red"}),
            html.Td(paradas_turno)
        ]))

    # Seção de paradas (simplificada)
    paradas_row = html.Tr([
        html.Td(html.Strong("Paradas: "), colSpan=7,
                style={"backgroundColor": "#f9f9f9", "fontStyle": "italic"})
    ])

    # Resumo de paradas por item dentro da tabela (se houver colunas agregadas de paradas por descrição)
    extra_rows = []
    if 'paradas_item_lista' in df_agrupado.columns and df_agrupado['paradas_item_lista'].notna().any():
        from collections import defaultdict
        horas_desc = defaultdict(float); occ_desc = defaultdict(int)
        for _, r in df_agrupado.iterrows():
            lst = r.get('paradas_item_lista')
            if isinstance(lst, list):
                for p in lst:
                    if isinstance(p, dict):
                        dsc = p.get('descricao') or p.get('parada') or p.get('desc')
                        h = float(p.get('horas') or 0)
                        occ = int(p.get('ocorrencias') or 1)
                        if dsc:
                            horas_desc[dsc]+=h; occ_desc[dsc]+=occ
        if horas_desc:
            # ordenar por horas desc e limitar
            resumo = sorted(horas_desc.items(), key=lambda x: x[1], reverse=True)[:15]
            # Exibir somente horas totais por descrição (removendo contagem de ocorrências)
            texto = ', '.join([f"{d} ({horas_desc[d]:.2f}h)" for d,_ in resumo])
            extra_rows.append(html.Tr([
                html.Td(html.Small(texto, style={'fontSize':'0.65rem'}), colSpan=7, style={'backgroundColor':'#fcfcfc'})
            ]))

    return html.Table([
        header,
        html.Tbody(rows + [paradas_row] + extra_rows)
    ], className="table table-striped table-bordered",
       style={"marginBottom": "20px"})


def format_number(value):
    """Formata números para exibição"""
    try:
        val = float(value)
        if val == 0:
            return "0"
        return f"{val:,.0f}".replace(",", ".")
    except:
        return "0"

def format_velocity(v):
    try:
        f = float(v)
        if f <= 0 or pd.isna(f):
            return '0'
        if f >= 1000:
            return f"{f/1000:.2f}k"
        return f"{f:.0f}"
    except Exception:
        return '0'

def format_efficiency(value):
    """Formata eficiência"""
    try:
        val = float(value)
        if val <= 0 or pd.isna(val):
            return "N/D"
        return f"{val*100:.1f}%"
    except:
        return "N/D"

# Callbacks para download dos dados
    df_detalhes = pd.DataFrame(detalhes)
    # Sanitizar: filtrar somente registros cujo id_velocidade (ou campo maquina derivado) pertence à máquina selecionada
    if 'id_velocidade' in df_detalhes.columns:
        df_detalhes = df_detalhes[df_detalhes['id_velocidade'].astype(str).str.startswith(str(maquina_selecionada)) | (df_detalhes['id_velocidade']=='')]
    # Se existir coluna centro_trabalho usar também
    if 'centro_trabalho' in df_detalhes.columns:
        df_detalhes = df_detalhes[(df_detalhes['centro_trabalho']==maquina_selecionada) | (df_detalhes['centro_trabalho'].isna())]
    # Garantir que as colunas necessárias existam
    for col in ['item', 'producao', 'velocidade', 'eficiencia', 'tempo_producao', 'id_velocidade', 'velocidade_padrao', 'turno']:
        if col not in df_detalhes.columns:
            if col == 'id_velocidade':
                if 'faca' in df_detalhes.columns:
                    df_detalhes[col] = df_detalhes.apply(
                        lambda row: f"{maquina_selecionada}-{row['faca']}" if pd.notna(row.get('faca')) else '', 
                        axis=1
                    )
                else:
                    df_detalhes[col] = ''
            elif col == 'velocidade_padrao':
                df_detalhes[col] = 0
                if 'id_velocidade' in df_detalhes.columns:
                    for idx, row in df_detalhes.iterrows():
                        id_vel = row.get('id_velocidade', '')
                        if id_vel in DICT_VELOCIDADE:
                            df_detalhes.at[idx, col] = DICT_VELOCIDADE[id_vel]
            elif col == 'turno':
                df_detalhes[col] = 'N/D'
            else:
                df_detalhes[col] = 0
    # Limpar valores NaN
    df_detalhes = df_detalhes.fillna({'item': 'Item sem descrição', 'turno': 'N/D'})
        
        # Debug para ver os valores de turnos disponíveis
    print(f"Valores de turnos encontrados: {df_detalhes['turno'].unique()}")
        
        # Normalizar os valores de turno - garantir que sejam strings e remover espaços
    df_detalhes['turno'] = df_detalhes['turno'].astype(str).str.strip()
        
        # Agrupar por turno
    turnos_disponiveis = sorted(df_detalhes['turno'].unique()) if 'turno' in df_detalhes.columns else ['N/D']
    print(f"Turnos disponíveis após normalização: {turnos_disponiveis}")
        
        # Garantir que todos os turnos padrão (1, 2, 3) estão incluídos mesmo se não houver dados
    turnos_padrao = ['1', '2', '3']
        
        # Verificar quais turnos padrão estão faltando
    turnos_faltando = [turno for turno in turnos_padrao if turno not in turnos_disponiveis]
    print(f"Turnos faltantes: {turnos_faltando}")
        
        # Garantir que os turnos padrão (1, 2, 3) sempre estão presentes
    turnos_a_mostrar = []
    for turno in turnos_padrao:
        if turno in turnos_disponiveis or turno in turnos_faltando:
            turnos_a_mostrar.append(turno)
                
        # Adicionar outros turnos que não são padrão mas existem nos dados
    turnos_nao_padrao = [turno for turno in turnos_disponiveis if turno not in turnos_padrao]
    turnos_a_mostrar.extend(turnos_nao_padrao)
        
        # Adicionar "Todos" no início da lista para mostrar dados consolidados
    turnos = ['Todos'] + sorted(turnos_a_mostrar)
    print(f"Turnos a exibir: {turnos}")
        
        # Criar uma tabela para cada turno
    tabelas_por_turno = []
        
        # Para verificação de dados duplicados: criar dicionário para armazenar itens por turno
    itens_por_turno = {}
        
    for turno in turnos:
            # Para o turno "Todos", criar uma tabela consolidada que mostre a soma total de todos os turnos
            if turno == 'Todos':
                # Para "Todos" os turnos, usar o DataFrame completo
                df_turno = df_detalhes.copy()
                # Garantir coluna descricao_item
                if 'descricao_item' not in df_turno.columns:
                    if 'item' in df_turno.columns:
                        df_turno['descricao_item'] = df_turno['item'].astype(str)
                    else:
                        df_turno['descricao_item'] = ''
                
                # Agrupar por item e somar as produções
                df_turno_agrupado = df_turno.groupby(['item']).agg({
                    'producao': 'sum',
                    'tempo_producao': 'sum',
                    'velocidade': 'mean',
                    'velocidade_padrao': 'first',  # mostra primeira velocidade padrão encontrada
                    'eficiencia': 'mean',
                    'descricao_item': 'first'
                }).reset_index()

                # Paradas agregadas por item (somar paradas_item_h se existir)
                if 'paradas_item_h' in df_turno.columns:
                    paradas_item_sum = df_turno.groupby(['item'])['paradas_item_h'].sum().reset_index().rename(columns={'paradas_item_h':'paradas_item_h_sum'})
                    df_turno_agrupado = df_turno_agrupado.merge(paradas_item_sum, on='item', how='left')
                else:
                    df_turno_agrupado['paradas_item_h_sum'] = 0.0
                
                # Ordenar conforme solicitado
                if ordenar_por == "producao":
                    df_turno_agrupado = df_turno_agrupado.sort_values(by="producao", ascending=False)
                elif ordenar_por == "velocidade":
                    df_turno_agrupado = df_turno_agrupado.sort_values(by="velocidade", ascending=False)
                elif ordenar_por == "item":
                    df_turno_agrupado = df_turno_agrupado.sort_values(by="item")
                
                total_paradas_consol = float(df_turno_agrupado['paradas_item_h_sum'].sum()) if 'paradas_item_h_sum' in df_turno_agrupado.columns else 0.0
                turno_header = f"Todos os Turnos (Produção Total) - Paradas Soma Itens: {total_paradas_consol:.2f}h"
                header = [
                    html.Thead([
                        html.Tr([
                            html.Th(turno_header, colSpan=8, style={"backgroundColor": "#e3f2fd", "textAlign": "center", "fontWeight": "bold"})
                        ]),
                        html.Tr([
                            html.Th("Item/Descrição"),
                            html.Th("Produção Total"),
                            html.Th("Tempo (horas)"),
                            html.Th("Velocidade Média (cart/h)"),
                            html.Th("Velocidade Padrão"),
                            html.Th("Eficiência (%)"),
                            html.Th("Paradas Item (h)"),
                            html.Th("Paradas Turno (h)")
                        ])
                    ])
                ]
                
            else:
                # Processamento para turnos específicos - cada turno deve mostrar APENAS seus próprios dados
                # Filtrar rigorosamente apenas os registros que tenham o turno exato
                # Garantir que tanto o turno no DataFrame quanto o turno da iteração são strings
                turno_str = str(turno).strip()
                df_detalhes_temp = df_detalhes.copy()
                df_detalhes_temp['turno'] = df_detalhes_temp['turno'].astype(str).str.strip()
                
                # Aplicar filtro exato de turno
                df_turno = df_detalhes_temp[df_detalhes_temp['turno'] == turno_str].copy()
                # Garantir coluna descricao_item
                if 'descricao_item' not in df_turno.columns:
                    if 'item' in df_turno.columns:
                        df_turno['descricao_item'] = df_turno['item'].astype(str)
                    else:
                        df_turno['descricao_item'] = ''
                
                # Debug para verificar a quantidade de dados por turno
                print(f"Turno {turno}: {len(df_turno)} registros de {len(df_detalhes_temp)} totais")
                print(f"Valores de turno no df_detalhes: {df_detalhes_temp['turno'].unique()}")
                
                # Se este é um dos turnos padrão que não tem dados, criar uma tabela vazia
                # mas apenas se for um dos turnos padrão (1, 2, 3)
                if len(df_turno) == 0 and turno in ['1', '2', '3']:
                    # Criar uma tabela vazia mas com as colunas corretas
                    df_turno_agrupado = pd.DataFrame(columns=['item', 'producao', 'tempo_producao', 'velocidade', 
                                                              'velocidade_padrao', 'eficiencia', 'descricao_item'])
                    # Adicionar uma linha vazia
                    df_turno_agrupado = pd.concat([df_turno_agrupado, pd.DataFrame([{
                        'item': f'Sem produção no Turno {turno}',
                        'producao': 0,
                        'tempo_producao': 0,
                        'velocidade': 0,
                        'velocidade_padrao': 0,
                        'eficiencia': 0,
                        'descricao_item': f'Sem dados para o Turno {turno}'
                    }])], ignore_index=True)
                elif len(df_turno) == 0:
                    continue
                else:    
                    # Agrupar por item e somar as produções apenas para este turno específico
                    df_turno_agrupado = df_turno.groupby(['item']).agg({
                        'producao': 'sum',
                        'tempo_producao': 'sum',
                        'velocidade': 'mean',
                        'velocidade_padrao': 'first',  # primeira referência padrão
                        'eficiencia': 'mean',
                        'descricao_item': 'first'
                    }).reset_index()
                
                # Ordenar conforme solicitado
                if ordenar_por == "producao":
                    df_turno_agrupado = df_turno_agrupado.sort_values(by="producao", ascending=False)
                elif ordenar_por == "velocidade":
                    df_turno_agrupado = df_turno_agrupado.sort_values(by="velocidade", ascending=False)
                elif ordenar_por == "item":
                    df_turno_agrupado = df_turno_agrupado.sort_values(by="item")
                
                # Criar tabela para este turno
                # Calcular paradas do turno
                total_paradas_turno = 0.0
                if 'paradas_turno_h' in df_turno.columns and df_turno['paradas_turno_h'].notna().any():
                    total_paradas_turno = df_turno['paradas_turno_h'].dropna().max()
                elif 'paradas_item_h' in df_turno.columns:
                    total_paradas_turno = df_turno['paradas_item_h'].sum()
                turno_header = f"Turno: {turno} (Paradas: {total_paradas_turno:.2f}h)" if turno != "N/D" else "Sem informação de turno"
                header = [
                    html.Thead([
                        html.Tr([
                            html.Th(turno_header, colSpan=8, style={"backgroundColor": "#f2f2f2", "textAlign": "center"})
                        ]),
                        html.Tr([
                            html.Th("Item/Descrição"),
                            html.Th("Produção Total"),
                            html.Th("Tempo (horas)"),
                            html.Th("Velocidade Média (peças/h)"),
                            html.Th("Velocidade Padrão"),
                            html.Th("Eficiência (%)"),
                            html.Th("Paradas Item (h)"),
                            html.Th("Paradas Turno (h)")
                        ])
                    ])
                ]
            
            rows = []
            for _, row in df_turno_agrupado.iterrows():
                # Usar descrição do item se disponível
                descricao_item = row.get('descricao_item', '')
                # Se a descrição do item estiver vazia, tente usar o item
                if pd.isna(descricao_item) or str(descricao_item).strip() == '' or str(descricao_item).upper() == 'N/D':
                    # Tentar usar o item como descrição
                    item_desc = str(row.get('item', ''))
                    if item_desc and item_desc.lower() != 'nan' and item_desc.lower() != 'none':
                        # Usar o próprio item como descrição
                        descricao_item = item_desc
                    else:
                        # Se item também não for válido, mostrar mensagem clara
                        descricao_item = 'Item sem descrição'
                else:
                    # Garantir que a descrição é uma string válida
                    descricao_item = str(descricao_item)
                
                # Formatar os valores numéricos
                try:
                    producao_valor = float(row['producao']) if pd.notna(row['producao']) else 0
                    producao = f"{producao_valor:,.0f}".replace(",", ".")
                except (ValueError, TypeError):
                    producao = "0"
                
                # Tratar o tempo de produção
                try:
                    tempo_valor = float(row.get('tempo_producao', 0))
                    if tempo_valor > 0.01:
                        tempo = f"{tempo_valor:,.2f}h".replace(",", ".")
                    else:
                        tempo = "N/D"
                except (ValueError, TypeError):
                    tempo = "N/D"
                
                # Velocidade
                velocidade = "N/D"
                try:
                    velocidade_valor = float(row.get('velocidade', 0))
                    if velocidade_valor > 0:
                        velocidade = f"{velocidade_valor:,.0f} cart/h".replace(",", ".")
                except (ValueError, TypeError):
                    pass
                
                # Velocidade padrão
                vel_padrao = "N/D"
                try:
                    vel_padrao_valor = float(row.get('velocidade_padrao', 0))
                    if vel_padrao_valor > 0:
                        vel_padrao = f"{vel_padrao_valor:,.0f}".replace(",", ".")
                except (ValueError, TypeError):
                    pass
                
                # Eficiência
                try:
                    eficiencia_valor = float(row.get('eficiencia', 0))
                    if eficiencia_valor > 0:
                        eficiencia = f"{eficiencia_valor*100:.1f}%"
                        eficiencia_cor = 'green' if eficiencia_valor >= 0.9 else 'red'
                    else:
                        eficiencia = "N/D"
                        eficiencia_cor = 'black'
                except (ValueError, TypeError):
                    eficiencia = "N/D"
                    eficiencia_cor = 'black'
                
                # Criar linha da tabela
                # Paradas do item (para "Todos" usar a soma calculada; para turnos específicos tentar coluna original)
                # Paradas por item removidas da visualização
                # Paradas turno repetidas por linha (usar header value)
                paradas_turno_cell = f"{total_paradas_turno:.2f}h" if 'total_paradas_turno' in locals() and total_paradas_turno>0 else 'N/D'
                tr = html.Tr([
                    html.Td(descricao_item),
                    html.Td(producao),
                    html.Td(tempo),
                    html.Td(velocidade),
                    html.Td(vel_padrao),
                    html.Td(eficiencia, style={'color': eficiencia_cor, 'font-weight': 'bold'}),
                    html.Td(paradas_item_cell),
                    html.Td(paradas_turno_cell)
                ])
                rows.append(tr)

                # Adicionar bloco de paradas detalhadas logo abaixo (fora da linha principal)
                if 'paradas_item_lista' in df_turno.columns and 'item' in df_turno.columns:
                    listas = []
                    for _, rdet in df_turno[df_turno['item'] == row.get('item')].iterrows():
                        ldet = rdet.get('paradas_item_lista')
                        if isinstance(ldet, list) and ldet:
                            listas.extend(ldet)
                    agg = {}
                    for ent in listas:
                        desc = ent.get('descricao','Sem descrição')
                        horas = ent.get('horas',0)
                        agg[desc] = agg.get(desc,0)+horas
                    if agg:
                        linhas_paradas = [html.Li(f"{d}: {h:.2f}h") for d,h in agg.items()]
                        detalhamento = html.Div([
                            html.Strong("Paradas:"),
                            html.Ul(linhas_paradas, style={'marginTop':'2px','marginBottom':'8px','fontSize':'12px'})
                        ])
                    else:
                        detalhamento = html.Div([html.Strong("Paradas:"), html.Span(" Sem paradas registradas", style={'fontSize':'12px'})])
                    rows.append(html.Tr([html.Td(detalhamento, colSpan=8, style={'backgroundColor':'#fafafa'})]))
            
            body = [html.Tbody(rows)]
            
            # Criar card para o turno
            if turno == 'Todos':
                turno_display = "Todos os Turnos"
                card_style = {
                    "backgroundColor": "#e3f2fd",  # Azul claro para destaque
                    "border": "2px solid #2196F3",  # Borda azul mais escura
                    "borderRadius": "10px",
                    "marginBottom": "20px",
                    "boxShadow": "0 4px 8px rgba(0,0,0,0.1)"
                }
            else:
                turno_display = f"Turno {turno}" if turno != 'N/D' else "Sem Turno"
                # Cores diferentes para cada turno para melhor visualização
                if turno == '1':
                    card_style = {
                        "backgroundColor": "#e8f5e9",  # Verde claro para turno 1
                        "border": "2px solid #4CAF50",  # Borda verde mais escura
                        "borderRadius": "10px",
                        "marginBottom": "20px"
                    }
                elif turno == '2':
                    card_style = {
                        "backgroundColor": "#fff8e1",  # Amarelo claro para turno 2
                        "border": "2px solid #FFC107",  # Borda amarela mais escura
                        "borderRadius": "10px",
                        "marginBottom": "20px"
                    }
                elif turno == '3':
                    card_style = {
                        "backgroundColor": "#ffebee",  # Vermelho claro para turno 3
                        "border": "2px solid #F44336",  # Borda vermelha mais escura
                        "borderRadius": "10px",
                        "marginBottom": "20px"
                    }
                else:
                    card_style = {
                        "backgroundColor": "#f5f5f5",  # Cinza claro para outros
                        "border": "2px solid #9E9E9E",  # Borda cinza mais escura
                        "borderRadius": "10px",
                        "marginBottom": "20px"
                    }
            
            tabela_turno = dbc.Card([
                dbc.CardHeader(f"{turno_display} - {len(df_turno_agrupado)} itens", style=card_style),
                dbc.CardBody([
                    dbc.Table(header + body, bordered=True, striped=True, hover=True, responsive=True)
                ])
            ], className="mb-3")
            
            tabelas_por_turno.append(tabela_turno)
        
    if not tabelas_por_turno:
        return html.Div("Nenhum dado disponível para exibir", style={"textAlign": "center", "padding": "20px"})
    return html.Div(tabelas_por_turno)

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
    app.run_server(debug=True)

# Adicione estas linhas no FINAL do seu arquivo relatorio.py (após a linha if __name__ == '__main__':)

# Configuração do servidor para deploy
server = app.server

# Configuração baseada no ambiente
import os

class Config:
    DEBUG = os.environ.get('DASH_DEBUG', 'False').lower() == 'true'
    HOST = os.environ.get('HOST', '0.0.0.0')
    PORT = int(os.environ.get('PORT', 10000))

config = Config()

if __name__ == '__main__':
    print(f"""
╔════════════════════════════════════════════════════════════╗
║               DASHBOARD DE PRODUÇÃO                        ║
╠════════════════════════════════════════════════════════════╣
║                                                            ║
║  🚀 Servidor: {config.HOST}:{config.PORT}                 ║
║  🔧 Debug: {config.DEBUG}                                  ║
║  📊 Status: Ativo                                          ║
║  🌐 Ambiente: {'Produção' if config.HOST == '0.0.0.0' else 'Desenvolvimento'} ║
║                                                            ║
╚════════════════════════════════════════════════════════════╝
    """)
    
    app.run_server(
        debug=config.DEBUG,
        host=config.HOST,
        port=config.PORT,
        dev_tools_hot_reload=config.DEBUG,
        dev_tools_ui=config.DEBUG
    )