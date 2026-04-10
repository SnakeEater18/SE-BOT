"""
SNAKE EATER v9.4 — MOMENTUM CONFIRMADO (PAPER)
================================================
CRITERIOS DE ENTRADA (solo 3):
  1. Breakout de high/low de la vela anterior
  2. ATR > 0.003 (mercado no completamente muerto)
  3. NO BEAR|BEAR en MTF (único combo con 0% WR histórico)

TODO LO DEMÁS ajusta dinámicamente el tamaño de posición,
SL y TP — nunca bloquea la entrada.

GESTIÓN DINÁMICA:
  BB > 10% → margen 100%, TP 5x ATR
  BB 6-10% → margen 75%,  TP 4x ATR
  BB < 6%  → margen 50%,  TP 3x ATR
  + ajustes por ML, MTF alineado, F&G, volumen, DOM

Instalación:
  pip install ccxt pandas numpy scikit-learn requests
"""

import ccxt
import pandas as pd
import numpy as np
import time, os, sys, json, pickle, requests, warnings
from datetime import datetime
from collections import deque

warnings.filterwarnings('ignore')

# ═══════════════════════════════════════════════════════
#  CONFIGURACIÓN — solo cambia estas líneas
# ═══════════════════════════════════════════════════════
SYMBOL       = 'SIREN/USDT'
WEBHOOK      = ''
AI_KEY       = ''
AI_MODEL     = 'qwen/qwen3.6-plus:free'
LOG_FILE     = 'bitacora_v94_paper.csv'
ML_FILE      = 'siren_ml_model.pkl'

CAPITAL      = 100.0   # Capital total en cuenta
MARGEN_BASE  = 50.0    # Margen base por trade
LEVERAGE     = 10      # Apalancamiento
META_DIARIA  = 27.5    # Meta diaria USDT
MAX_DD_PCT   = 0.20    # Máx pérdida diaria (paper: solo alerta)
COMISION     = 0.0005  # 0.05% por lado Binance Futures (0.1% por trade completo)

exchange = ccxt.binance({
    'apiKey': '',
    'secret': '',
    'options': {'defaultType': 'future'},
    'enableRateLimit': True
})

# ═══════════════════════════════════════════════════════
#  PARÁMETROS DE ESTRATEGIA — calibrados para REAL
# ═══════════════════════════════════════════════════════
ATR_MIN      = 0.008   # Mercado debe tener volatilidad real
BB_ALTO      = 12.0    # BB > 12% → posición completa (zona dorada: WR 52%)
BB_MEDIO     = 8.0     # BB 8-12% → posición media
# BB < 8% → posición mínima (mercado lateral, pocas ganancias)

MIN_PROFIT   = 1.5     # Ganancia mínima para activar trailing
PEGADO_SEG   = 45      # Reducido: 90s sin avance → salir (antes 120s)
PEGADO_ATR   = 0.35    # Movimiento mínimo en ATR para no considerarlo pegado
MIN_SEG      = 25      # Duración mínima antes de SL
COOLDOWN     = 60      # Espera tras pérdida (aumentado para evitar overtrading)
COOLDOWN_MIN = 0.10    # Ganancia mínima real para no activar cooldown

# ═══════════════════════════════════════════════════════
#  INDICADORES
# ═══════════════════════════════════════════════════════
def calc_rsi(s, n=14):
    d = s.diff()
    g = d.where(d>0,0.0).rolling(n).mean()
    l = -d.where(d<0,0.0).rolling(n).mean().replace(0,np.nan)
    return 100-(100/(1+g/l))

def calc_atr(df, n=14):
    hl = df['high']-df['low']
    hc = (df['high']-df['close'].shift()).abs()
    lc = (df['low']-df['close'].shift()).abs()
    return pd.concat([hl,hc,lc],axis=1).max(axis=1).rolling(n).mean()

def calc_bb(s, n=20, k=2):
    m = s.rolling(n).mean()
    d = s.rolling(n).std()
    return m+k*d, m-k*d, (4*k*d/m)*100

def calc_vwap(df):
    tp = (df['high']+df['low']+df['close'])/3
    return (tp*df['volume']).cumsum()/df['volume'].cumsum()

def calc_momentum(df, n=3):
    d = np.sign(df['close']-df['open'])
    m = pd.Series(0, index=df.index)
    for i in range(n, len(df)):
        v = d.iloc[i-n:i]
        if (v==1).all():  m.iloc[i] = 1
        elif (v==-1).all(): m.iloc[i] = -1
    return m

# ═══════════════════════════════════════════════════════
#  DATOS DE MERCADO
# ═══════════════════════════════════════════════════════
def get_data():
    try:
        raw = exchange.fetch_ohlcv(SYMBOL, '1m', limit=210)
        df  = pd.DataFrame(raw, columns=['ts','open','high','low','close','volume'])
        df['rsi']      = calc_rsi(df['close'])
        df['atr']      = calc_atr(df)
        ml             = df['close'].ewm(span=12,adjust=False).mean()-df['close'].ewm(span=26,adjust=False).mean()
        df['macd']     = ml
        df['macd_hist']= ml - ml.ewm(span=9,adjust=False).mean()
        df['sma200']   = df['close'].rolling(200).mean()
        df['ema24']    = df['close'].ewm(span=24,adjust=False).mean()
        df['ema9']     = df['close'].ewm(span=9,adjust=False).mean()
        l14,h14        = df['low'].rolling(14).min(), df['high'].rolling(14).max()
        df['stoch']    = 100*(df['close']-l14)/(h14-l14)
        df['bb_u'],df['bb_l'],df['bb_w'] = calc_bb(df['close'])
        df['vwap']     = calc_vwap(df)
        df['vol_r']    = df['volume']/df['volume'].rolling(20).mean()
        df['momentum'] = calc_momentum(df)
        df['body_r']   = (df['close']-df['open']).abs()/(df['high']-df['low']).replace(0,np.nan)
        df['pvol_r']   = df['vol_r'].shift(1)
        df['pbody_r']  = df['body_r'].shift(1)

        mtf = {}
        for tf in ['1h','15m','5m']:
            c = [x[4] for x in exchange.fetch_ohlcv(SYMBOL,tf,limit=4)]
            if c[-1]>c[-2]>c[-3]: mtf[tf]='BULL 🟢'
            elif c[-1]<c[-2]<c[-3]: mtf[tf]='BEAR 🔴'
            else: mtf[tf]='NEUTRAL ⚪'

        last, prev = df.iloc[-1], df.iloc[-2]
        tend = 'ALCISTA' if float(last['close'])>float(last['sma200']) else 'BAJISTA'

        def f(v,n=5): return round(float(v),n) if not np.isnan(float(v)) else 0.0

        return {
            'close':f(last['close'],6), 'prev_high':f(prev['high'],6),
            'prev_low':f(prev['low'],6),
            'rsi':f(last['rsi'],2), 'atr':f(last['atr'],6),
            'macd':f(last['macd'],6), 'macd_hist':f(last['macd_hist'],6),
            'stoch':f(last['stoch'],2), 'bb_w':f(last['bb_w'],2),
            'bb_u':f(last['bb_u'],6), 'bb_l':f(last['bb_l'],6),
            'vwap':f(last['vwap'],6), 'sma200':f(last['sma200'],6),
            'ema24':f(last['ema24'],6), 'ema9':f(last['ema9'],6),
            'vol_r':f(last['vol_r'],2), 'pvol_r':f(prev['vol_r'],2),
            'momentum':int(last['momentum']), 'body_r':f(last['body_r'],3),
            'pbody_r':f(prev['body_r'],3),
            '1h':mtf['1h'],'15m':mtf['15m'],'5m':mtf['5m'],
            'tendencia':tend, 'df':df,
        }
    except Exception as e:
        print(f"[ERROR datos] {e}")
        return None

# ═══════════════════════════════════════════════════════
#  DOM
# ═══════════════════════════════════════════════════════
def get_dom():
    try:
        ob  = exchange.fetch_order_book(SYMBOL, limit=20)
        bv  = sum(b[1] for b in ob['bids'][:10])
        av  = sum(a[1] for a in ob['asks'][:10])
        if av==0: return 'NEUTRAL',1.0
        r   = bv/av
        if r>=1.3:   return 'LONG',round(r,2)
        elif r<=0.77: return 'SHORT',round(r,2)
        return 'NEUTRAL',round(r,2)
    except:
        return 'NEUTRAL',1.0

# ═══════════════════════════════════════════════════════
#  FEAR & GREED
# ═══════════════════════════════════════════════════════
_fg = {'v':50,'l':'Neutral','ts':None}
def get_fg():
    global _fg
    try:
        if _fg['ts'] and (datetime.now()-_fg['ts']).seconds<3600:
            return _fg['v'],_fg['l']
        r = requests.get('https://api.alternative.me/fng/?limit=1',timeout=5).json()['data'][0]
        _fg = {'v':int(r['value']),'l':r['value_classification'],'ts':datetime.now()}
    except: pass
    return _fg['v'],_fg['l']

# ═══════════════════════════════════════════════════════
#  AI — QWEN via OpenRouter
# ═══════════════════════════════════════════════════════
_ai = {'score':7,'razon':'Iniciando...','ts':None}
def get_ai(data, fg_v, fg_l):
    global _ai
    try:
        if _ai['ts'] and (datetime.now()-_ai['ts']).seconds<3600:
            return _ai['score'],_ai['razon']
        prompt = f"""Analiza SIREN/USDT Binance Futures.
Precio:{data['close']} RSI:{data['rsi']} BB:{data['bb_w']}% ATR:{data['atr']}
Vol:{data['vol_r']}x Tend:{data['tendencia']} MTF:1H{data['1h'][:4]} 15M{data['15m'][:4]}
F&G:{fg_v}({fg_l})
Score 1-10 para operar ahora. Solo JSON: {{"score":7,"razon":"max 12 palabras"}}"""
        r = requests.post('https://openrouter.ai/api/v1/chat/completions',
            headers={'Authorization':f'Bearer {AI_KEY}'},
            json={'model':AI_MODEL,'messages':[{'role':'user','content':prompt}]},
            timeout=15)
        txt = r.json()['choices'][0]['message']['content']
        txt = txt.strip().replace('```json','').replace('```','').strip()
        if '{' in txt: txt = txt[txt.index('{'):txt.rindex('}')+1]
        res = json.loads(txt)
        _ai = {'score':max(1,min(10,int(res.get('score',7)))),
               'razon':res.get('razon','OK'),'ts':datetime.now()}
    except Exception as e:
        _ai['razon'] = f"Error:{str(e)[:20]}"
    return _ai['score'],_ai['razon']

# ═══════════════════════════════════════════════════════
#  MACHINE LEARNING
# ═══════════════════════════════════════════════════════
_model = None
_trained = False
_buf = []
FEATS = ['rsi','atr','macd_hist','stoch','bb_w','vol_r','pvol_r','body_r','momentum']

def cargar_modelo():
    global _model,_trained
    if os.path.isfile(ML_FILE):
        try:
            with open(ML_FILE,'rb') as f: _model=pickle.load(f)
            _trained=True; print("✅ Modelo ML cargado"); return
        except: pass
    bitacoras = [f'bitacora_v{v}_paper.csv' for v in
                 ['82','83','84','85','86','87','88','90','91','92']]
    dfs=[]
    for b in bitacoras:
        if os.path.isfile(b):
            try: dfs.append(pd.read_csv(b))
            except: pass
    if not dfs: print("⚠️  Sin datos para ML — score neutro 0.6"); return
    all_df=pd.concat(dfs,ignore_index=True)
    all_df['PNL_USD']=pd.to_numeric(all_df['PNL_USD'],errors='coerce')
    cm={'RSI_entrada':'rsi','ATR_entrada':'atr','MACD_Hist_entrada':'macd_hist',
        'Stoch_entrada':'stoch','BB_width_entrada':'bb_w',
        'Vol_Ratio_entrada':'vol_r','Prev_Vol_Ratio':'pvol_r',
        'Prev_Body_Pct':'body_r'}
    all_df=all_df.rename(columns=cm)
    feats=[f for f in FEATS if f in all_df.columns]
    if len(feats)<4: return
    all_df[feats]=all_df[feats].apply(pd.to_numeric,errors='coerce')
    all_df=all_df.dropna(subset=feats+['PNL_USD'])
    all_df['g']=(all_df['PNL_USD']>=3.0).astype(int)
    try:
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.preprocessing import StandardScaler
        from sklearn.pipeline import Pipeline
        m=Pipeline([('s',StandardScaler()),
                    ('rf',RandomForestClassifier(100,max_depth=6,
                     min_samples_leaf=5,random_state=42,class_weight='balanced'))])
        m.fit(all_df[feats].values, all_df['g'].values)
        _model=m; _trained=True
        with open(ML_FILE,'wb') as f: pickle.dump(m,f)
        print(f"✅ ML entrenado: {len(all_df)} trades, {all_df['g'].sum()} ganadores")
    except Exception as e:
        print(f"⚠️  ML error: {e}")

def ml_score(data):
    if not _trained or _model is None: return 0.6
    try:
        x=np.array([float(data.get(f,0)) for f in FEATS]).reshape(1,-1)
        p=_model.predict_proba(x)[0]
        return round(float(p[1]) if len(p)>1 else 0.6,3)
    except: return 0.6

def ml_reentrenar():
    global _buf
    if len(_buf)>=50:
        print("🔄 Re-entrenando ML...")
        cargar_modelo(); _buf=[]

# ═══════════════════════════════════════════════════════
#  PARÁMETROS DINÁMICOS — ajusta margen/SL/TP
# ═══════════════════════════════════════════════════════
def params_dinamicos(data, ml, fg, dom_sig, dom_r, pnl_hoy):
    bb     = data['bb_w']
    pvol   = data['pvol_r']
    h1     = data['1h'][:4]
    m15    = data['15m'][:4]
    notas  = []

    # ── Margen y TP base por BB ──
    if bb >= BB_ALTO:
        mp, tp_m = 1.00, 5.0; notas.append(f"BB>{BB_ALTO:.0f}% ALTA")
    elif bb >= BB_MEDIO:
        mp, tp_m = 0.75, 4.0; notas.append(f"BB>{BB_MEDIO:.0f}% MED")
    else:
        mp, tp_m = 0.50, 3.0; notas.append(f"BB<{BB_MEDIO:.0f}% BAJA")

    # ── SL por volumen ──
    if pvol >= 1.5:
        sl_m = 0.8; notas.append("SL-tight")
    elif pvol >= 0.8:
        sl_m = 1.2
    else:
        sl_m = 1.5; notas.append("SL-wide")

    # ── Ajustes adicionales ──
    if ml >= 0.70:
        mp = min(1.0, mp+0.10); notas.append("+ML")
    if (h1=='BULL' and m15 in ['BULL','NEUT']) or \
       (h1=='BEAR' and m15 in ['BEAR','NEUT']):
        mp = min(1.0, mp+0.15); notas.append("+MTF")
    if fg<20 or fg>80:
        mp = max(0.30, mp-0.20); notas.append("-FG")
    if dom_sig!='NEUTRAL' and dom_r>=1.5:
        tp_m += 1.0; notas.append("+DOM")

    # ── Proteger meta ──
    prog = pnl_hoy/META_DIARIA if META_DIARIA>0 else 0
    if prog >= 0.80:
        mp = max(0.30, mp*0.50); notas.append("META80%")

    margen = max(15.0, min(MARGEN_BASE, round(MARGEN_BASE*mp, 1)))
    return margen, sl_m, tp_m, ' '.join(notas)

# ═══════════════════════════════════════════════════════
#  SEÑALES DE ENTRADA — 5 CRITERIOS BALANCEADOS
# ═══════════════════════════════════════════════════════
def señal_long(data, curr):
    # 1. Breakout del high anterior
    if curr <= data['prev_high']:
        return False, "Sin breakout high"
    # 2. ATR mínimo — mercado con volatilidad real
    if data['atr'] < ATR_MIN:
        return False, f"ATR bajo {data['atr']:.5f}<{ATR_MIN}"
    # 3. No BEAR|BEAR — único combo con 0% WR histórico
    if 'BEAR' in data['1h'] and 'BEAR' in data['15m']:
        return False, "BEAR|BEAR bloqueado"
    # 4. Volumen mínimo — breakout con volumen real
    if data['pvol_r'] < 0.6:
        return False, f"Vol insuficiente {data['pvol_r']:.2f}x"
    # 5. Momentum — no entrar contra 3 velas rojas consecutivas
    if data['momentum'] == -1:
        return False, "Momentum bajista"
    info = f"BB:{data['bb_w']:.1f}% ATR:{data['atr']:.5f} MOM:{data['momentum']}"
    return True, f"LONG ✅ {info}"

def señal_short(data, curr):
    # 1. Breakout del low anterior
    if curr >= data['prev_low']:
        return False, "Sin breakout low"
    # 2. ATR mínimo
    if data['atr'] < ATR_MIN:
        return False, f"ATR bajo {data['atr']:.5f}<{ATR_MIN}"
    # 3. No BEAR|BEAR
    if 'BEAR' in data['1h'] and 'BEAR' in data['15m']:
        return False, "BEAR|BEAR bloqueado"
    # 4. Volumen mínimo
    if data['pvol_r'] < 0.6:
        return False, f"Vol insuficiente {data['pvol_r']:.2f}x"
    # 5. Momentum — no entrar contra 3 velas verdes consecutivas
    if data['momentum'] == 1:
        return False, "Momentum alcista"
    info = f"BB:{data['bb_w']:.1f}% ATR:{data['atr']:.5f} MOM:{data['momentum']}"
    return True, f"SHORT ✅ {info}"

# ═══════════════════════════════════════════════════════
#  NIVELES DINÁMICOS
# ═══════════════════════════════════════════════════════
def niveles(ent, tipo, atr, sl_m, tp_m):
    sd = atr*sl_m; td = atr*tp_m
    if tipo=='LONG': return round(ent-sd,6), round(ent+td,6)
    return round(ent+sd,6), round(ent-td,6)

# ═══════════════════════════════════════════════════════
#  EVALUACIÓN DE SALIDA
# ═══════════════════════════════════════════════════════
def evaluar_salida(curr, ent, tipo, pnl, mpnl, seg, sl, tp, atr, phist, margen):
    """
    Sistema de salida mejorado con trailing ATR dinámico.
    
    Niveles de trailing (activo desde $2 de ganancia):
      $2-4  → sale si retrocede 0.4x ATR desde el máximo
      $4-8  → sale si retrocede 0.35x ATR desde el máximo  
      $8+   → sale si retrocede 0.25x ATR desde el máximo (muy ajustado)
    
    Elimina la espera de 120s — reacciona al precio directamente.
    """
    noc = margen * LEVERAGE

    # ── SL dinámico — mínimo MIN_SEG antes de ejecutar ──
    if seg >= MIN_SEG:
        if tipo=='LONG' and curr<=sl:  return True, f"SL {pnl:.2f}"
        if tipo=='SHORT' and curr>=sl: return True, f"SL {pnl:.2f}"
    elif pnl <= -abs(sl-ent)/ent*noc*1.5:  # caída libre — cortar inmediato
        if tipo=='LONG' and curr<=sl:  return True, f"SL {pnl:.2f}"
        if tipo=='SHORT' and curr>=sl: return True, f"SL {pnl:.2f}"

    # ── TP ──
    if tipo=='LONG' and curr>=tp:  return True, f"TP {pnl:.2f}"
    if tipo=='SHORT' and curr<=tp: return True, f"TP {pnl:.2f}"

    # ── Trailing ATR dinámico — prioriza capturar ganancias pequeñas ──
    if mpnl >= MIN_PROFIT and len(phist) >= 3:
        # Escalonado: más ganancia = trailing más ajustado
        if mpnl >= 10.0:
            retroceso_max = atr * 0.20   # muy ajustado — protege ganancias grandes
        elif mpnl >= 6.0:
            retroceso_max = atr * 0.28
        elif mpnl >= 3.0:
            retroceso_max = atr * 0.38
        else:
            retroceso_max = atr * 0.50   # suave — da espacio en ganancias pequeñas

        reciente = list(phist)[-5:]
        if tipo == 'LONG':
            retroceso = max(reciente) - curr
        else:
            retroceso = curr - min(reciente)

        if retroceso >= retroceso_max and pnl >= MIN_PROFIT:
            return True, f"Trailing ${pnl:.2f} (max ${mpnl:.2f})"

    # ── Precio pegado — sin movimiento en PEGADO_SEG segundos ──
    if pnl >= MIN_PROFIT and seg >= PEGADO_SEG and len(phist) >= 8:
        if (max(list(phist)[-8:]) - min(list(phist)[-8:])) < atr * PEGADO_ATR:
            return True, f"Pegado ${pnl:.2f} (max ${mpnl:.2f})"

    # ── BE automático — llegó al 40% del TP y regresó a 0 ──
    tp_usd = abs(tp-ent)/ent*noc
    if mpnl >= tp_usd*0.40 and pnl < 0.10:
        return True, f"BE auto max=${mpnl:.2f}"

    return False, ""

# ═══════════════════════════════════════════════════════
#  EXPLICACIÓN EDUCATIVA DE ENTRADA
# ═══════════════════════════════════════════════════════
def explicar(tipo, data, curr, sl, tp, margen, sl_m, tp_m, atr):
    noc    = margen*LEVERAGE
    sl_d   = abs(curr-sl); tp_d = abs(tp-curr)
    sl_pct = round(sl_d/curr*100,3); tp_pct = round(tp_d/curr*100,3)
    sl_usd = round(sl_d/curr*noc,2); tp_usd = round(tp_d/curr*noc,2)
    rr     = round(tp_d/sl_d,2) if sl_d>0 else 0

    mom_txt = 'momentum confirmado' if data['momentum']!=0 else 'breakout puro sin momentum'
    ap = (f"Breakout {'alcista' if tipo=='LONG' else 'bajista'} — {mom_txt}. "
          f"Cuerpo de vela: {data['pbody_r']:.0%} del rango.")
    tec = (f"RSI {data['rsi']} | BB {data['bb_w']:.1f}% | "
           f"ATR {atr:.5f} | MTF {data['1h'][:4]}|{data['15m'][:4]}")
    obj = (f"SL={sl} (-{sl_pct}%=-${sl_usd}) basado en {sl_m}×ATR. "
           f"TP={tp} (+{tp_pct}%=+${tp_usd}) basado en {tp_m}×ATR. R:R=1:{rr}")
    gest= (f"${margen}×{LEVERAGE}x=${noc:.0f} nocional. "
           f"Riesgo máx: ${sl_usd} ({sl_usd/CAPITAL*100:.1f}% del capital)")
    return {'ap':ap,'tec':tec,'obj':obj,'gest':gest,
            'sl_usd':sl_usd,'tp_usd':tp_usd,'rr':rr,'noc':noc}

# ═══════════════════════════════════════════════════════
#  BITÁCORA
# ═══════════════════════════════════════════════════════
COLS=['Fecha_Apertura','Fecha_Cierre','Duracion_seg','Tipo',
      'Entrada','Salida','SL','TP','Margen','Nocional',
      'PNL_USD','Max_PNL','Tier',
      'RSI_entrada','ATR_entrada','MACD_Hist_entrada','Stoch_entrada',
      'BB_width_entrada','Vol_Ratio_entrada','Prev_Vol_Ratio',
      'Body_Ratio','Momentum','VWAP_entrada','EMA9_entrada','EMA24_entrada',
      '1H','15M','5M','Tendencia',
      'DOM_Signal','DOM_Ratio','Fear_Greed','Fear_Greed_Label',
      'AI_Score','AI_Razon','ML_Score',
      'Resultado','Razon_Salida']

def init_log():
    if not os.path.isfile(LOG_FILE):
        pd.DataFrame(columns=COLS).to_csv(LOG_FILE,index=False)

def log_trade(tipo,ent,curr,pnl,mpnl,sl,tp,margen,tier,
              razon,snap,ts0,ts1,dom_s,dom_r,fg_v,fg_l,ai_s,ai_r,ml_s):
    dur = int((ts1-ts0).total_seconds())
    res = 'GANANCIA' if pnl>0 else ('BE' if abs(pnl)<0.01 else 'PERDIDA')
    noc = margen*LEVERAGE
    # Fechas en formato ISO sin espacios para que Excel no las corrompa
    f0 = ts0.strftime('%Y-%m-%dT%H:%M:%S')
    f1 = ts1.strftime('%Y-%m-%dT%H:%M:%S')
    razon_safe = str(razon).replace(',',';')
    tier_safe  = str(tier).replace(',',';')
    ai_r_safe  = str(ai_r).replace(',',';')
    row=[f0,f1,dur,tipo,ent,curr,sl,tp,margen,noc,
         round(pnl,3),round(mpnl,3),tier_safe,
         snap['rsi'],snap['atr'],snap['macd_hist'],snap['stoch'],
         snap['bb_w'],snap['vol_r'],snap['pvol_r'],
         snap['body_r'],snap['momentum'],snap['vwap'],
         snap['ema9'],snap['ema24'],
         snap['1h'],snap['15m'],snap['5m'],snap['tendencia'],
         dom_s,dom_r,fg_v,fg_l,ai_s,ai_r_safe,ml_s,res,razon_safe]
    with open(LOG_FILE,'a') as f:
        f.write(','.join(map(str,row))+'\n')

_sesion_inicio = None  # Se fija en main() al arrancar

def stats_sesion():
    """Lee TODOS los trades del CSV — sin filtrar por sesión ni hora."""
    try:
        if not os.path.isfile(LOG_FILE):
            return 0.0, 0, 0.0, []
        df = pd.read_csv(LOG_FILE, on_bad_lines='skip')
        df['PNL_USD'] = pd.to_numeric(df['PNL_USD'], errors='coerce')
        df = df.dropna(subset=['PNL_USD'])
        if len(df) == 0:
            return 0.0, 0, 0.0, []
        pnl = round(float(df['PNL_USD'].sum()), 2)
        tt  = len(df)
        w   = int((df['PNL_USD'] > 0).sum())
        wr  = round(w / tt * 100, 1)
        return pnl, tt, wr, df.tail(6).to_dict('records')
    except:
        return 0.0, 0, 0.0, []

def stats_hoy():
    return stats_sesion()

# ═══════════════════════════════════════════════════════
#  DISCORD
# ═══════════════════════════════════════════════════════
def notify(msg):
    try: requests.post(WEBHOOK,json={"content":msg},timeout=5)
    except: pass

def notify_entrada(tipo,curr,sl,tp,margen,tier,ml_s,ai_s,fg_v,fg_l,
                   dom_s,dom_r,exp):
    e='🚀' if tipo=='LONG' else '📉'
    msg=(f"{e} **[PAPER v9.4] {tipo} — SIREN/USDT**\n"
         f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
         f"📍 **Entrada:** `{curr}`\n"
         f"🛑 **SL:** `{sl}` → `-${exp['sl_usd']} USDT`\n"
         f"🎯 **TP:** `{tp}` → `+${exp['tp_usd']} USDT`\n"
         f"📊 **R:R:** `1:{exp['rr']}` | **Nocional:** `${exp['noc']:.0f}`\n"
         f"⚡ **Tier:** `{tier}`\n"
         f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
         f"🤖 ML:`{ml_s:.0%}` 🧠 AI:`{ai_s}/10` 😱 F&G:`{fg_v} {fg_l}`\n"
         f"📦 DOM:`{dom_s} {dom_r}x`\n"
         f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
         f"📖 **Acción precio:** {exp['ap']}\n"
         f"📐 **Técnico:** {exp['tec']}\n"
         f"🎯 **Objetivos:** {exp['obj']}\n"
         f"💼 **Gestión:** {exp['gest']}\n"
         f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
         f"💡 _Ejecuta en real si lo consideras_")
    notify(msg)

def notify_cierre(tipo,ent,curr,pnl,mpnl,razon,dur):
    e='✅' if pnl>0 else '❌'
    notify(f"{e} **[PAPER v9.4] CIERRE {tipo}**\n"
           f"`{ent}` → `{curr}` | PNL:`{pnl:+.2f}` Max:`{mpnl:.2f}`\n"
           f"Razón: {razon} | {dur}s")

# ═══════════════════════════════════════════════════════
#  DASHBOARD
# ═══════════════════════════════════════════════════════
def dashboard(estado, pnl_act, mpnl, data,
              sl=0, tp=0, margen=MARGEN_BASE, tier='',
              ai_s=7, ai_r='', ml_s=0.6,
              fg_v=50, fg_l='Neutral',
              dom_s='NEUTRAL', dom_r=1.0,
              msg='', rech=0, exp=None,
              ot=False, ot_msg=''):
    ph,tt,wr,ops = stats_hoy()
    prog = round(ph/META_DIARIA*100,1)
    noc  = margen*LEVERAGE
    dd   = round(abs(ph)/CAPITAL*100,2) if ph<0 else 0.0
    R,G,RS,Y,B,C,M = ("\033[91m","\033[92m","\033[0m",
                       "\033[93m","\033[94m","\033[96m","\033[95m")
    sys.stdout.write("\033[H\033[J")
    print(f"🐍 SNAKE EATER v9.4 — MOMENTUM CONFIRMADO (PAPER)")
    print(f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    barra_n    = int(min(max(prog,0),100)/5)
    barra      = f"{G}{'█'*barra_n}{'░'*(20-barra_n)}{RS}"
    cap_actual = round(CAPITAL + ph, 2)
    print(f"💰 CAPITAL  : ${CAPITAL} → Actual:{G if cap_actual>=CAPITAL else R}${cap_actual}{RS} | Margen:{G}${margen}{RS}×{LEVERAGE}x={Y}${noc:.0f}{RS}")
    print(f"⚡ TIER     : {M}{tier}{RS}")
    print(f"📊 PNL SESIÓN: {G if ph>=0 else R}{ph:+.2f} USDT{RS}  {barra}  {prog:.1f}% de ${META_DIARIA}")
    print(f"📈 TRADES   : {B}{tt}{RS} | WR:{G if wr>=45 else Y if wr>=30 else R}{wr}%{RS} | RECH:{Y}{rech}{RS}")
    if ot: print(f"🚨 {R}{ot_msg}{RS}")
    elif dd>10: print(f"⚠️  {Y}DD: -{dd:.1f}% del capital (límite {MAX_DD_PCT*100:.0f}%){RS}")
    print(f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    print(f"📡 ESTADO   : {C}{estado}{RS}")
    if estado.startswith("DENTRO") and sl and tp:
        cp = G if pnl_act>=0 else R
        curr = data['close']
        sl_d  = abs(curr-sl); tp_d = abs(tp-curr)
        sl_u  = round(sl_d/curr*noc,2) if curr else 0
        tp_u  = round(tp_d/curr*noc,2) if curr else 0
        print(f"  PNL: {cp}{pnl_act:+.2f}{RS} | MAX:{Y}{mpnl:+.2f}{RS}")
        print(f"  🛑 SL:{R}{sl}{RS}(-${sl_u}) | 🎯 TP:{G}{tp}{RS}(+${tp_u})")
        if exp:
            print(f"  📖 {exp.get('ap','')[:70]}")
            print(f"  📐 {exp.get('tec','')[:70]}")
    print(f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    print(f"📊 MTF  : 1H:{data['1h']} 15M:{data['15m']} 5M:{data['5m']} {Y}{data['tendencia']}{RS}")
    print(f"🎯 RSI:{Y}{data['rsi']}{RS} ATR:{data['atr']} BB:{Y}{data['bb_w']}%{RS} STOCH:{data['stoch']}")
    print(f"📦 VOL:{Y}{data['vol_r']}x{RS} MOM:{data['momentum']} BODY:{data['body_r']} VWAP:{data['vwap']}")
    print(f"🌐 DOM:{C}{dom_s}{RS}({dom_r}x) F&G:{Y}{fg_v}{RS}({fg_l})")
    print(f"🧠 AI:{Y}{ai_s}/10{RS} {ai_r[:40]} | 🤖 ML:{Y}{ml_s:.2f}{RS}")
    if msg: print(f"⛔ {Y}{msg}{RS}")
    print(f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    print(f"📝 HISTORIAL SESIÓN (últimos {len(ops)}):")
    for op in reversed(ops):
        try:
            pv = float(op['PNL_USD'])
            c  = G if pv>0 else R
            ts = str(op.get('Fecha_Apertura','')).split(' ')
            ts = ts[1][:5] if len(ts)>1 else '??:??'
            dur= op.get('Duracion_seg','?')
            rz = str(op.get('Razon_Salida',''))[:25]
            print(f"  • {ts}|{str(op.get('Tipo','?')):5}|{c}{pv:+.2f}{RS}|{dur}s|{rz}")
        except: pass

# ═══════════════════════════════════════════════════════
#  PNL
# ═══════════════════════════════════════════════════════
def calc_pnl(curr, ent, tipo, margen):
    """PNL descontando comisiones reales de Binance Futures (0.05% por lado)."""
    d = (curr-ent) if tipo=='LONG' else (ent-curr)
    pnl_bruto = (d/ent)*(margen*LEVERAGE)
    nocional  = margen * LEVERAGE
    comision  = nocional * COMISION * 2  # entrada + salida
    return pnl_bruto - comision

# ═══════════════════════════════════════════════════════
#  MOTOR PRINCIPAL
# ═══════════════════════════════════════════════════════
def main():
    global _sesion_inicio
    _sesion_inicio = datetime.now()
    print("🚀 Snake Eater v9.4 iniciando...")
    print(f"   Sesión desde: {_sesion_inicio.strftime('%Y-%m-%d %H:%M:%S')}")
    cargar_modelo()
    init_log()
    print(f"   Log: {LOG_FILE}\n   Ctrl+C para detener\n")
    time.sleep(2)

    # Estado
    in_pos   = False
    ent_p    = sl = tp = atr_e = 0.0
    tipo     = None
    mpnl     = 0.0
    snap     = {}
    ts0      = None
    phist    = deque(maxlen=20)
    margen   = MARGEN_BASE
    sl_m     = 1.2
    tp_m     = 4.0
    tier     = ''
    exp      = {}
    dom_snap = ('NEUTRAL',1.0)
    fg_snap  = (50,'Neutral')
    rech     = 0
    msg      = ''
    ts_loss  = None
    ts_resumen = datetime.now().date()

    # Cache módulos externos
    ai_s, ai_r = 7, 'Iniciando...'
    ml_s       = 0.6
    fg_v, fg_l = 50, 'Neutral'
    dom_s, dom_r = 'NEUTRAL', 1.0

    while True:
        try:
            # Resumen diario
            hoy = datetime.now().date()
            if hoy != ts_resumen:
                ph,tt,wr,_ = stats_hoy()
                notify(f"📊 **Resumen diario v9.4**\nPNL:`{ph:+.2f}` Trades:`{tt}` WR:`{wr}%`")
                ts_resumen = hoy
                ml_reentrenar()

            data = get_data()
            if not data: time.sleep(2); continue
            curr = exchange.fetch_ticker(SYMBOL)['last']

            # Módulos externos — no bloquean el loop
            try:
                fg_v, fg_l   = get_fg()
                dom_s, dom_r = get_dom()
                ml_s         = ml_score(data)
                ai_s, ai_r   = get_ai(data, fg_v, fg_l)
            except: pass

            if in_pos:
                pnl  = calc_pnl(curr, ent_p, tipo, margen)
                mpnl = max(mpnl, pnl)
                seg  = int((datetime.now()-ts0).total_seconds())
                phist.append(curr)

                dashboard(f"DENTRO {tipo}", pnl, mpnl, data,
                          sl=sl, tp=tp, margen=margen, tier=tier,
                          ai_s=ai_s, ai_r=ai_r, ml_s=ml_s,
                          fg_v=fg_v, fg_l=fg_l,
                          dom_s=dom_s, dom_r=dom_r,
                          rech=rech, exp=exp)

                cerrar, razon = evaluar_salida(
                    curr, ent_p, tipo, pnl, mpnl,
                    seg, sl, tp, atr_e, list(phist), margen
                )
                if cerrar:
                    ts1    = datetime.now()
                    in_pos = False
                    phist.clear()
                    log_trade(tipo,ent_p,curr,pnl,mpnl,sl,tp,margen,tier,
                              razon,snap,ts0,ts1,
                              dom_s,dom_r,fg_v,fg_l,ai_s,ai_r,ml_s)
                    _buf.append({'pnl':pnl,**snap})
                    if pnl < COOLDOWN_MIN: ts_loss = ts1
                    notify_cierre(tipo,ent_p,curr,pnl,mpnl,razon,
                                  int((ts1-ts0).total_seconds()))

            else:
                # Cooldown
                if ts_loss:
                    seg_cd = int((datetime.now()-ts_loss).total_seconds())
                    if seg_cd < COOLDOWN:
                        msg = f"Cooldown {COOLDOWN-seg_cd}s"
                        dashboard("COOLDOWN",0,0,data,margen=margen,tier=tier,
                                  ai_s=ai_s,ai_r=ai_r,ml_s=ml_s,
                                  fg_v=fg_v,fg_l=fg_l,
                                  dom_s=dom_s,dom_r=dom_r,
                                  msg=msg,rech=rech)
                        time.sleep(0.5); continue

                # Calcular parámetros dinámicos
                ph_now,tt_now,_,_ = stats_hoy()
                margen, sl_m, tp_m, tier = params_dinamicos(
                    data, ml_s, fg_v, dom_s, dom_r, ph_now
                )
                ot  = abs(ph_now)/CAPITAL >= MAX_DD_PCT if ph_now<0 else False
                otm = f"⚠️ DD {abs(ph_now)/CAPITAL*100:.1f}% (límite {MAX_DD_PCT*100:.0f}%)" if ot else ''

                l_ok, l_m = señal_long(data, curr)
                s_ok, s_m = señal_short(data, curr)

                if l_ok:
                    sl, tp = niveles(curr,'LONG',data['atr'],sl_m,tp_m)
                    in_pos=True; ent_p=curr; tipo='LONG'
                    mpnl=0.0; snap=data.copy(); ts0=datetime.now()
                    atr_e=data['atr']; phist.clear(); msg=''
                    exp = explicar('LONG',data,curr,sl,tp,margen,sl_m,tp_m,data['atr'])
                    notify_entrada('LONG',curr,sl,tp,margen,tier,ml_s,ai_s,
                                   fg_v,fg_l,dom_s,dom_r,exp)

                elif s_ok:
                    sl, tp = niveles(curr,'SHORT',data['atr'],sl_m,tp_m)
                    in_pos=True; ent_p=curr; tipo='SHORT'
                    mpnl=0.0; snap=data.copy(); ts0=datetime.now()
                    atr_e=data['atr']; phist.clear(); msg=''
                    exp = explicar('SHORT',data,curr,sl,tp,margen,sl_m,tp_m,data['atr'])
                    notify_entrada('SHORT',curr,sl,tp,margen,tier,ml_s,ai_s,
                                   fg_v,fg_l,dom_s,dom_r,exp)

                else:
                    rech += 1
                    if curr > data['prev_high']:   msg = f"LONG rech — {l_m}"
                    elif curr < data['prev_low']:  msg = f"SHORT rech — {s_m}"
                    else:                          msg = "Sin breakout"
                    dashboard("ACECHANDO",0,0,data,
                              margen=margen,tier=tier,
                              ai_s=ai_s,ai_r=ai_r,ml_s=ml_s,
                              fg_v=fg_v,fg_l=fg_l,
                              dom_s=dom_s,dom_r=dom_r,
                              msg=msg,rech=rech,
                              ot=ot,ot_msg=otm)

        except KeyboardInterrupt:
            print("\n🛑 Detenido.")
            ph,tt,wr,_ = stats_hoy()
            print(f"Trades:{tt} PNL:{ph:+.2f} WR:{wr}%")
            notify(f"🛑 Bot detenido.\nPNL:`{ph:+.2f}` Trades:`{tt}` WR:`{wr}%`")
            break
        except Exception as e:
            print(f"[ERROR] {e}")
            time.sleep(2)
        time.sleep(0.5)

if __name__ == "__main__":
    main()
