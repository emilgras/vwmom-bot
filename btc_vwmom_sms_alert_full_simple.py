
#!/usr/bin/env python3
"""
BTC VWMOM — SMS Alert Bot (Simple SMS format, alerts-only, single position)
- Entry: VWMOM crosses thresholds
- Exit: SL / TP / Trailing / Signal Reverse / Time / Vol regime (optional)
- SMS are short and easy to read
"""

import os, time, json
from dataclasses import dataclass, asdict
from typing import Optional, Dict, Any, Tuple
from datetime import datetime
from zoneinfo import ZoneInfo
import numpy as np, pandas as pd, ccxt

try:
    from twilio.rest import Client
except Exception:
    Client = None

try:
    from dotenv import load_dotenv
except Exception:
    def load_dotenv():
        return

STATE_PATH = os.getenv("STATE_PATH", "vwmom_alert_state.json")

def load_state() -> Dict[str, Any]:
    if os.path.exists(STATE_PATH):
        try:
            with open(STATE_PATH, "r") as f:
                return json.load(f)
        except Exception:
            pass
    return {"last_signal":"flat","last_alert_ts":0,"position":None}

def save_state(state: Dict[str, Any]):
    tmp = STATE_PATH + ".tmp"
    with open(tmp, "w") as f:
        json.dump(state, f)
    os.replace(tmp, STATE_PATH)

def init_exchange():
    ex_name = os.getenv("EXCHANGE","binance").lower()
    klass = getattr(ccxt, ex_name, None)
    if klass is None: raise RuntimeError(f"Unsupported exchange '{ex_name}'")
    return klass({"enableRateLimit":True})

def fetch_ohlcv(ex, symbol, timeframe, limit):
    o = ex.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
    if not o: raise RuntimeError("Empty OHLCV response.")
    df = pd.DataFrame(o, columns=["ts","open","high","low","close","volume"])
    df["ts"] = pd.to_datetime(df["ts"], unit="ms", utc=True)
    return df

def compute_atr(df: pd.DataFrame, period: int) -> pd.Series:
    high, low, close = df["high"], df["low"], df["close"]
    prev_close = close.shift(1)
    tr = pd.concat([(high-low).abs(), (high-prev_close).abs(), (low-prev_close).abs()], axis=1).max(axis=1)
    return tr.rolling(period).mean()

def realized_vol(close: pd.Series, window: int) -> float:
    rets = close.pct_change().dropna()
    if len(rets) < window: return np.nan
    r = rets.iloc[-window:]
    return float(r.std() * np.sqrt(365*24*60))

def compute_vwmom(df: pd.DataFrame, L: int, vol_ma_windows: int):
    if len(df) < (L + vol_ma_windows + 10): return None, None, None
    vol_L = df["volume"].rolling(L).sum()
    vol_L_avg = vol_L.rolling(vol_ma_windows).mean()
    ret_L = df["close"]/df["close"].shift(L) - 1.0
    vol_ratio = vol_L / (vol_L_avg + 1e-12)
    vwmom = ret_L * vol_ratio
    i = -2
    last_vwmom = float(vwmom.iloc[i])
    last_ret = float(ret_L.iloc[i])
    last_vol_ratio = float(vol_ratio.iloc[i])
    if np.isnan(last_vwmom): return None, None, None
    return last_vwmom, last_ret, last_vol_ratio

@dataclass
class Position:
    side: str
    entry_price: float
    entry_ts: int
    atr_at_entry: float
    R: float
    stop_price: float
    target_price: float
    trailing_stop: Optional[float]=None
    highest_since_entry: Optional[float]=None
    lowest_since_entry: Optional[float]=None
    bars_in_trade: int=0
    entry_bar_index: int=0

def init_twilio():
    sid = os.getenv("TWILIO_ACCOUNT_SID")
    token = os.getenv("TWILIO_AUTH_TOKEN")
    from_num = os.getenv("TWILIO_FROM_NUMBER")
    to_num = os.getenv("TWILIO_TO_NUMBER")
    if not all([sid, token, from_num, to_num]):
        raise RuntimeError("Missing Twilio vars.")
    if Client is None:
        raise RuntimeError("twilio package not installed. pip install twilio")
    return Client(sid, token), from_num, to_num

def send_sms(client, from_num, to_num, body: str):
    try:
        client.messages.create(to=to_num, from_=from_num, body=body)
        print(f"[SMS] {body}")
    except Exception as e:
        print(f"[SMS ERROR] {e}")

def fmt_time(ts: int, tz: ZoneInfo) -> str:
    return datetime.fromtimestamp(ts, tz).strftime("%H:%M %Z")

def fmt_now(tz: ZoneInfo) -> str:
    return datetime.now(tz).strftime("%H:%M %Z")

def msg_entry(side, symbol, timeframe, price, sl, tp, ts_local):
    side_txt = "BUY" if side=="long" else "SELL"
    return f"{side_txt} {symbol} {timeframe} @ {price:,.2f} | SL {sl:,.2f} | TP {tp:,.2f} | {ts_local}"

def msg_exit(reason, side, symbol, timeframe, price, pnl_pct, pnl_abs, ts_local):
    tag = {"STOP LOSS":"SL","TAKE PROFIT":"TP","TRAILING":"TRAIL","SIGNAL REVERSE":"REV","TIME EXIT":"TIME","VOL REGIME":"VOL"}.get(reason,reason)
    wl = "WIN" if pnl_pct>0 else ("LOSS" if pnl_pct<0 else "FLAT")
    base = f"EXIT {tag} {symbol} {timeframe} @{price:,.2f} | {wl} {pnl_pct*100:.2f}%"
    if pnl_abs is not None: base += f" ({pnl_abs:+.2f} USD)"
    return f"{base} | {ts_local}"

def create_position(side, entry_price, atr, cfg, entry_ts, entry_bar_index):
    R = float(cfg["ATR_MULT_STOP"]) * atr
    if side=="long":
        stop = entry_price - R
        target = entry_price + float(cfg["TP_R_MULT"]) * R
    else:
        stop = entry_price + R
        target = entry_price - float(cfg["TP_R_MULT"]) * R
    return Position(side=side, entry_price=entry_price, entry_ts=entry_ts,
                    atr_at_entry=atr, R=R, stop_price=stop, target_price=target,
                    highest_since_entry=entry_price, lowest_since_entry=entry_price,
                    entry_bar_index=entry_bar_index)

def update_trailing_and_stops(pos: Position, df: pd.DataFrame, atr_series: pd.Series, cfg: Dict[str,Any]):
    i=-2; high=float(df["high"].iloc[i]); low=float(df["low"].iloc[i]); close=float(df["close"].iloc[i]); atr=float(atr_series.iloc[i])
    pos.highest_since_entry = max(pos.highest_since_entry or -np.inf, high)
    pos.lowest_since_entry  = min(pos.lowest_since_entry  or  np.inf, low)
    be_mult = float(cfg["BREAKEVEN_AT_R"])
    if pos.side=="long" and close >= pos.entry_price + be_mult*pos.R:
        pos.stop_price = max(pos.stop_price, pos.entry_price)
    elif pos.side=="short" and close <= pos.entry_price - be_mult*pos.R:
        pos.stop_price = min(pos.stop_price, pos.entry_price)
    trail_after = float(cfg["TRAIL_AFTER_R"]); chand_mult=float(cfg["CHANDELIER_MULT"])
    if pos.side=="long" and close >= pos.entry_price + trail_after*pos.R:
        chandelier = pos.highest_since_entry - chand_mult*atr
        pos.trailing_stop = max(pos.trailing_stop or -np.inf, chandelier)
        pos.stop_price = max(pos.stop_price, pos.trailing_stop)
    elif pos.side=="short" and close <= pos.entry_price - trail_after*pos.R:
        chandelier = pos.lowest_since_entry + chand_mult*atr
        pos.trailing_stop = min(pos.trailing_stop or  np.inf, chandelier)
        pos.stop_price = min(pos.stop_price, pos.trailing_stop)
    pos.bars_in_trade += 1
    return pos

def check_exit(pos: Position, df: pd.DataFrame, vwmom: float, cfg: Dict[str,Any], bar_index:int, rv_spike: Optional[bool]):
    i=-2; close=float(df["close"].iloc[i])
    reason=None
    if pos.side=="long" and close <= pos.stop_price: reason="STOP LOSS"
    elif pos.side=="short" and close >= pos.stop_price: reason="STOP LOSS"
    if reason is None:
        if pos.side=="long" and close >= pos.target_price: reason="TAKE PROFIT"
        elif pos.side=="short" and close <= pos.target_price: reason="TAKE PROFIT"
    if reason is None and os.getenv("EXIT_ON_SIGNAL_REVERSE","true").lower()=="true":
        if pos.side=="long" and vwmom < -float(cfg["THRESH_DN"]): reason="SIGNAL REVERSE"
        elif pos.side=="short" and vwmom > float(cfg["THRESH_UP"]): reason="SIGNAL REVERSE"
    if reason is None and os.getenv("EXIT_ON_VOL_SPIKE","false").lower()=="true" and rv_spike is not None and rv_spike:
        reason="VOL REGIME"
    if reason is None:
        if (bar_index - pos.entry_bar_index) >= int(cfg["MAX_BARS_IN_TRADE"]): reason="TIME EXIT"
    if reason is None: return None, None
    return reason, close

def main():
    load_dotenv()
    cfg = {
        "SYMBOL": os.getenv("SYMBOL","BTC/USDT"),
        "TIMEFRAME": os.getenv("TIMEFRAME","5m"),
        "LOOKBACK_BARS": int(os.getenv("LOOKBACK_BARS","288")),
        "VOL_SMA_WINDOWS": int(os.getenv("VOL_SMA_WINDOWS","20")),
        "THRESH_UP": float(os.getenv("THRESH_UP","0.003")),
        "THRESH_DN": float(os.getenv("THRESH_DN","0.003")),
        "ENABLE_SHORT_ALERTS": os.getenv("ENABLE_SHORT_ALERTS","true"),
        "POLL_SECONDS": int(os.getenv("POLL_SECONDS","60")),
        "COOLDOWN_MINUTES": int(os.getenv("COOLDOWN_MINUTES","30")),
        "TIMEZONE": os.getenv("TIMEZONE","Europe/Copenhagen"),
        "ATR_PERIOD": int(os.getenv("ATR_PERIOD","96")),
        "ATR_MULT_STOP": float(os.getenv("ATR_MULT_STOP","2.0")),
        "TP_R_MULT": float(os.getenv("TP_R_MULT","2.0")),
        "BREAKEVEN_AT_R": float(os.getenv("BREAKEVEN_AT_R","1.0")),
        "TRAIL_AFTER_R": float(os.getenv("TRAIL_AFTER_R","1.5")),
        "CHANDELIER_MULT": float(os.getenv("CHANDELIER_MULT","3.0")),
        "MAX_BARS_IN_TRADE": int(os.getenv("MAX_BARS_IN_TRADE","576")),
        "RV_WINDOW": int(os.getenv("RV_WINDOW","288")),
        "RV_CEILING_MULT": float(os.getenv("RV_CEILING_MULT","3.0")),
    }
    NOTIONAL_USD = float(os.getenv("NOTIONAL_USD","0"))
    TZ = ZoneInfo(cfg["TIMEZONE"])
    MAX_BARS = max(cfg["LOOKBACK_BARS"] + cfg["VOL_SMA_WINDOWS"] + 50, 600)

    state = load_state()
    ex = init_exchange()
    twilio_client, FROM, TO = init_twilio()
    bar_counter = 0

    print(f"Running simple-SMS alerts for {cfg['SYMBOL']} @ {cfg['TIMEFRAME']} on {ex.id}")

    while True:
        try:
            df = fetch_ohlcv(ex, cfg["SYMBOL"], cfg["TIMEFRAME"], limit=MAX_BARS)
            atr_series = compute_atr(df, cfg["ATR_PERIOD"])
            i=-2
            last_close=float(df['close'].iloc[i])
            last_ts=int(df['ts'].iloc[i].timestamp())
            vwmom, ret_L, vol_ratio = compute_vwmom(df, cfg["LOOKBACK_BARS"], cfg["VOL_SMA_WINDOWS"])
            if vwmom is None or np.isnan(atr_series.iloc[i]):
                time.sleep(cfg["POLL_SECONDS"]); continue

            rv_spike=None
            if os.getenv("EXIT_ON_VOL_SPIKE","false").lower()=="true":
                rv = realized_vol(df["close"], cfg["RV_WINDOW"])
                rv_med = None
                med_window = cfg["RV_WINDOW"]*20
                if len(df) >= med_window:
                    rets = df["close"].pct_change().dropna()
                    rv_med = float(rets.rolling(med_window).std().dropna().median() * np.sqrt(365*24*60))
                if rv and rv_med:
                    rv_spike = bool(rv > cfg["RV_CEILING_MULT"] * rv_med)

            cooldown = cfg["COOLDOWN_MINUTES"]*60
            now_epoch=int(time.time())
            can_entry_alert=(now_epoch - float(state.get("last_alert_ts",0))) >= cooldown

            pos_data = state.get("position")
            bar_counter += 1

            # ENTRY
            if pos_data is None:
                new_signal = state.get("last_signal","flat")
                side=None
                if vwmom > cfg["THRESH_UP"] and state.get("last_signal")!="long":
                    side="long"; new_signal="long"
                elif cfg["ENABLE_SHORT_ALERTS"].lower()=="true" and vwmom < -cfg["THRESH_DN"] and state.get("last_signal")!="short":
                    side="short"; new_signal="short"
                if side and can_entry_alert:
                    atr=float(atr_series.iloc[i])
                    pos=create_position(side,last_close,atr,cfg,last_ts,bar_counter)
                    body = msg_entry(side, cfg["SYMBOL"], cfg["TIMEFRAME"], last_close, pos.stop_price, pos.target_price, fmt_time(last_ts, TZ))
                    send_sms(twilio_client, FROM, TO, body)
                    state["position"]=asdict(pos); state["last_alert_ts"]=now_epoch; state["last_signal"]=new_signal; save_state(state)
                elif side and not can_entry_alert:
                    state["last_signal"]=new_signal; save_state(state)

            # EXIT
            else:
                pos=Position(**pos_data)
                pos=update_trailing_and_stops(pos, df, atr_series, cfg)
                reason, exit_price = check_exit(pos, df, vwmom, cfg, bar_counter, rv_spike)
                if reason:
                    if pos.side=="long": pnl_pct=(exit_price-pos.entry_price)/pos.entry_price
                    else: pnl_pct=(pos.entry_price-exit_price)/pos.entry_price
                    pnl_abs = NOTIONAL_USD * pnl_pct if NOTIONAL_USD>0 else None
                    body = msg_exit(reason, pos.side, cfg["SYMBOL"], cfg["TIMEFRAME"], exit_price, pnl_pct, pnl_abs, fmt_now(TZ))
                    send_sms(twilio_client, FROM, TO, body)
                    state["position"]=None; save_state(state)
                else:
                    state["position"]=asdict(pos); save_state(state)

        except Exception as e:
            print(f"[Loop] {e}")
        time.sleep(cfg["POLL_SECONDS"])

if __name__=="__main__":
    main()
