# generate_option_strategies.py
# Unified scanner: Calls, Puts, Vertical Debit Spreads → filters by Greeks → CSV + top‑candidates + payoff plot

import numpy as np
import pandas as pd
import yfinance as yf
from scipy.stats import norm
from datetime import datetime
import matplotlib.pyplot as plt

# Black-Scholes call/put pricing
def bs_price(S, K, T, r, sigma, option='call'):
    if T <= 0 or sigma <= 0:
        return max((S - K) if option=='call' else (K - S), 0.0)
    d1 = (np.log(S/K) + (r + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
    d2 = d1 - sigma*np.sqrt(T)
    if option=='call':
        return S*norm.cdf(d1) - K*np.exp(-r*T)*norm.cdf(d2)
    else:
        return K*np.exp(-r*T)*norm.cdf(-d2) - S*norm.cdf(-d1)

if __name__=='__main__':
    # === CONFIG ===
    TICKER    = 'CPRT'
    RFR       = 0.05      # risk-free rate
    STRIKE_PCT= (0.90,1.20)
    GREEKS    = dict(delta=(0.2,0.35), theta=(-0.10,None), vega=(None, None))
    SPREAD_WIDTH_PCT = 0.05  # 5% width for debit spreads
    LOOKBACK_SPOT_DAYS = 1
    
    tk = yf.Ticker(TICKER)
    # get spot
    hist = tk.history(period=f"{LOOKBACK_SPOT_DAYS}d")
    if hist.empty: raise SystemExit(f"[ERROR] No spot data for {TICKER}")
    spot = hist['Close'].iloc[-1]

    # expiries
    today = datetime.utcnow().date()
    expiries = [e for e in tk.options if (pd.to_datetime(e).date()>today)]
    if not expiries: raise SystemExit("[ERROR] No future expirations")

    results=[]
    for exp in expiries:
        T_days = (pd.to_datetime(exp).date() - today).days
        T = T_days/365.0
        chain = tk.option_chain(exp)
        for kind, df_chain in [('call',chain.calls), ('put',chain.puts)]:
            # filter strikes
            valid = df_chain[(df_chain.strike>=spot*STRIKE_PCT[0]) & (df_chain.strike<=spot*STRIKE_PCT[1])]
            for _, row in valid.iterrows():
                K = row.strike
                # premiums: lastPrice > bid/ask mid > BS fallback
                lp = float(getattr(row,'lastPrice',np.nan) or np.nan)
                bid= float(getattr(row,'bid',np.nan) or np.nan)
                ask= float(getattr(row,'ask',np.nan) or np.nan)
                iv = float(getattr(row,'impliedVolatility',np.nan) or np.nan)
                if lp>0:
                    prem=lp
                elif bid>0 and ask>=bid:
                    prem=(bid+ask)/2
                else:
                    prem=bs_price(spot,K,T,RFR,iv,option=kind)
                # Greeks from row or estimate
                delta = getattr(row,'delta',None)
                theta = getattr(row,'theta',None)
                vega  = getattr(row,'vega',None)
                # apply Greek filters
                ok=True
                lo,hi=GREEKS['delta'];
                if delta is None or not (delta>=lo and (hi is None or delta<=hi)): ok=False
                lo,hi=GREEKS['theta'];
                if theta is None or not (theta>=lo if lo is not None else True): ok=False
                lo,hi=GREEKS['vega'];
                if vega  is None or not (vega>=lo if lo is not None else True): ok=False
                if not ok: continue
                # payoff
                prices=np.linspace(0.5*spot,1.5*spot,200)
                if kind=='call': payoff=(np.maximum(prices-K,0)-prem)*100
                else: payoff=(np.maximum(K-prices,0)-prem)*100
                results.append({
                    'ticker':TICKER,'strategy':kind,'expiry':exp,'strike':K,'premium':round(prem,2),
                    'iv':round(iv*100,2) if iv>0 else None,'delta':delta,'theta':theta,'vega':vega,
                    'days_to_expiry':T_days,'max_payoff':round(float(payoff.max()),2),
                    'breakeven':round((K+prem) if kind=='call' else (K-prem),2)
                })
        # debit vertical spreads
        calls = chain.calls
        long_calls = calls[(calls.strike>=spot*STRIKE_PCT[0])&(calls.strike<=spot*STRIKE_PCT[1])]
        for _, long in long_calls.iterrows():
            K1 = long.strike
            K2 = K1*(1+SPREAD_WIDTH_PCT)
            short = calls[calls.strike==K2]
            if short.empty: continue
            short=short.iloc[0]
            # price legs
            def leg_price(r):
                lp=getattr(r,'lastPrice',0) or 0
                bid=getattr(r,'bid',0) or 0; ask=getattr(r,'ask',0) or 0
                if lp>0: return lp
                if bid>0 and ask>=bid: return (bid+ask)/2
                return bs_price(spot, r.strike, T, RFR, getattr(r,'impliedVolatility',0),'call')
            p1=leg_price(long); p2=leg_price(short)
            net = p1 - p2
            prices=np.linspace(0.5*spot,1.5*spot,200)
            payoff = ((np.maximum(prices-K1,0)-np.maximum(prices-K2,0))-net)*100
            results.append({
                'ticker':TICKER,'strategy':'debit_spread','expiry':exp,'strike':f"{K1}->{K2}",
                'premium':round(net,2),'iv':None,'delta':None,'theta':None,'vega':None,
                'days_to_expiry':T_days,'max_payoff':round(float(payoff.max()),2),
                'breakeven':round(K1+net,2)
            })

    df=pd.DataFrame(results)
    out='../output/option_strategies.csv'
    df.to_csv(out,index=False)
    print(f"✅ Saved {len(df)} strategies to {out}")
    if df.empty:
        print("⚠️ No candidates after filtering.")
        exit(0)

    # show top 5
    top = df.sort_values('max_payoff',ascending=False).head(5)
    print("\n🥇 Top 5 Strategies:")
    print(top.to_string(index=False))

    # plot payoff for best
    best = top.iloc[0]
    kind,exp=best.strategy,best.expiry
    strike = best.strike
    if kind=='debit_spread':
        K1,K2=map(float,strike.split('->'))
        net = best.premium
        prices=np.linspace(0.5*spot,1.5*spot,200)
        payoff=((np.maximum(prices-K1,0)-np.maximum(prices-K2,0))-net)*100
        label=f"{TICKER} {K1}->{K2} Debit"
        title=f"Top: {TICKER} Debit Spread {K1}->{K2} Exp {exp}\nPrem: ${net}, BE: {best.breakeven}"
    else:
        K=float(strike)
        prem=best.premium
        if kind=='call': payoff=(np.maximum(prices-K,0)-prem)*100
        else: payoff=(np.maximum(K-prices,0)-prem)*100
        label=f"{kind.upper()} {TICKER} {int(K)}"
        title=f"Top: {TICKER} {kind.title()} {K} Exp {exp}\nPrem: ${prem}, BE: {best.breakeven}"
    plt.figure(figsize=(8,5))
    plt.plot(prices,payoff,label=label)
    plt.axhline(0,color='black',lw=1)
    plt.title(title)
    plt.xlabel('Underlying Price @ Expiry')
    plt.ylabel('P/L per Contract ($)')
    plt.grid(ls='--',alpha=0.6)
    plt.legend()
    plt.tight_layout()
    plt.show()
