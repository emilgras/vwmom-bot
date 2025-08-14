# Heroku Deploy — BTC VWMOM SMS Alert Pro Bot

This upgraded bot includes:
- Multi-horizon VWMOM (8h, 24h, 3d) with majority voting
- Cost-aware entry thresholds
- Volatility-managed sizing & risk caps
- Order Flow Imbalance (OFI) + spread execution filter
- Sharpened exits (ATR-based, trailing stops, breakeven)
- Pyramiding winners up to 3 legs

## Deployment (GitHub + Heroku dashboard)
1. Replace your existing bot files with these in your local repo.
2. Commit and push to GitHub.
3. In Heroku Dashboard:
   - Deploy tab → Manual Deploy → select branch → Deploy Branch
   - Settings → Reveal Config Vars → add TWILIO_... keys, NOTIONAL_USD (optional), etc.
   - Resources tab → turn on Worker dyno.
4. Check logs to confirm bot is running.
