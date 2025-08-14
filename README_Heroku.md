# Heroku Deploy — BTC VWMOM SMS Alert Bot (alerts-only)

## Files
- `btc_vwmom_sms_alert_full_simple.py` — the bot with simple SMS format
- `requirements.txt` — Python dependencies
- `Procfile` — runs the bot as a Heroku worker
- `runtime.txt` — selects Python version

## Quick deploy
```bash
# in an empty folder locally
unzip Heroku_VWMOM_Bot.zip
git init
heroku create vwmom-bot    # or any name you like
heroku git:remote -a vwmom-bot
git add .
git commit -m "init"
git push heroku main

# set only the essentials (others use built-in defaults)
heroku config:set TWILIO_ACCOUNT_SID=ACxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
heroku config:set TWILIO_AUTH_TOKEN=your_auth_token_here
heroku config:set TWILIO_FROM_NUMBER=+1234567890
heroku config:set TWILIO_TO_NUMBER=+4512345678

# optional: show $ PnL in exit SMS
heroku config:set NOTIONAL_USD=10000

# start the worker
heroku ps:scale worker=1
heroku logs --tail
```
