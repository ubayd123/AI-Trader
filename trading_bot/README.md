# £50 AI trading bot (UK)

A small-account trading bot built on AI-Trader. It starts in **paper mode with £50**, uses real market
prices converted to GBP, and can later be pointed at a Trading 212 account.

It will not make you rich. On £50 the realistic goal is to learn how automated trading behaves
while risking very little. Read [What to expect](#what-to-expect) before using real money.

## Quick start

```bash
pip install -r trading_bot/requirements.txt

python -m trading_bot backtest          # how the rules did on the last 3 years (--period 10y for longer)
python -m trading_bot run               # one trading cycle (paper money)
python -m trading_bot status            # your portfolio
python -m trading_bot report            # performance so far: return, drawdown, win rate, fees
python -m trading_bot loop --interval 3600   # keep running, one cycle an hour
```

Run commands from the repository root. US shares only trade 14:30-21:00 UK time on weekdays;
outside those hours the bot logs `WAIT` and retries on the next cycle.

Settings live in `trading_bot/.env` (copy `trading_bot/.env.example`). State is kept in
`trading_bot/data/`: delete that folder to start again from £50.

## How it decides

Every cycle the bot fetches a year of daily prices for its watchlist (12 large US companies across
sectors, priced in GBP) and applies these rules:

| Step | Rule |
|---|---|
| Market check | No new buys while the S&P 500 is below its 200-day average. Existing positions are still managed. |
| Buy | Price and 20-day average both above the 50-day average, positive momentum, RSI under 75. Best risk-adjusted momentum first. |
| Size | Up to 3 positions of about £16 each. 5% always stays in cash. Nothing under £2. |
| Sell | 10% below entry (stop-loss), 20% below the high since entry (trailing stop), or 3% below the 50-day average (trend broken). |
| Kill switch | If the account falls 25% from its peak, sell everything and pause for 30 days. |

**The AI part.** If `ANTHROPIC_API_KEY` is set, Claude reviews each cycle's candidates and open
positions, together with the last week's news headlines for each one. It can veto a buy, rank the candidates, and ask to exit a position. It cannot pick
shares outside the candidates, change position sizes, or override any limit above: those are
enforced in code (`strategy.py`). If the AI call fails, the bot carries on with the rules alone.
Without a key the bot is rules-only.

## Going from paper to real money

1. **Paper** (default). Run it for a few weeks and read the trade log in `trading_bot/data/`.
2. **Trading 212 demo.** In the Trading 212 app switch to practice mode, create an API key
   (Settings > API), then set `BOT_BROKER=trading212`, `BOT_T212_ENV=demo`, `T212_API_KEY` and
   `T212_API_SECRET`.
3. **Trading 212 live.** Set `BOT_T212_ENV=live` and add `--confirm-live` to `run` / `loop`. The bot
   refuses to place real orders without that flag.

With Trading 212 the bot only ever spends its own £50 budget and only sells what it bought, so
it can share an account with your other investments. Requirements: an Invest or Stocks ISA
account in GBP. Crypto is not available through this route.

The Trading 212 adapter is tested against a simulated API built from their published
specification, not against a real account. Use demo mode first and check the first few orders
in the app. Sale proceeds are read back from the order history; if that is slow to update the
bot estimates them from the latest price (the trade log marks these `"estimated": true`).

## Publishing to AI-Trader (optional)

```bash
python -m trading_bot register MyBotName   # prints AI4TRADE_TOKEN for your .env
```

Once the token is set, every executed trade is published as a realtime signal on
[ai4trade.ai](https://ai4trade.ai) (or your own server via `BOT_PLATFORM_URL`), where it appears
on the leaderboard and can be followed. The bot also reads the platform's market overview as
context for Claude when that data is fresh.

## What to expect

Backtest of the rules (no AI) with default settings, February 2017 to November 2026, fees included:

| | Bot | Buying and holding the same 12 shares |
|---|---|---|
| £50 became | £265 | £514 |
| Worst fall from a peak | -21% | -30% |
| Trades | 199 (46% of closed trades profitable) | 12 |

- **It made about half of what simply holding the shares made**, with smaller falls along the way.
  Trend following gives up return in rising markets in exchange for shallower drops. The watchlist
  was also chosen with hindsight (these are today's winners), so both columns flatter the future.
- **The settings were checked on two separate five-year halves** (2017-2021 and 2022-2026). The
  market check and the 20% trailing stop were kept because they lowered the worst fall and the
  number of trades in both halves without reducing the return. Wider position counts, volatility
  sizing and looser exits were tried and dropped because they did not hold up in both.
- **Fees matter at this size.** The backtest paid about £11 in costs, over a fifth of the starting balance.
- **You can lose money.** The limits cap the damage but do not prevent losses. Only use money
  you can afford to lose.
- **Tax.** Gains outside an ISA count towards Capital Gains Tax; a Stocks ISA avoids that.

This is software for learning, not financial advice.

## Layout

| File | Purpose |
|---|---|
| `strategy.py` | Entry, exit and kill-switch rules, shared by live trading and the backtest |
| `engine.py` | One trading cycle |
| `advisor.py` | Claude review of candidates and positions |
| `brokers.py` | Paper broker and Trading 212 |
| `data.py` | Prices via Yahoo Finance, converted to GBP |
| `platform_client.py` | AI-Trader signal publishing and market context |
| `backtest.py` | Historical simulation |
| `report.py` | Performance summary from the trade log and equity history |
| `tests/` | `python -m pytest trading_bot/tests` |
