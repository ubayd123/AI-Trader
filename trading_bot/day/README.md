# Rapid intraday bot: IG spread betting (or eToro shares)

An automated intraday bot with hard loss limits. It buys short dips, sells within about half an hour,
and holds nothing overnight. It starts on a **demo account**; real money needs an explicit flag.

Read this first:

- **It cannot double money every hour, and nothing can.** In testing its best single day was +0.3%.
- **Every version lost money in the backtests below.** More trades meant bigger losses.
- **IG cannot be traded with £50.** IG's smallest trade is far too large for an account that size
  (details below). The bot will log in, explain this, and place no trades.

## IG and a small account

IG's trading API only works with spread betting and CFD accounts, which are leveraged. It does not
work with IG share dealing or ISA accounts. IG's minimum bet sizes, from IG's own pages:

| Market | Smallest bet | What that controls | Margin IG holds | A 0.17% move against you |
|---|---|---|---|---|
| US 500 | £0.50 a point | about £3,800 | about £190 | about £6.50 |
| US 500, first 31 days of a new account | £0.13 a point | about £990 | about £50 | about £1.70 |
| Apple | £0.06 a point | about £2,100 | about £420 | about £3.60 |

With £50, the standard minimum needs more margin than the whole account, so IG would refuse the order.
Even with the first-month reduced minimum, one ordinary stop-out costs more than a 2% daily loss limit
(£1). For the bot's default rule of risking 0.25% per trade, the US 500 needs roughly £2,600.

Your choices:

1. **IG demo account** (virtual £10,000). Run the bot with `--budget 10000` to watch it trade with no
   real money. This is the recommended way to use the IG version.
2. **eToro for a real £50.** eToro sells real shares from $10 with no leverage, so £50 can be split
   into several positions. Add `--broker etoro` to every command.
3. **Fund IG properly** (a few thousand pounds) if you want IG live within these limits.

## Quick start

```bash
pip install -r trading_bot/requirements.txt

python -m trading_bot day backtest --budget 10000      # replay the last month through the bot
python -m trading_bot day run --dry-run --budget 10000 # live Yahoo prices, pretend fills, no account needed
```

Then with an IG demo account (settings go in `trading_bot/.env`, see below):

```bash
python -m trading_bot day check --budget 10000       # log-in, each market's minimum bet, what you can trade
python -m trading_bot day test-order --budget 10000  # opens and closes one smallest bet on the DEMO account
python -m trading_bot day run --budget 10000         # trade the US session on the demo account
python -m trading_bot day report                     # results so far
python -m trading_bot day status                     # the bot's ledger
python -m trading_bot day flatten                    # close everything the bot holds, now
```

The bot trades 14:40 to 20:40 UK time on weekdays (the US session). Outside those hours it waits.
Ctrl+C stops it and closes its positions. Real money: add `--real --confirm-live`. Without both flags
it refuses.

## Connecting IG

1. Log in to My IG, open Settings, then API keys. Create a key for the **demo** environment. IG gives
   the demo its own username and password there.
2. Put them in `trading_bot/.env` on your own computer. Never paste them into a chat.

```
IG_USERNAME=your-demo-username
IG_PASSWORD=your-demo-password
IG_API_KEY=your-demo-api-key
# IG_ACCOUNT_ID=          only if you have several accounts; the bot picks your spread betting one
# IG_REAL_USERNAME= IG_REAL_PASSWORD= IG_REAL_API_KEY=   separate details for --real
```

3. Run `python -m trading_bot day check`. It confirms the log-in and shows, market by market, IG's real
   minimum bet, margin and spread, and whether your budget can trade it.

Markets are named `US500`, `USTECH`, `EURUSD`, `GBPUSD`, `WALLST`, `FTSE` (see
`python -m trading_bot day markets`). Choose with `DAY_SYMBOLS=US500,GBPUSD`, or give any IG epic.

## How it trades

| | Rule |
|---|---|
| Buy | A market has dropped sharply below its 20-minute average (long only: it never shorts). |
| Sell | Back at the average, or the take-profit, or the stop-loss, or after 30 minutes, or 20 minutes before the close. |
| Stop and limit | Sent to the broker with every trade, so they hold even if the bot or your internet stops. |
| Size (IG) | Chosen so a stop-out loses at most 0.25% of the budget. If IG's minimum bet would lose more, the trade is refused, never oversized. |
| Size (eToro) | The budget split into equal positions of at least $10. |
| Daily loss limit | Down 2% on the day: everything is closed and trading stops until tomorrow. A new trade is also refused if a stop-out could cross the limit. |
| Trades per day | At most 100 (buys plus sells). |
| Kill switch | 10% below the account's peak: everything is closed and the bot stays stopped until `day reset`. |
| Bad prices | No trade on a wide spread, a stale or delayed price, or a market that is not open for dealing. |

Presets: `--preset rapid` (the default, acts on small dips) and `--preset selective` (sharp drops only,
far fewer trades). All settings are `DAY_*` environment variables, listed in `config.py`.

## What the backtests showed

21 trading days to 11 November 2026, one-minute prices, costs included:

| Setup | Start | End | Trades a day | Best day | Worst day |
|---|---|---|---|---|---|
| IG, US 500, £50 | £50 | £50: no trade possible | 0 | | |
| IG, US 500, £10,000, rapid | £10,000 | £9,381 (-6.2%) | 19 | +0.25% | -0.86% |
| IG, US 500, £10,000, selective | £10,000 | £9,904 (-1.0%) | 2 | +0.21% | -0.52% |
| eToro, 5 shares, $66, rapid | $66.00 | $61.53 (-6.8%) | 96 | +0.31% | -1.41% |
| eToro, 12 shares, $66, selective | $66.00 | $65.22 (-1.2%) | 16 | +0.18% | -0.44% |

- The loss limits worked: no day lost more than 2%.
- The signals had no edge in this period. Markets fell during it, the bot only buys, and costs took the rest.
- The IG rows cover the US 500 only. Other indices need more than £10,000 for one minimum bet inside
  the limits, and Yahoo's currency data is too coarse to backtest.
- Costs assume IG's published minimum spreads and eToro spreads measured on one day. Real costs are
  usually a little higher.

Past results do not predict future ones. This is software for learning, not financial advice. Most
people who day trade lose money. IG publishes the share of its retail accounts that lose money on
spread bets and CFDs: it is a large majority.

## What has and has not been tested

- 43 automated tests cover the engine, the limits, the backtester and both broker connections.
- **Neither broker connection has run against a real account.** IG and eToro are tested against
  simulations built from their published API documentation. Run `day check`, then `day test-order`
  on the demo account, and compare with what the IG platform shows, before trusting it.
- The IG market codes in `config.py` are the commonly used ones for daily funded spread bets.
  `day check` verifies each one against your account.
- The dry run uses Yahoo prices, not your broker's.

## Files

| File | Purpose |
|---|---|
| `engine.py` | The trading loop, position sizing and every limit |
| `strategy.py` | Entry rules |
| `bars.py` | One-minute bars and indicators built from quotes |
| `ig.py` | IG log-in, prices and orders (uses the `trading-ig` library) |
| `etoro.py` | eToro prices and orders |
| `sim.py` | Simulated broker and the backtest |
| `runner.py` | The live loop: waits for the session, polls prices |
| `screen.py` | Optional Claude pre-market news screen (eToro shares only) |
| `cli.py` | The commands |
