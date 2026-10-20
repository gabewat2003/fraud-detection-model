# Memo: a cost-based approve / review / decline policy for card-not-present fraud

**To:** Fraud operations lead · **From:** <!-- DRAFT: your name --> · **Re:** recommended decision policy

<!-- DRAFT: this memo is written for a non-technical reader. Rewrite it in your voice after `make all`;
     the table below is regenerated from the pipeline by `make writeup`. -->

## Recommendation

Replace the fixed "decline the riskiest 1% of transactions" rule with a **three-tier policy**: approve, send to
manual review, or decline, where the decline cutoff depends on the amount at stake and manual review is used only
where it saves more than it costs, within the team's daily capacity.

## What it would have done on the most recent ~4 weeks of data (held-out test window)

<!-- BEGIN:memo_numbers -->
| | Approve all | Rules engine | Recommended policy |
|---|---|---|---|
| Total cost (test window) | $515,854 | $518,517 | $266,532 |
| Fraud $ stopped | $0 | $3,441 | $263,015 |
| Good customers declined | 0 | 0 | 1,206 (1.41%) |
| Manual reviews | 0 | 2,320 | 907 |
| Top-1%-decline rule, for reference | | | $428,738 |
<!-- END:memo_numbers -->

## Why it works

- **Reviews go where they pay off.** Most of the saving comes from sending about 1% of traffic, the uncertain,
  high-value cases, to an analyst instead of guessing. Automatic declines alone save about half as much.
- **Declines are priced, not quota-based.** A transaction is declined only when its expected fraud loss is larger
  than what a wrong decline costs: the support contact plus the margin on the lost sale.
- **Every threshold was set on earlier data** and then tested, unchanged, on later data the model never saw.

## What it depends on

- **The cost of a wrongly declined customer.** We assumed $5 (support contact, churn risk) plus 25% of the order
  (lost margin). An earlier version without the margin term declined 15% of good customers, so this assumption
  matters a lot; Finance and CX should agree on it. See the sensitivity chart.
- **Review capacity.** Sized at 1% of daily volume. On some days the queue exceeds that; staffing or an
  overflow rule (auto-approve below a small amount) is needed.
- **Calibrated scores.** The policy multiplies probabilities by dollars, so the model's probabilities are checked
  against observed fraud rates and should be re-checked monthly.

## Risks and next steps

1. **Pilot in shadow mode** for 2-4 weeks: score live traffic, log decisions, change nothing; compare.
2. **Monitor drift** (feature and score stability report) and the daily review queue; re-fit thresholds monthly.
3. **Watch who carries the friction.** The model's false alarms cluster in product line C: about half of the
   highest-scored legitimate orders are product C, against under 10% of normal traffic. Track decline, complaint
   and repeat-purchase rates for product C customers during the pilot, and consider a separate threshold for it.
4. **Close the label loop.** Chargebacks arrive weeks late; reviewer outcomes should feed back as faster labels.
