# Memo: a cost-based approve / review / decline policy for card-not-present fraud

**To:** Fraud operations lead · **Re:** recommended decision policy

<!-- the table below is regenerated from the pipeline by `make writeup` -->

## Recommendation

Replace the fixed "decline the riskiest 1% of transactions" rule with a **three-tier policy**: approve, send to
manual review, or decline, where the decline cutoff depends on the amount at stake and manual review is used only
where it saves more than it costs, within the team's daily capacity.

## What it would have done on the most recent ~4 weeks of data (held-out test window)

<!-- BEGIN:memo_numbers -->
| | Approve all | Rules engine | Recommended policy |
|---|---|---|---|
| Total cost (test window) | $515,854 | $525,286 | $299,562 |
| Fraud $ stopped | $0 | $3,097 | $248,274 |
| Good customers declined | 0 | 226 | 1,277 (1.49%) |
| Manual reviews | 0 | 2,320 | 926 |
| Top-1%-decline rule, for reference | | | $428,738 |
<!-- END:memo_numbers -->

## Why it works

- **Reviews go where they pay off.** About 1% of traffic, the uncertain, high-value cases, goes to an analyst
  instead of being guessed. Automatic approve/decline alone cuts cost by about 27%; the review tier takes it to
  about 42%, assuming analysts get 90% of cases right.
- **Declines are priced, not quota-based.** A transaction is declined only when its expected fraud loss is larger
  than what a wrong decline costs: the support contact plus the margin on the lost sale.
- **Every threshold was set on earlier data** and then tested, unchanged, on later data the model never saw.

## What it depends on

- **The cost of a wrongly declined customer.** We assumed $5 (support contact, churn risk) plus 25% of the order
  (lost margin). An earlier version without the margin term declined 15% of good customers, so this assumption
  matters a lot; Finance and CX should agree on it. See the sensitivity chart.
- **Analyst accuracy.** The review tier is only as good as the analysts. At 80% accuracy the saving falls to
  about 36%; if they were always right it would be about 48%. Measuring reviewer accuracy during the pilot is
  the cheapest way to firm up this number.
- **Review capacity.** Sized at 1% of daily volume. The figures above let the queue run over, which happens on about
  half the days. Run live, the policy should add a hard daily cap: once the day's slots are full, remaining cases
  are approved or declined automatically. On the test window that costs about 2 points of saving.
- **Calibrated scores.** The policy multiplies probabilities by dollars, so the model's probabilities are checked
  against observed fraud rates and should be re-checked monthly.

## Risks and next steps

1. **Pilot in shadow mode** for 2-4 weeks: score live traffic, log decisions, change nothing; compare.
2. **Monitor drift** (feature and score stability report) and the daily review queue; re-fit thresholds monthly.
3. **Watch who carries the friction.** The model's false alarms cluster in product line C: about half of the
   highest-scored legitimate orders are product C, against under 10% of normal traffic. Track decline, complaint
   and repeat-purchase rates for product C customers during the pilot, and consider a separate threshold for it.
4. **Close the label loop.** Chargebacks arrive weeks late; reviewer outcomes should feed back as faster labels.
