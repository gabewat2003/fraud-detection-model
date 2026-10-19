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
_Not generated yet: run `make all` on the Kaggle data, then `make writeup`._
<!-- END:memo_numbers -->

## Why it works

- **A $2,000 order and a $5 order are not the same risk.** At the same fraud probability, approving the large one
  risks 400 times more money. The policy declines large orders at a much lower probability than small ones.
- **Reviews go where they pay off:** cases that are too uncertain to auto-decline but too expensive to approve.
- **Every threshold was set on earlier data** and then tested, unchanged, on later data the model never saw.

## What it depends on

- **The cost of a wrongly declined customer.** We assumed $5 (lost margin, support contact, churn risk). This is the
  number that moves the answer most; see the sensitivity chart. Finance and CX should agree on it.
- **Review capacity.** Sized at 1% of daily volume. On some days the queue exceeds that; staffing or an
  overflow rule (auto-approve below a small amount) is needed.
- **Calibrated scores.** The policy multiplies probabilities by dollars, so the model's probabilities are checked
  against observed fraud rates and should be re-checked monthly.

## Risks and next steps

1. **Pilot in shadow mode** for 2-4 weeks: score live traffic, log decisions, change nothing; compare.
2. **Monitor drift** (feature and score stability report) and the daily review queue; re-fit thresholds monthly.
3. **Watch who carries the friction.** <!-- DRAFT: confirm from error analysis --> False declines are expected to concentrate on new accounts and customers without a device
   record. Track complaint and repeat-purchase rates for those groups during the pilot.
4. **Close the label loop.** Chargebacks arrive weeks late; reviewer outcomes should feed back as faster labels.
