# Learn this before any interview

If you cannot explain a line, you cannot defend it. Read, close the file, say it out loud,
then change the code and see what happens.

## 1. How to answer "revenue fell, why?" (the interview version)
1. Is it real? Compare with normal variation, same weekdays, same length. (permutation test)
2. Split revenue = orders x average order value. Which one moved? (volume vs value)
3. Slice by dimensions: channel, device, country, category, new vs returning. Which slice carries the change?
4. Check intersections. Does one combination explain more than any single slice?
5. Check mix vs rate. Did a segment's own value change, or did the mix of orders shift?
6. Say what you would check next: a deploy, a campaign stopping, a price change, tracking breaking, a holiday.
That is exactly what the tool does, in that order.

## 2. The permutation test (`_perm_pvalue`)
Question: if nothing changed, how often would we see a change this big?
If nothing changed, "before" and "after" are just labels. For each of the 14 day pairs we may swap the two days.
That gives 2^14 = 16,384 equally likely worlds. p-value = share of those worlds with a change at least as big as ours.
Why not a plain bootstrap? We tried it first. It said "real" in 12% of no-change runs instead of 5%,
because 14 days is too few for the percentile bootstrap. The permutation test gave about 6%.

## 3. Contribution, EP and excess (`decompose.py`)
- Contribution = segment after minus segment before. For one dimension they add up to the total change.
- EP (explanatory power) = contribution / total change. Can exceed 100% when other segments offset it.
- Excess = contribution minus what the segment "should" have changed if it moved like the REST of the business.
Why not compare with the overall trend? If one segment collapses, the overall trend falls, and every healthy
segment looks like it outperformed. Comparing with the rest (leave one out) fixes most of that.
A big contributor is not always the cause: a large segment can have a big raw change just by being large.

## 4. Volume vs value (exact split)
revenue = orders x AOV.  r1 - r0 = (o1 - o0) x average AOV + (a1 - a0) x average orders.
The "average" of the two periods (midpoint) is what makes the split add up exactly, with no leftover term.

## 5. Mix vs rate (why AOV can fall with no price change)
AOV = sum over segments of (share of orders) x (segment AOV).
- Rate effect: a segment's own AOV changed.
- Mix effect: a segment's share of orders changed.
Example in the demo `mix_shift`: social orders rise, social orders are small, so AOV falls, yet no segment's AOV changed.
Interview point: this is the same mechanism as Simpson's paradox. Always check mix before blaming price.

## 6. Surprise (an idea from Adtributor)
Jensen-Shannon divergence compares the before shares with the after shares. If one channel suddenly holds
a very different share, that dimension is "surprising". We use it to rank dimensions.
Read the Adtributor paper (Microsoft Research, revenue debugging in advertising). Know its idea in 1 minute.

## 7. Why pairs, and why the concentration rule
If the real cause is "mobile AND paid search", neither "mobile" nor "paid search" alone is right.
But if you search all pairs, half of any broken segment looks like a pair cause by chance.
So a pair is reported only if it holds most of each parent's movement while being a much smaller slice.

## 8. Honesty points that impress interviewers
- A segment that moved is a lead, not proof of causation. Check logs, releases, campaigns.
- Not every change is real. The tool says "do not claim a cause" when the change is within normal variation.
- The naive "biggest contributor" method scores 0% on intersection causes (see the study).

## 9. Questions to practice
- Revenue fell 8%. Walk me through your analysis.
- How do you know it is not just a bad week?
- AOV fell but no category got cheaper. How is that possible?
- Why compare a segment with the rest of the business instead of the overall trend?
- What would you do if two segments fell at the same time?
- What does a permutation test assume? When would it break? (a sale or holiday inside one window)
- What would you add next? (sessions and conversion funnel, triples, a forecast baseline instead of "before")
