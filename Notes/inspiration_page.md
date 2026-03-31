# Virtù and Fortuna
## Inspiration

> When something works, how do we know it wasn’t just luck?

At first, I wasn’t trying to build a research project. I was trying to understand something that didn’t make sense.

I would see people online talk about trading strategies like they had cracked a code. A chart goes up, and suddenly it’s called “skill.” A strategy works for a few months, and it becomes a system people believe in. But something about that felt off to me. If markets are uncertain, how do we know when someone is actually good, and when they just got lucky?

That question stayed with me, but I didn’t have a way to answer it until I came across an idea outside of finance.

While reading about Niccolò Machiavelli, I found his distinction between *virtù* and *fortuna*. *Virtù* represented skill, control, and intentional action. *Fortuna* represented randomness, chaos, and luck. What stood out to me was that Machiavelli didn’t treat success as proof of ability. He understood that outcomes are often shaped by forces outside of control.

That idea felt directly applicable to markets.

A trader could succeed because of *virtù*, real decision-making skill, or because of *fortuna*, randomness that just happened to go their way. The problem is that both can look identical if you only look at the result.

That became the core idea of my project.

I set out to build a system that doesn’t just measure performance, but questions it. Instead of asking “Did this strategy make money?”, I asked, “Was that outcome actually meaningful, or could it have happened by chance?” To do this, I constructed a framework where each strategy is tested against thousands of randomized versions of itself, using Monte Carlo simulation to create a baseline of what luck alone would produce.

As I developed the model, another layer became clear. Performance isn’t constant. Strategies behave differently depending on market conditions. What looks like skill in one environment may disappear in another. This led me to incorporate regime-based analysis, separating markets into calm, neutral, and stressed periods to understand when, not just if, a strategy works.

The result is a system that tries to separate *virtù* from *fortuna* in a quantitative way. It doesn’t assume that success means skill. Instead, it asks whether that success stands out against randomness, and whether it holds under different conditions.

In a space where outcomes are often taken at face value, this project is an attempt to step back and ask a deeper question: when something works, how do we know it wasn’t just luck?
