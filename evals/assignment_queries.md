# How the assistant answers the assignment's sample queries

_This file gets overwritten by `python -m evals.capture_assignment_responses`._

The four queries from the assignment brief, run verbatim against the default
student (Arjun, grade 10 CBSE):

1. `I am weak in Algebra. What should I do next?`
2. `What should I study this week?`
3. `Which topic should I prioritize first?`
4. `I have a Maths test coming up. Help me prepare.`

Run the capture script with your `ANTHROPIC_API_KEY` set:

```bash
python -m evals.capture_assignment_responses                  # against S123 (Arjun)
python -m evals.capture_assignment_responses --student S128   # against Neha (JEE)
```

Each section of the captured output shows the verbatim query, the tools the
agent chose to call and with what inputs, the `material_id` citations it
surfaced, token/latency usage, and the final streamed answer. Nothing is hand-
edited — re-run to refresh.

---

<!-- the capture script will replace everything below this line on run -->
