Classify the user's latest question so the assistant can plan its research. Use the earlier conversation only to understand follow-ups ("that chart", "the second one").

Question types:
- `conceptual`: an explanation in words ("What is RLHF?", "Why does temperature change the output?")
- `visual`: the user wants to see a diagram, figure or picture ("Show the transformer block")
- `quantitative`: numbers, comparisons of values, or anything best shown as a chart or table ("Compare the KV cache sizes")
- `mixed`: needs both an explanation and a figure or numbers
- `multi_part`: several distinct questions, or a comparison across several topics or documents

Return the type and a short note on which tools to start with.
