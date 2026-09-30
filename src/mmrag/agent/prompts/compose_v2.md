Now write the final answer from the evidence gathered above, as ordered blocks:

- `text`: markdown explanation. Cite every claim with the ids of the passages, figures or tables it comes from.
- `image`: an original figure worth showing, by its element id, with a one-line caption.
- `chart`: a chart you created with `make_chart`, by its `chart_id`. Only charts that were accepted.
- `table`: a table from the documents, by its element id, when the user benefits from seeing its rows.

Guidance:
- Lead with a direct answer in a text block, then the figure, chart or table that supports it.
- Visual questions should include the most relevant original figure. Quantitative questions should include a chart or table when the evidence has the numbers.
- Keep text focused: a few short paragraphs at most.
- Be complete: answer every part of the question, and include each key point the evidence gives for it (the definitions, roles, steps or numbers it states), not just the first one.
- Put citations only in the block's `citations` list. Never write ids, "(id: ...)", file names or page numbers in the markdown; they are shown from the citations.
- Quote numbers exactly as the evidence prints them (for example "33.5 GB to 6.7 GB"). A ratio or difference ("5x smaller") must come from a `compute` result; never estimate one yourself.
- If the documents do not answer the question, return one text block saying so, with `not_found` true and `missing` naming what was not found. Do not guess.

{limit_note}
