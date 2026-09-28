You are describing one figure cut from a technical PDF so that people can find it by searching text, and so that its numbers can be charted later. Describe only what is visible in the image. The page context below helps you name things correctly, but never add facts that the image does not show.

## Where the figure comes from
- Document: {source_file}, page {page}
- Section: {section_path}
- Caption printed in the PDF: {pdf_caption}
- Text just before the figure: {context_before}
- Text just after the figure: {context_after}
- Sentences that refer to this figure: {figure_refs}

## Step 1: decide the figure type
Choose exactly one `figure_type`:
- `diagram`: boxes, blocks or components connected by arrows or lines (architectures, pipelines, model blocks).
- `flowchart`: a process with decisions or ordered steps.
- `chart`: bar, line, pie or scatter chart with values.
- `table_image`: a table shown as an image.
- `screenshot`: a captured screen, slide or user interface.
- `photo`: a photograph.
- `equation`: mostly a mathematical formula.
- `decorative`: a logo, icon or ornament with no information.

## Step 2: follow the instructions for that type
- **diagram / flowchart**: in `detailed_description`, name every box or component using its exact label, then describe every arrow as "A → B" with its label if it has one, and finish with what the whole flow does. Mention colours only when they carry meaning (for example "green = allowed").
- **chart**: give the chart kind, axis labels and units, and each series. Put the values in `extracted_data.chart`, one point per bar or marked point:
  - A point whose number is printed on the figure (on the bar, in a label or legend): copy that number and use `"flag": "exact"`.
  - A point with no number printed next to it: read its value off the axis scale and use `"flag": "estimated"`. If the chart has no numeric axis to read from, leave the point out. Never copy a neighbour's number and never mark a point `exact` unless its own number is printed.
  - `value` is always a plain number such as 41 or 0.85 (a percentage as 41, with the unit in `unit`). Never write words such as "high" or "low", and never `null`. If a series is only qualitative, with no numbers and no scale, leave `extracted_data` as `null` and describe it in words.
- **table_image**: transcribe the table into `extracted_data.table` with its column headers and every row, cell by cell, as printed. Use an empty string for an empty cell.
- **screenshot / photo / equation**: describe what is shown and copy any readable text.
- **decorative**: one short sentence is enough.

## Fields
- `short_caption`: one line (under 15 words) a reader would see under the image.
- `detailed_description`: the full description from step 2, in plain sentences. This is what search will match, so use the figure's own terms.
- `visible_text`: every distinct piece of text you can read in the image, each label as its own string, exactly as written. List a label once even if it appears many times (for example the row and column labels of a grid); at most 80 items.
- `extracted_data`: `{{"chart": ...}}` for charts, `{{"table": ...}}` for table images, otherwise `null`.
- `keywords`: 3 to 10 search terms, including the technical terms the figure shows.
- `confidence`: `high` if every label was readable and the structure is clear; `medium` if some text was hard to read; `low` if you had to guess.

## Output
Reply with one JSON object and nothing else: no Markdown fences, no comments. It must match this schema:

{schema}
