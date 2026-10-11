-- Phase 10: planet placements read off South Indian horoscope charts (`mmrag charts read`), one row
-- per body per chart. "Charts with Saturn in the 7th house" is then an exact query; ranking search
-- documents by meaning could not tell a 7th-house Saturn from an 8th-house one.

CREATE TABLE chart_placements (
    element_id  text NOT NULL REFERENCES elements ON DELETE CASCADE,  -- the figure; goes with its document
    chart_no    int  NOT NULL CHECK (chart_no >= 1),  -- several charts can share one image (Rasi beside Navamsa)
    kind        text NOT NULL,                        -- 'Rasi chart', 'Navamsa (D-9) chart', ...
    lagna       text,                                 -- NULL when the chart marks no Lagna
    body        text NOT NULL,                        -- Sun ... Ketu, or Lagna
    sign        text NOT NULL,
    house       int CHECK (house BETWEEN 1 AND 12),   -- from the Lagna; NULL without one (never guessed)
    PRIMARY KEY (element_id, chart_no, body)
);
CREATE INDEX chart_placements_body_house ON chart_placements (body, house);
CREATE INDEX chart_placements_body_sign ON chart_placements (body, sign);
