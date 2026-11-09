# data/

Only two small files are committed. Everything else comes from the gated Hugging Face dataset
(`c2r hub pull-dataset`) or from running the pipeline yourself.

| path | committed | produced by | description |
|---|:-:|---|---|
| `images/` | no | `c2r collect` / pull | `<YYYYmmdd_HHMMSS>.jpg` |
| `labels/rucaptcha.csv` | no | `c2r label consolidate` | raw + normalized paid human labels |
| `labels/llm.csv` | no | Claude | second-opinion labels annotated by Claude, `high`/`low` confidence (weak: 78.3% exact) |
| `corrections.csv` | **yes** | you, by hand | `filename,before,after,stage,note`, one row per fix |
| `splits.csv` | **yes** | `c2r data merge` | frozen `filename,split` assignment |
| `labels.csv` | no | `c2r data merge` | the training source of truth |

## Fixing a label

1. Look at the image and decide on the right label.
2. Append a row to `corrections.csv`: `before` = the current label, `after` = the right one
   (empty = drop the image), plus a `stage` and a `note`.
3. Run `c2r data merge`. It fails if `before` doesn't match the current label, so a fix can never
   silently overwrite a re-labeled source.

Raw input files are never edited. Every change goes through `corrections.csv`, so the full history
stays reviewable in git.
