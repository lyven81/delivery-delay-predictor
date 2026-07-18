# How to push this to GitHub

This folder is the refined `delivery-delay-predictor` app, ready to update
**https://github.com/lyven81/delivery-delay-predictor**.

## Option A — clone, copy, push (recommended)

```bash
# 1. clone the existing repo somewhere temporary
git clone https://github.com/lyven81/delivery-delay-predictor.git
cd delivery-delay-predictor

# 2. copy every file from this folder over the repo (replaces the old app)
#    (on Windows, copy the contents of this "github" folder into the cloned repo)

# 3. commit and push
git add -A
git commit -m "Refine: Vertex-tuned+calibrated model, live BigQuery, verified recs, two-tab UI"
git push origin main
```

## Option B — GitHub web upload
Open the repo on github.com → "Add file" → "Upload files" → drag in every file from
this folder → commit to `main`.

## Note on the large file
`model.pkl` is ~2 MB (fine for a normal git push; no Git LFS needed).

## After pushing
The repo is the source of record; the **live app is already deployed** and does not
re-deploy from GitHub automatically. If you want the repo and the live service to stay
in lock-step, redeploy from the pushed repo with the `gcloud run deploy` command in `README.md`.
