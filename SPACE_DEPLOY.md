# Deploying to Hugging Face Spaces (Quill-AI)

This file lists exact steps to publish this repository to a Hugging Face Space that uses Docker. It also contains guidance for handling large files via Git LFS.

1) Prepare the repo
- Ensure `.gitattributes` exists (this repo includes one to track common model/binary patterns).
- Ensure `.gitignore` excludes local caches and large folders like `/checkpoints` and `/tmp`.

2) Create a Space (if not already)
- On Hugging Face, create a new Space: https://huggingface.co/new-space
- Choose the 'Docker' option so the `Dockerfile` in this repo is used.

3) Push the repo branch to the Space remote
- Add the HF remote (if not added already):
```powershell
git remote add hf https://huggingface.co/spaces/<your-username>/<space-name>.git
```
- Push a branch (safe):
```powershell
git push hf prep/spaces-publish
```

4) If push is rejected for file-size reasons
- Either remove large files from history (we did this on the `prep/spaces-publish` branch) or use Git LFS.
- To migrate large files into LFS (recommended if you need to keep them), run:
```powershell
git lfs install
git lfs track "textbooks/*.pdf"
git add .gitattributes
git lfs migrate import --include="textbooks/*.pdf,*.bin,checkpoints/**"
git push --force hf prep/spaces-publish
```

5) After push
- Open the Space in a browser and check Files. If using Docker, HF will build the image using the `Dockerfile`.
- If the repo card warns about missing metadata, add a small `README.md` or a `README.md` and optionally a `space.yaml` / model card.

6) Troubleshooting
- If git push is rejected even after auth: ensure `hf auth login` was run and that the token has repo/space write permissions.
- On Windows, install Git Credential Manager: `winget install --id GitCredentialManager.GitCredentialManager -e` and then re-run `hf auth login`.

Contact
- If you want me to migrate large files to LFS or force-push cleaned branches to other remotes, say so and I'll do it.
