# Git Setup & Push Guide

## Current Status

✅ Git repository initialized  
✅ Initial commit created  
✅ All files staged and committed

## Push to Remote Repository

### Option 1: GitHub (Recommended)

1. **Create a new repository on GitHub**
   - Go to https://github.com/new
   - Repository name: `CARDIOGraph` (or your preferred name)
   - Description: "AI-powered knowledge graph for cardiotoxicity risk analysis"
   - Choose Public or Private
   - **DO NOT** initialize with README, .gitignore, or license (we already have these)
   - Click "Create repository"

2. **Add remote and push**
   ```bash
   cd CARDIOGRAPH
   
   # Add remote (replace YOUR_USERNAME with your GitHub username)
   git remote add origin https://github.com/YOUR_USERNAME/CARDIOGraph.git
   
   # Or if using SSH:
   git remote add origin git@github.com:YOUR_USERNAME/CARDIOGraph.git
   
   # Push to GitHub
   git branch -M main
   git push -u origin main
   ```

### Option 2: GitLab

1. **Create a new project on GitLab**
   - Go to https://gitlab.com/projects/new
   - Project name: `CARDIOGraph`
   - Visibility: Public or Private
   - **DO NOT** initialize with README
   - Click "Create project"

2. **Add remote and push**
   ```bash
   cd CARDIOGRAPH
   
   git remote add origin https://gitlab.com/YOUR_USERNAME/CARDIOGraph.git
   
   git branch -M main
   git push -u origin main
   ```

### Option 3: Other Git Hosting

Follow similar steps for other platforms (Bitbucket, Azure DevOps, etc.)

## Verify Remote Setup

Check your remote configuration:

```bash
git remote -v
```

You should see:
```
origin  https://github.com/YOUR_USERNAME/CARDIOGraph.git (fetch)
origin  https://github.com/YOUR_USERNAME/CARDIOGraph.git (push)
```

## Future Commits

After making changes:

```bash
# Check what changed
git status

# Add changes
git add .

# Commit
git commit -m "Description of changes"

# Push to remote
git push
```

## Important Notes

⚠️ **Never commit sensitive information:**
- `.env` file is already in `.gitignore` ✅
- Never commit Neo4j passwords
- Never commit API keys
- Never commit large data files

✅ **Safe to commit:**
- Source code
- Documentation
- Configuration templates (`.env.example`)
- Requirements files
- Notebooks (without sensitive outputs)

## Troubleshooting

### If you get "remote origin already exists"
```bash
git remote remove origin
git remote add origin YOUR_REPO_URL
```

### If you get authentication errors
- Use SSH keys instead of HTTPS
- Or use GitHub CLI: `gh auth login`

### If you need to change the remote URL
```bash
git remote set-url origin NEW_REPO_URL
```

