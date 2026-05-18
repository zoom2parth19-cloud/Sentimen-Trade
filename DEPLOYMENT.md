# SentimenTrade Deployment Guide

## 🚀 Deploy on GitHub Pages (Static Site)

Your project has been set up with:
- ✅ Interactive Streamlit dashboard (`dashboard/app.py`)
- ✅ Static documentation site (`docs/index.html`)
- ✅ GitHub Actions CI/CD workflow

### Option 1: GitHub Pages (Recommended for Static Content)

1. **Enable GitHub Pages:**
   - Go to: https://github.com/zoom2parth19-cloud/Sentimen-Trade/settings/pages
   - Select: **Deploy from a branch**
   - Branch: **main**
   - Folder: **/docs**
   - Click **Save**

2. **Your site will be live at:**
   ```
   https://zoom2parth19-cloud.github.io/Sentimen-Trade
   ```

### Option 2: Deploy Streamlit App (For Interactive Dashboard)

Since Streamlit apps are Python-based, they can't run on GitHub Pages (static hosting).
Choose one of these alternatives:

#### **A. Railway (Recommended)**
```bash
# Install Railway CLI
npm install -g @railway/cli

# Deploy
railway login
railway init
railway up
```

#### **B. Replit (Fastest)**
1. Go to https://replit.com
2. Click "Import Repository"
3. Paste: `https://github.com/zoom2parth19-cloud/Sentimen-Trade`
4. It auto-runs with a public URL

#### **C. Heroku (Classic)**
```bash
# Install Heroku CLI
heroku login
heroku create sentimentrade
git push heroku main
```

### Option 3: GitHub Actions (Testing/CI Only)

The `.github/workflows/deploy.yml` file runs:
- ✅ Dependency installation
- ✅ Unit tests on every push
- ✅ Code validation

## 📋 File Structure After Deployment

```
SentimenTrade/
├── .github/workflows/deploy.yml    # CI/CD pipeline
├── Procfile                         # Deployment config
├── setup.sh                         # Environment setup
├── docs/index.html                  # Static landing page
├── dashboard/app.py                 # Interactive Streamlit app
├── src/                             # Python modules
├── tests/                           # Unit tests
└── requirements.txt                 # Dependencies
```

## ✅ Quick Steps

1. **For Static Site (Easiest):**
   - Enable GitHub Pages in settings
   - Live at: `zoom2parth19-cloud.github.io/Sentimen-Trade`

2. **For Interactive App:**
   - Use Railway.app or Replit.com
   - Deploy in < 5 minutes

3. **For Testing/CI:**
   - GitHub Actions automatically runs on every commit
   - Check the "Actions" tab for status

---

Need help? Check the README.md for more details!
