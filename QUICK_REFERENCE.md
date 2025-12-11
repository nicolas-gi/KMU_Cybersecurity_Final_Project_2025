# 🚀 Quick Reference: Automated Pipeline

## One-Time Setup (5 minutes)

```bash
# 1. Add SONAR_TOKEN to GitHub Secrets
# Go to: repo → Settings → Secrets and variables → Actions
# Add: SONAR_TOKEN = <your-token-from-sonarcloud.io>

# 2. Commit changes
git add .
git commit -m "feat: integrate SonarQube with GitHub Actions"
git push origin main

# 3. Monitor workflows
# Go to: repo → Actions tab
```

---

## Before Every Push

```bash
# Run pre-push checks
bash pre-push-check.sh

# Only push if checks pass ✅
git push
```

---

## Local Development Commands

```bash
npm run lint          # Check code style
npm run build         # Build project
npm run sonar         # Run SonarQube (needs token in .env.local)
npm run dev           # Start dev server
npm run ml:serve      # Start ML service
npm run ml:train      # Train ML model
```

---

## What Runs Automatically

| Trigger | Workflows | Time |
|---------|-----------|------|
| Push to main | All 4 | ~5-10 min |
| Push to develop | All 4 | ~5-10 min |
| Pull request | All 4 | ~5-10 min |

---

## View Results

```
GitHub Actions:  repo → Actions tab
SonarCloud:      sonarcloud.io → Your Project
```

---

## If Something Fails

| Issue | Solution |
|-------|----------|
| SonarQube "Not authorized" | Add SONAR_TOKEN to GitHub secrets |
| Build fails | Run `npm run build` locally |
| Lint errors | Run `npm run lint` and fix issues |
| Python errors | Run `bash setup.sh` to install venv |

---

## Files Created/Modified

✅ sonarqube-scanner installed  
✅ sonar-project.properties enhanced  
✅ package.json updated (sonar script added)  
✅ .env.local created  
✅ .github/workflows/sonarqube.yml created  
✅ .github/workflows/tests.yml created  
✅ .github/workflows/lint.yml created  
✅ pre-push-check.sh created  
✅ AUTOMATION_SETUP.md created  
✅ PIPELINE_CHECKLIST.md created  

---

## Key Configuration Values

```
Project Key: nicolas-gi_KMU_Cybersecurity_Final_Project_2025
Organization: nicolas-gi
Python Version: 3.8
Node.js: 18+ (tests 18.x, 20.x)
Quality Gate: Enabled (sonar.qualitygate.wait=true)
```

---

**CRITICAL: Don't forget to add SONAR_TOKEN to GitHub secrets!**
