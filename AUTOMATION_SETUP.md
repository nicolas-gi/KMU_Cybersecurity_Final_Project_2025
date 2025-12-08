# Automated Pipeline Setup Guide

This document outlines the complete automated pipeline setup for the Network Anomaly Detection System, including SonarQube integration, testing, linting, and code quality checks.

## 📋 What's Been Set Up

### 1. **SonarQube Integration** ✅
- **Configuration File:** `sonar-project.properties`
- **Scanner:** `sonarqube-scanner` npm package
- **Command:** `npm run sonar`

### 2. **GitHub Actions Workflows** ✅
Four automated workflows are configured:

#### a) **SonarQube Analysis** (`sonarqube.yml`)
- Triggers on: Push to main/develop, Pull requests
- Analyzes code quality with SonarCloud
- Enforces quality gate checks
- **Requirements:** `SONAR_TOKEN` secret

#### b) **Testing Pipeline** (`tests.yml`)
- Triggers on: Push to main/develop, Pull requests
- Tests Node.js 18.x and 20.x compatibility
- Builds Next.js project
- Tests Python ML service
- Security vulnerability scans
- **No requirements:** Uses built-in tools

#### c) **Code Quality & Linting** (`lint.yml`)
- Triggers on: Push to main/develop, Pull requests
- ESLint for JavaScript/TypeScript
- TypeScript type checking
- Prettier code formatting
- Python linting (Flake8, Pylint)
- **No requirements:** Uses built-in tools

#### d) **Build Workflow** (`build.yml`)
- Legacy workflow for SonarQube scanning
- Can be kept for redundancy

---

## 🔧 Prerequisites & Setup Steps

### Step 1: GitHub Secrets Configuration ⚠️ CRITICAL

You **MUST** add the `SONAR_TOKEN` secret to your GitHub repository:

1. Go to your GitHub repository
2. Navigate to **Settings → Secrets and variables → Actions**
3. Click **New repository secret**
4. Add:
   - **Name:** `SONAR_TOKEN`
   - **Value:** Your SonarQube token from [SonarCloud](https://sonarcloud.io/account/security)

> **How to get your SonarQube token:**
> 1. Log in to [sonarcloud.io](https://sonarcloud.io)
> 2. Click your avatar → **My Account → Security**
> 3. Generate a new token
> 4. Copy and paste into GitHub secret

### Step 2: Verify Local Environment ✅

All required dependencies are already installed:

```bash
# Check SonarQube scanner
npm list sonarqube-scanner

# Check Node.js version
node --version  # Should be 18+

# Check Python version
python3 --version  # Should be 3.8+
```

### Step 3: Environment Configuration ✅

`.env.local` file is created with:
- `ML_API_URL=http://localhost:5000`
- `NEXT_PUBLIC_API_URL=http://localhost:3000`

---

## 🚀 Running Workflows Locally

### Test Next.js Build
```bash
npm run build
```

### Run Linting
```bash
npm run lint
```

### Run SonarQube Analysis (Local)
```bash
npm run sonar -- -Dsonar.login=<your-token>
```

### Run All Tests
```bash
npm run test  # (if tests are configured)
```

---

## 📊 Workflow Execution Flow

### On Every Push to `main` or `develop`:

1. **Testing Pipeline** starts immediately
   - Builds Next.js project
   - Tests Python ML service
   - Checks dependencies

2. **Code Quality & Linting** runs in parallel
   - ESLint checks
   - TypeScript validation
   - Python linting

3. **SonarQube Analysis** analyzes code
   - Code quality metrics
   - Security vulnerabilities
   - Code coverage (when available)
   - Quality gate check

### On Every Pull Request:

All above workflows run + GitHub checks status displayed in PR

---

## ⚠️ Important Considerations

### 1. **Quality Gate Failures**
- If SonarQube quality gate fails, you'll see it in PR checks
- Adjust quality gate rules in SonarCloud if needed
- Currently set to wait for quality gate: `sonar.qualitygate.wait=true`

### 2. **Python ML Service**
- Tests import `train_model` and `api` modules
- Uses Flake8 for code quality
- Ignores long lines (E501) and whitespace issues (W503)

### 3. **Node.js Compatibility**
- Tests against Node.js 18.x and 20.x
- Your project uses Next.js 16.0.3 (supports both)

### 4. **Ignored Directories**
SonarQube is configured to ignore:
- `node_modules/**`
- `__pycache__/**`
- `venv/**`
- `dist/**`
- `.next/**`

---

## 📈 Monitoring & Results

### GitHub Actions Dashboard
- Go to **Actions** tab in your repository
- View workflow runs and logs
- See success/failure status
- Download artifacts (ESLint reports)

### SonarQube Dashboard
- Go to [SonarCloud](https://sonarcloud.io)
- Find your project: `nicolas-gi_KMU_Cybersecurity_Final_Project_2025`
- View:
  - Code quality metrics
  - Security hotspots
  - Code coverage (when enabled)
  - Historical trends

---

## 🔍 Troubleshooting

### Workflow Fails: "SonarQube answered with Not authorized"
- ✅ Verify `SONAR_TOKEN` is set in GitHub secrets
- ✅ Token must be a **User token**, not a project token
- ✅ Token must be valid and not expired

### Next.js Build Fails
- Check for TypeScript errors: `npx tsc --noEmit`
- Check ESLint issues: `npm run lint`
- Review error logs in GitHub Actions

### Python Tests Fail
- Verify requirements are installed: `pip install -r ml-service/requirements.txt`
- Check Python version: `python3 --version`
- Test imports locally: `python3 -c "import train_model"`

### ESLint Reports Empty
- Ensure eslint configuration is correct: `eslint.config.mjs`
- Check file patterns match your project structure
- Run locally: `npm run lint`

---

## ✅ Verification Checklist

- [ ] `SONAR_TOKEN` added to GitHub secrets
- [ ] `.env.local` file exists
- [ ] `sonarqube-scanner` installed: `npm list sonarqube-scanner`
- [ ] `sonar-project.properties` configured with correct project key
- [ ] All workflow files in `.github/workflows/`: sonarqube.yml, tests.yml, lint.yml, build.yml
- [ ] `.gitignore` excludes node_modules and Python cache
- [ ] Local build works: `npm run build`
- [ ] Local lint works: `npm run lint`
- [ ] Push to main branch to trigger workflows

---

## 🎯 Next Steps

1. **Add SONAR_TOKEN to GitHub Secrets** (MUST DO)
2. **Commit and push these changes** to trigger workflows
3. **Monitor GitHub Actions tab** for results
4. **Check SonarCloud dashboard** for detailed analysis
5. **Fix any issues found** and push updates

---

## 📚 Resources

- [SonarCloud Documentation](https://docs.sonarcloud.io/)
- [GitHub Actions Documentation](https://docs.github.com/en/actions)
- [ESLint Configuration](https://eslint.org/docs/latest/use/configure)
- [Next.js Documentation](https://nextjs.org/docs)
