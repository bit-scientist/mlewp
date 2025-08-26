# Git Workflow Cheat Sheet  
### Fork, Fix, Pull Request, Sync — ML Engineering with Python

---

## 🔁 Recommended Workflow
**Sync → Branch → Fix → Push → PR → Merge to your main**

---

## 🔧 Step 1: Set Up Remotes (One Time)
```
git remote add upstream https://github.com/PacktPublishing/Machine-Learning-Engineering-with-Python-Second-Edition.git
git remote -v
```

🔄 Step 2: Sync Your main with Upstream
```
git checkout main
git fetch upstream
git merge upstream/main
git push origin main
```
✅ Start fresh with latest official code. 


## 📌 Step 3: Create a Feature Branch
```
git checkout -b fix/ch01_modifications
```


✅ Never work on main directly. 

## 🛠️ Step 4: Make & Commit Changes
```
git add .
git commit -m "Fix forecast plot CI display and improve plotting robustness"
```
✅ Keep commits focused and descriptive. 

## Step 5: Push to Your Fork
```
git push -u origin fix/ch01_modifications
```
✅ Keep commits focused and descriptive. 

Step 5: Push to Your Fork
```
git push -u origin fix/ch01_modifications
```
✅ -u sets tracking. Now visible on GitHub. 

## 🌐 Step 6: Open Pull Request (PR)
1. Go to: https://github.com/bit-scientist/mlewp
2. Click "Compare & pull request"
3. Set:
        Base repo: PacktPublishing/...
        Base: main
        Head: fix/ch01_modifications
4. Write a clear description
5. Click Create Pull Request
✅ You’ve contributed! 

## Step 7: Merge Fix into Your main
```
git checkout main
git merge fix/ch01_modifications
git push origin main
```
✅ Keeps your fork improved — even if PR isn’t merged. 

## 🔁 Step 8: Future Syncing
```
git checkout main
git fetch upstream
git merge upstream/main
git push origin main
```
✅ Resolves conflicts early. Keeps your fork current. 

🚫 Don’t Do This

    ❌ Edit main directly
    ❌ Push to upstream
    ❌ Skip syncing before new work
    ❌ Ignore PR hygiene\n

💡 Pro Tips
Private email:
```
git config user.email "12345678+sokhib@users.noreply.github.com"
```

Better editor:
```
git config --global core.editor "nano" or "code -w"
```

Shortcut alias:
```
git config --global alias.update-main '!git fetch upstream && git checkout main && git merge upstream/main && git push origin main'
```
Then run: `git update-main`









