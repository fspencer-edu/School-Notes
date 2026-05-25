```python
# setup

git config --global user.name "firstname lastname"

git config --global user.email "valid-email"

git config --global color.ui auto

# init
git init
git clone [url]

# stage and snapshot

git status
git add [file]
git reset [file]
git diff
git diff --staged
git commit m "message"

# branch and merge
git branch
git branch [branch-name]
git checkout
git merge [branch]
git log

# inspect
git log
git log branchA..branchB
git log --follow [file]
git diff branchB..branchA
git show [SHA]

# trakcing path changes
git rm [file]
git mv [path][new-path]
git log --stat -M

# share and update
git remote add [alias][url]
git fetch [alias]
get merge [alias]/[branch]
get push [alias][branch]
git pull

# rewrite history
get rebase [branch]
get reset --hard [commit]

# temporary commits
git stash
git stash list
git stash pop
git stash drop

# ignoring patterns
git config --global core.excludesfile [file]
```