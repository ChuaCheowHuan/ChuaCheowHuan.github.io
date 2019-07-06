---
layout: posts
author: Huan
title: .bash_profile for Mac
---
This post demonstrates how to create customized functions to bundle commands in
a .bash_profile file on Mac.

Edit .bash_profile for Mac.

1. Start Terminal
2. Enter "cd ~/" to go to home folder
3. Edit .bash_profile with "open -e .bash_profile" to open in TextEdit.
4. Enter ". .bash_profile" to reload .bash_profile.

---

## Examples

To bundle common git operations, add the following to .bash_profile file:

```
function lazy_git() {
    git checkout test_ver
    git add .
    git commit -a -m "$1"
    git checkout master
    git merge test_ver
    git push
    git checkout test_ver
}
```
---

To bundle common jekyll operations, add the following to .bash_profile file:

The command ```serve``` runs localhost.

```
function lazy_jekyll_serve() {
    cd /Users/tester/gitHubRepo/ChuaCheowHuan.github.io
    pwd
    bundle exec jekyll serve
}
```

The command ```build``` build the site. This command is neccessary for
generating sitemap.xml & robot.txt.

```
function lazy_jekyll_build() {
    cd /Users/tester/gitHubRepo/ChuaCheowHuan.github.io
    pwd
    bundle exec jekyll build
}
```

---

<br>
