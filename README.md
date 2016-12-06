Hooks
=====

In order to build the documenation automatically with every commit, please add the provided hooks to your git workflow by executing

```
ln -s hooks/post-commit.sh .git/hooks/post-commit
ln -s hooks/pre-commit.sh .git/hooks/pre-commit
```
in the root of the repositiory. This will add an automatic build of the documentation to any commit. If you want to commit without that (because the build fails due to missing dependencies), run
```
git commit --no-verify
```
to skip this step.


