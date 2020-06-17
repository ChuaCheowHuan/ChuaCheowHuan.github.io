---
layout: posts
author: Huan
title: Finding the `ray_results` folder in colab
---

The location of `ray_results` folder in colab when using RLlib &/or tune.

---

If you're NOT using `tune.run()` but using only RLlib's python API & if
you set the `local_dir` in RLlib's `config` to a non-default directory,
(The default is `~/ray_results`.) you will only find the `results.json` file
(& your checkpoint folders if you also specify them) in your specified
directory. The other files such as the tensorboard event files, the params.pkl
& params.json will still be saved in the default directory.

However, if you use `tune.run()` & set the `local_dir` argument to your
specified directory, all the files will be saved there.

If you're using colab, the way to access `~/ray_results` is to click on a small
folder icon on the left of the panel, it will open up a side panel, simply go to
`root/ray_results`. All those files will be saved in there.

Note: It seems like setting the (1)`local_dir` in RLlib's `config` will not
automatically set the (2)`local_dir` in the `Experiment` class in `experiment.py`
if you're not using `tune.run()`.

---

<br>
