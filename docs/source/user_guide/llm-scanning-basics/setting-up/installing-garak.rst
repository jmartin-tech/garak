
😇 Installing garak
===================

So you'd like to install garak and scan some language models for vulnerabilities? Great! You're in the right place.&#x20;

garak runs from the command line / terminal, and works best on Linux and Mac OSX. If you're on windows, please give it a try, and let us know how you get on!

The easiest way to install garak is using Python's ``pip`` package manager. You need at least Python 3.10 to use garak.


.. raw:: html

   <details>

   <summary>Easy install with pip</summary>

   You can use this one-liner from a terminal and you should be good to go:

   ```
   python -m pip install -U garak
   ```

   Happy hunting!

   </details>



.. raw:: html

   <details>

   <summary>I don't have pip!</summary>

   No worries! You can still use garak. If you don't have pip, you can install it. There are guides for that here on [pypa.io](https://pip.pypa.io/en/stable/cli/pip_install/); here's a summary.&#x20;

   If you're on Mac, then installing and using [brew](https://brew.sh/) to install pip is strongly recommended - it doesn't run super quickly the first time, but the experience is really good after. Try something like:

   1. Open terminal
   2. Install brew using the instructions on [brew.sh](https://brew.sh/)
   3. Run `brew install python`
   4. To finalize the pip install, run `brew unlink python && brew link python`

   If you're on Linux, you probably want to start with a distribution package manager. Try:

   1. Install pip from package - [https://linuxconfig.org/install-pip-on-linux](https://linuxconfig.org/install-pip-on-linux)
   2. Upgrade your pip installation, with `python -m pip install -U pip`

   If you're on Windows, let us know how you get on!

   </details>



.. raw:: html

   <details>

   <summary>pip install for python version 3.9 or lower</summary>

   garak requires Python 3.10. If you don't have that, you can run garak in a conda container using a higher Python version, without disrupting everything else on your computer.&#x20;

   First, install conda - instructions are here, [https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html#regular-installation](https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html#regular-installation)

   Next, create a conda environment for garak; let's call it garak:

   ```
   conda create --name garak "python>=3.10,<=3.12"
   conda activate garak
   ```

   Finally, install garak:

   ```
   python -m pip install garak
   ```

   When you're done, you can run conda deactivate to get back to where you were.

   ```
   conda deactivate
   ```

   Any time you'd like to run garak, just remember to activate the conda environment first:

   If you get stuck, come chat with us on [Discord](https://discord.gg/uVch4puUCs)!

   ```
   conda activate garak
   ```

   </details>



.. raw:: html

   <details>

   <summary>Install the latest development version with pip</summary>

   The standard pip version of `garak` is updated periodically. To get the freshest version possible, from [GitHub](https://github.com/leondz/garak/), try:

   ```
   python -m pip install -U git+https://github.com/leondz/garak.git@main
   ```

   </details>

