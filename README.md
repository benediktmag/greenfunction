# Numerical calculations for the global extremal function

The global extremal function for a set $U\subset ℂ^n$ and a function $q:U\to (-\infty,+\infty]$ is defined as

$$V_{U,q}(z)=\sup\\{u(z)\colon u\in ℒ, u|_U\leq q\\}$$

where $ℒ$ is the Lelong class which consists of those plurisubharmonic functions $u$ on $ℂ^n$ that fulfill

$$u(z)-\log|z|\leq  O(1),\quad\text{when } |z|\to\infty.$$

It is known that under certain conditions, if $\\{p_j\\}$ is an orthonormal basis for the space of polynomials on U weighted relative to $q$, then

$$\lim_{n\to\infty} \frac{1}{2n}\log\left(\sum_{j=1}^n p_j(z)\overline{p_j(z)}\right)=V_{U,q}(z)$$.

This code is intended to numerically approximate the global extremal function for subsets of $ℂ$ by this limit.

## Usage

The package needs `python 3`, `matplotlib` and `numpy`. For those using _Conda_ there is an `environment.yml` file you can use to create an environment:

```
conda env create -f environment.yml
conda activate greenfunction
pip install -e .
```

To run the program run

```
python src/greenfunction/main.py
```
Use the `-h` flag to see avaiable options.