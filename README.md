# thinking-space
assortment of codes that I'm messing around with for practice/brainstorming


this repository will be (nearly unavoidably) messy and disorganized since it consists of a variety of projects rather than a specific topic

to make things a bit easier:
-all .py script names begin with "PY", so alphabetically they all appear together

-the names of .csv files should be descriptive enough to convey what's in them, but if you have questions let me know (most of them have just been copied from Caleb's repository)

-there is a separate folder containing all of the plots that have been produced, so they're out of the way



current issues to resolve:
-PY_DMCoreMass.py --> this code can be used to produce plots for sub-GeV m_chi vs. M_DM and runs without errors, though it isn't efficient. it can't successfully produce plots for m_chi vs. M_DM in the WIMP mass range, specifically there is an error occuring in line 778 (tau_eq = (C * Ca)** (1/2)), where Ca (annihilation coefficient) is equalling zero after the function has been called a few dozen times or so

-PY_effective_temp --> this succesfully calculates the effective temperature for pop III stars and creates plots for mass vs. effective temp and Eddington luminosity vs. effective temp. the plots look super weird, but Ellie (and I) will be looking into effective temp more closely, so I'll compare and update this later
