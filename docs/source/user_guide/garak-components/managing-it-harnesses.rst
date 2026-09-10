
🏇 Managing it: harnesses
=========================

Harnesses determine which detectors and which probes will be run. They connect the generator and probes, and conduct the actual vulnerability scan through the detectors.

There are two important harnesses in garak:


* ``probewise``\ , the default harness, uses the detectors recommended by each individual probe
* ``pxd``\ , or probes x detectors, runs all specified probes with all specified detectors
