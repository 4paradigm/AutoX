FAQ
===


    1. **How can I use autox with windows?**

       We recommend to use `Anaconda <https://www.continuum.io/downloads#windows>`_. After installing, open the
       Anaconda Prompt, create an environment and set up AutoX
       (Please be aware that we're using multiprocessing, which can be `problematic <http://stackoverflow.com/questions/18204782/runtimeerror-on-windows-trying-python-multiprocessing>`_.):

       .. code:: Bash

           conda create -n ENV_NAME python=VERSION
           activate ENV_NAME
           pip install autox
