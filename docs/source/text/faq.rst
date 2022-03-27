FAQ
===


    1. **Does autox support different time series lengths?**

       Yes, it supports different time series lengths. However, some feature calculators can demand a minimal length
       of the time series. If a shorter time series is passed to the calculator, a NaN is returned for those
       features.



    2. **Is it possible to extract features from rolling/shifted time series?**

       Yes, the :func:`autox.dataframe_functions.roll_time_series` function allows to conviniently create a rolled
       time series datframe from your data. You just have to transform your data into one of the supported autox
       :ref:`data-formats-label`.
       Then, the :func:`autox.dataframe_functions.roll_time_series` give you a DataFrame with the rolled time series,
       that you can pass to autox.
       On the following page you can find a detailed description: :ref:`forecasting-label`.


    3. **How can I use autox with windows?**

       We recommend to use `Anaconda <https://www.continuum.io/downloads#windows>`_. After installing, open the
       Anaconda Prompt, create an environment and set up autox
       (Please be aware that we're using multiprocessing, which can be `problematic <http://stackoverflow.com/questions/18204782/runtimeerror-on-windows-trying-python-multiprocessing>`_.):

       .. code:: Bash

           conda create -n ENV_NAME python=VERSION
           conda install -n ENV_NAME pip requests numpy pandas scipy statsmodels patsy scikit-learn tqdm
           activate ENV_NAME
           pip install autox


    4. **Does autox support different sampling rates in the time series?**

        Yes! The feature calculators in autox do not care about the sampling frequency.
        You will have to use the second input format, the stacked DataFramed (see :ref:`data-formats-label`)