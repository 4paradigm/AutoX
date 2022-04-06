====
API
====

.. toctree::
   :maxdepth: 1

   Feature engineer <./feature_engineer>
   Feature selection <./feature_selection>
   Metrics <./metrics>

.. list-table:: Overview of feature engineer API
   :widths: 15 30
   :header-rows: 1

   * - operation
     - description
   * - count
     - count the occurrences of some categorical features within dataset.
   * - cumsum
     - the calculation of the cumulative sum.
   * - denoising autoencoder
     - train a denoising autoencoder neural network for feature extraction. `reference <https://towardsdatascience.com/denoising-autoencoders-explained-dbb82467fc2>`_
   * - dimension reduction
     - use dimension reduction technology for feature extraction, such as Principal Component Analysis (PCA).
   * - gbdt
     - Generating Features with Gradient Boosted Decision Trees. `reference <https://towardsdatascience.com/feature-generation-with-gradient-boosted-decision-trees-21d4946d6ab5>`_
   * - rank
     - Compute numerical data ranks.
   * - rolling
     - statistics calculation within rolling windows.
   * - shift
     - lag feature.
   * - diff
     - first Difference.
   * - statistics
     - statistics calculation.
   * - time parse feature
     - parse time feature for time column, such as year, month, day, hour, dayofweek, and so on.
   * - cross feature
     - synthetic feature formed by multiplying (crossing) two or more features. `reference <https://developers.google.com/machine-learning/crash-course/feature-crosses/video-lecture?hl=zh_cn>`_


.. list-table:: Overview of feature selection API
   :widths: 15 30
   :header-rows: 1

   * - operation
     - description
   * - Adversarial Validation
     - a feature selection solution for battling overfitting. `reference <https://towardsdatascience.com/adversarial-validation-ca69303543cd>`_
   * - GRN
     - a feature selection using Gated Residual Networks (GRN) and Variable Selection Networks (VSN). `reference <https://keras.io/examples/structured_data/classification_with_grn_and_vsn/>`_

.. list-table:: Overview of Metrics API
   :widths: 15 30
   :header-rows: 1

   * - operation
     - description
   * - MAE
     - mean absolute error
   * - MAPE
     - mean absolute percentage error
   * - MSE
     - mean squared error
   * - MSLE
     - mean squared logarithmic error
   * - RMSLE
     - root mean squared logarithmic error