# DataPrepper for Machine Learning

### Disclaimer:
This package is still in work and just written to further understand the techniques of inspecting and adjusting data methods for machine learning.

However, it should not be part of the final pipeline and the actual modelling process of any algorithm. Its sole function is, to
exploratory learn about data and experimentally dive deeper into the field of machine learning.

The code and functionality in this API should not be part of any grading process, since it is only in place to carry out "helper-functions" and keep the code less complicated and more clean, but does not tell anything about the functionality and model building process itself. The API only works as supposed if correctly used by any user.

My motivation in writing this API was, that I do not like the way pipelines "hide" their functionality and what they are actually doing to the data "under the hood". So I wanted to write and understand the functions myself, so I can better interpret the pipeline approach in machine learning.

## Improvements

In the future, I want to program the functions in a way, so that they are statically callable.
However, that will be done after the semester (it is not critical for the package to work sufficient for what we have to do in class).

Since scaling and encoding functions are represented in pipeline schemes, evaluation and inspection are not. So in future this script will be split into 2 scripts: One for datascaling and training and one for evaluation and inspection.

Furthermore, I want to implement some safety guards. At the moment, it solely depends on the user whether the script is being used correctly or not. So I want to think about robustness and security a bit more and optimize my classes. At the moment it somehow seems like everything is just "glued" together and can fall apart any moment (haha).

Last but not least: There are still a lot of functionalities missing. This is caused by a "when needed" approach. Meaning, whenever I needed a function in past assignments I tried to add it to the API. So that will also be updated after the semester at the last. For example: The Scaler-Section does not have an entry for the "Normalization()".

## Installation

The used python version of the script is **3.13.7**. <br>
For installment, simply run
```
pip install <*.whl>
```
Fill in the name of the correct wheel file, if you have more than one .whl in your directory.