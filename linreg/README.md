# Linear regression

This directory contains 4 files : 
* **README.md** (me), explaning what you will do during this exercise
* **main.py**, the main code to execute
* **make_data.py**, the code that will generate random data along a straight line, data that we will
  use to train fit a linear regression.
* **regressor.py**, containing the class LinearRegressor that you will have to work on.

## What to do

Start by opening **make_data.py**, and try to understand it. Then run it :

```bash
cd linreg
python make_data.py
```
and open and inspect the produced file **data.csv**.

Then, open **main.py** and understand it. Do not run it, it would not work : the imported class *LinearRegressor*
is not finished

Then, open **regressor.py**. Spend some time on it to broadly understand what is in it. Then, you will
have to code the missing parts of the code (look for "Code here" in the file). That will take a while.

You can use the slides from the theoretical course to help you : https://fr.overleaf.com/read/pjmmfpntzkfq

If the code works, you should be able to run **main.py** without error and produce some graphs. 

Once this is done, you can either wait for everyone to reach this point, or start looking online about "normalising"
your data before a learning, and try to implement that.