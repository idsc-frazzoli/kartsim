# Validation Pipeline

# Setting up Matlab for Python Environment

- In Matlab enter: 
```ruby 
matlabroot 
``` 
- take the returned rootPath and in a system console enter:
```ruby
cd /rootPath/extern/engines/python
python setup.py install
```
  - if you are using Anaconda use the following command, where anacondaPath is the path to your anaconda directory in your environment
  ```ruby
  python setup.py install --prefix="anacondaPath"
  ```
Then go to 
https://ch.mathworks.com/help/matlab/matlab_external/get-started-with-matlab-engine-for-python.html 
for further instructions
