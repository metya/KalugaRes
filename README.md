# PreAct ResNet with some tests on cifar10

To run train without tests just do

```shell script
python main.py --train --epochs num
```

To run train alongside with some unit tests just do 
```shell script
python main.py --test --train --epochs num
``` 

If you want to run just tests, then do
```shell script
python main.py --test
```

### TODO

- Regression metrics tests
- Gradient explosion and gradient vanishing check
- Pythonistic Unit Tests