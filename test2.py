class A:
    def __init__(self, value):
        self.__value = value
        
a = A(1)
a.__value = 2
print(a._A__value)